"""Shared infrastructure for the SB3 PPO and Nevergrad NGOpt examples.

Use --launch-server to manage a local Pokemon Showdown server automatically.
NGOpt assigns one team to every environment and replaces the population after
each PPO rollout. One controller learns continuously against a selectable
baseline whose opponent team is regenerated every battle.
"""

import argparse
import json
import os
import signal
import socket
import subprocess
import time
from contextlib import contextmanager
from functools import lru_cache
from pathlib import Path
from threading import Lock

import nevergrad as ng
import numpy as np
import torch
from poke_env.data import GenData
from poke_env.data.normalize import to_id_str
from poke_env.ps_client import AccountConfiguration
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CallbackList
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecEnv, VecMonitor

from pokemon_codesign_env.configs import (
    FormatConfig,
    ObservationConfig,
    TeamSearchConfig,
)
from pokemon_codesign_env.baseline_adapters import (
    TeamDesignVectorAdapter,
    compile_candidates,
    evaluate_candidate_population,
    make_all_attribute_search_config,
)
from pokemon_codesign_env.baseline_artifacts import RunMetrics, write_run_artifacts
from pokemon_codesign_env.envs import (
    MaxBasePowerPlayerEnv,
    RandomPlayerEnv,
    SimpleHeuristicsPlayerEnv,
)
from pokemon_codesign_env.envs.base import BaselineOpponentEnv
from pokemon_codesign_env.evaluation import evaluate_isolated
from pokemon_codesign_env.observations import HumanVisibleBattleEncoder
from pokemon_codesign_env.scaling import (
    ShowdownClusterConfig,
    managed_showdown_cluster,
)
from pokemon_codesign_env.sharded_vec_env import ShardedVecEnv
from pokemon_codesign_env.team_pool import RandomTeamPool
from pokemon_codesign_env.team_designer import TeamDesigner

TEAM_SIZE = 6
OBSERVATION_CONFIG = ObservationConfig(team_size=TEAM_SIZE)
INVALID_CANDIDATE_LOSS = 1_000_000.0
SHOWDOWN_HOST = "127.0.0.1"
SHOWDOWN_PORT = 8000
SERVER_STARTUP_TIMEOUT = 120.0
ENVIRONMENTS = {
    "random": RandomPlayerEnv,
    "max-base-power": MaxBasePowerPlayerEnv,
    "simple-heuristics": SimpleHeuristicsPlayerEnv,
}
EVALUATION_OPPONENT_NAMES = {
    RandomPlayerEnv: "RandomPlayer",
    MaxBasePowerPlayerEnv: "MaxPowerPlayer",
    SimpleHeuristicsPlayerEnv: "SimpleHeuristic",
}


def begin_startup_stage(args: argparse.Namespace, wandb_logger, stage: str, **metrics) -> float:
    """Report the beginning of a potentially slow startup stage."""
    if args.verbose:
        details = " ".join(f"{name}={value}" for name, value in metrics.items())
        print(
            f"startup_stage={stage} status=starting{f' {details}' if details else ''}", flush=True
        )
    wandb_logger.log_system(f"{stage}_starting", **metrics)
    return time.monotonic()


def finish_startup_stage(
    args: argparse.Namespace, wandb_logger, stage: str, started_at: float, **metrics
) -> None:
    """Report completion and elapsed time for one startup stage."""
    duration = time.monotonic() - started_at
    if args.verbose:
        details = " ".join(f"{name}={value}" for name, value in metrics.items())
        print(
            f"startup_stage={stage} status=ready duration_s={duration:.3f}{f' {details}' if details else ''}",
            flush=True,
        )
    wandb_logger.log_system(f"{stage}_ready", startup_stage_s=duration, **metrics)


@contextmanager
def startup_stage(args: argparse.Namespace, wandb_logger, stage: str, **metrics):
    """Time and report a synchronous startup stage."""
    started_at = begin_startup_stage(args, wandb_logger, stage, **metrics)
    try:
        yield
    except Exception:
        if args.verbose:
            print(
                f"startup_stage={stage} status=failed duration_s={time.monotonic() - started_at:.3f}",
                flush=True,
            )
        wandb_logger.log_system(
            f"{stage}_failed", startup_stage_s=time.monotonic() - started_at, **metrics
        )
        raise
    finish_startup_stage(args, wandb_logger, stage, started_at, **metrics)


class InitialResetCallback(BaseCallback):
    """Report when SB3 finishes its implicit first vector reset."""

    def __init__(self, args: argparse.Namespace, wandb_logger, started_at: float):
        super().__init__()
        self.args = args
        self.wandb_logger = wandb_logger
        self.started_at = started_at

    def _on_training_start(self) -> None:
        finish_startup_stage(
            self.args,
            self.wandb_logger,
            "initial_reset",
            self.started_at,
            environments=self.model.n_envs,
        )

    def _on_step(self) -> bool:
        return True


class GPUStatsCallback(BaseCallback):
    """Periodically report CUDA utilization and allocator memory."""

    def __init__(
        self,
        log_interval: int,
        wandb_logger=None,
        label: str | None = None,
        evaluation: int | None = None,
        round_index: int | None = None,
    ):
        super().__init__()
        self.log_interval = log_interval
        self._next_log = log_interval
        self.wandb_logger = wandb_logger
        self.label = label
        self.evaluation = evaluation
        self.round_index = round_index

    def _on_training_start(self) -> None:
        device = self.model.device
        if device.type != "cuda":
            print(f"training_device={device} gpu_profiling=unavailable")
            return
        properties = torch.cuda.get_device_properties(device)
        print(
            f"training_device={device} gpu_name={properties.name} gpu_memory_mib={properties.total_memory / 2**20:.0f}"
        )

    def _on_step(self) -> bool:
        if self.num_timesteps < self._next_log:
            return True
        self._next_log += self.log_interval
        device = self.model.device
        if device.type != "cuda":
            return True
        utilization = None
        if hasattr(torch.cuda, "utilization"):
            try:
                utilization = float(torch.cuda.utilization(device))
            except Exception:
                pass
        allocated = torch.cuda.memory_allocated(device) / 2**20
        reserved = torch.cuda.memory_reserved(device) / 2**20
        peak = torch.cuda.max_memory_allocated(device) / 2**20
        print(
            f"gpu_device={device} timesteps={self.num_timesteps} "
            f"utilization={'unavailable' if utilization is None else f'{utilization:.0f}%'} memory_mib={allocated:.0f} "
            f"reserved_mib={reserved:.0f} peak_mib={peak:.0f}"
        )
        if self.wandb_logger is not None:
            metrics = {
                "gpu/memory_mib": allocated,
                "gpu/reserved_mib": reserved,
                "gpu/peak_mib": peak,
            }
            if utilization is not None:
                metrics["gpu/utilization_percent"] = utilization
            self.wandb_logger.log_training(
                self.label, self.evaluation, self.num_timesteps, metrics, self.round_index
            )
        return True


class WandbRLCallback(BaseCallback):
    """Stream SB3 rollout and optimization statistics to the experiment run."""

    def __init__(self, wandb_logger, label: str, evaluation: int, round_index: int | None = None):
        super().__init__()
        self.wandb_logger = wandb_logger
        self.label = label
        self.evaluation = evaluation
        self.round_index = round_index
        self.started_at = None
        self.completed_episodes = 0
        self.recovered_environments = 0
        self.restarted_workers = 0
        self._next_telemetry = None
        self._last_telemetry_at = None
        self._last_telemetry_timesteps = 0

    def _on_training_start(self) -> None:
        self.started_at = time.monotonic()
        self._last_telemetry_at = self.started_at
        self._next_telemetry = self.model.n_envs
        self.wandb_logger.log_system(
            "rollout_started",
            label=self.label,
            candidate=self.evaluation,
            ready_environments=self.model.n_envs,
            rollout_size=self.model.n_envs * self.model.n_steps,
            round_index=self.round_index,
        )

    def _log(self) -> None:
        metrics = {}
        for name, value in self.model.logger.name_to_value.items():
            if isinstance(value, (int, float, np.integer, np.floating)):
                metrics[f"rl/{name}"] = float(value)
        self.wandb_logger.log_training(
            self.label, self.evaluation, self.num_timesteps, metrics, self.round_index
        )

    def _on_step(self) -> bool:
        dones = self.locals.get("dones")
        if dones is not None:
            self.completed_episodes += int(np.sum(dones))
        infos = self.locals.get("infos") or ()
        recovered_infos = [info for info in infos if info.get("sharded_worker_restarted")]
        if recovered_infos:
            self.recovered_environments += len(recovered_infos)
            self.restarted_workers += len(
                {info["sharded_worker_index"] for info in recovered_infos}
            )
        interval = self.wandb_logger.telemetry_log_interval
        if self.num_timesteps >= self._next_telemetry:
            now = time.monotonic()
            elapsed = max(now - self.started_at, 1e-9)
            interval_elapsed = max(now - self._last_telemetry_at, 1e-9)
            fps = (self.num_timesteps - self._last_telemetry_timesteps) / interval_elapsed
            rollout_size = self.model.n_envs * self.model.n_steps
            metrics = {
                "rl/rollout/vector_steps": self.n_calls,
                "rl/rollout/transitions_per_second": self.num_timesteps / elapsed,
                "rl/rollout/fps": fps,
                "rl/rollout/average_fps": self.num_timesteps / elapsed,
                "rl/rollout/completed_episodes": self.completed_episodes,
                "rl/rollout/buffer_fill": ((self.num_timesteps - 1) % rollout_size + 1)
                / rollout_size,
                "rl/recovery/environments": self.recovered_environments,
                "rl/recovery/worker_restarts": self.restarted_workers,
            }
            self.wandb_logger.log_training(
                self.label, self.evaluation, self.num_timesteps, metrics, self.round_index
            )
            self._last_telemetry_at = now
            self._last_telemetry_timesteps = self.num_timesteps
            self._next_telemetry = self.num_timesteps + interval
        return True

    def _on_rollout_end(self) -> None:
        self._log()

    def _on_training_end(self) -> None:
        self._log()


class TrainingProgressCallback(BaseCallback):
    """Display transition progress with recent and average environment FPS."""

    def __init__(self, total_timesteps: int, label: str, position: int = 0):
        super().__init__()
        self.requested_timesteps = total_timesteps
        self.label = label
        self.position = position
        self.progress = None
        self.started_at = None
        self._last_measurement_at = None
        self._last_measurement_timesteps = 0

    def _on_training_start(self) -> None:
        from tqdm.auto import tqdm

        rollout_size = self.model.n_envs * self.model.n_steps
        total_timesteps = (
            (self.requested_timesteps + rollout_size - 1) // rollout_size
        ) * rollout_size
        self.started_at = time.monotonic()
        self._last_measurement_at = self.started_at
        self.progress = tqdm(
            total=total_timesteps,
            desc=self.label,
            unit="step",
            position=self.position,
            dynamic_ncols=True,
        )

    def _on_step(self) -> bool:
        increment = self.num_timesteps - self.progress.n
        if increment > 0:
            self.progress.update(increment)
        now = time.monotonic()
        measurement_elapsed = now - self._last_measurement_at
        if measurement_elapsed >= 1.0 or self.num_timesteps >= self.progress.total:
            fps = (self.num_timesteps - self._last_measurement_timesteps) / max(
                measurement_elapsed, 1e-9
            )
            average_fps = self.num_timesteps / max(now - self.started_at, 1e-9)
            self.progress.set_postfix(
                fps=f"{fps:,.0f}", average_fps=f"{average_fps:,.0f}", refresh=False
            )
            self._last_measurement_at = now
            self._last_measurement_timesteps = self.num_timesteps
        return True

    def _on_training_end(self) -> None:
        if self.progress is not None:
            self.progress.close()


class PeriodicEvaluationCallback(BaseCallback):
    """Run single-environment evaluation after configured PPO updates."""

    def __init__(self, interval_rollouts: int, total_rollouts: int, evaluator):
        super().__init__()
        self.interval_rollouts = interval_rollouts
        self.total_rollouts = total_rollouts
        self.evaluator = evaluator
        self.last_evaluation_rollout = 0

    def _on_step(self) -> bool:
        return True

    def _on_rollout_start(self) -> None:
        rollout_size = self.model.n_envs * self.model.n_steps
        completed_rollouts = self.num_timesteps // rollout_size
        should_skip = (
            completed_rollouts == 0
            or completed_rollouts >= self.total_rollouts
            or completed_rollouts % self.interval_rollouts
            or completed_rollouts == self.last_evaluation_rollout
        )
        if should_skip:
            return
        self.evaluator(self.model, completed_rollouts, self.num_timesteps)
        self.last_evaluation_rollout = completed_rollouts


def sample_team_population(
    optimizer, search_config: TeamSearchConfig, population_size: int, seed: int
):
    """Ask NGOpt for one valid packed team per environment."""
    (team_adapter,) = get_ngopt_parameter_choices(search_config)
    candidates = []
    parameters = []
    teams = []
    attempts = 0
    while len(teams) < population_size:
        needed = population_size - len(teams)
        proposed_candidates = [optimizer.ask() for _ in range(needed)]
        proposed_parameters = [
            decode_ngopt_parameter_values(candidate.value, team_adapter)
            for candidate in proposed_candidates
        ]
        unique_proposals = [
            (candidate, values)
            for candidate, values in zip(proposed_candidates, proposed_parameters, strict=True)
            if team_species_are_trackable(values, search_config.format.generation)
        ]
        for candidate, values in zip(proposed_candidates, proposed_parameters, strict=True):
            if not team_species_are_trackable(values, search_config.format.generation):
                optimizer.tell(candidate, INVALID_CANDIDATE_LOSS)
        if not unique_proposals:
            attempts += needed
            continue
        proposed_candidates, proposed_parameters = map(list, zip(*unique_proposals, strict=True))
        compiled_candidates = compile_candidates(
            team_adapter,
            search_config,
            proposed_parameters,
            [seed + attempts + index for index in range(len(proposed_parameters))],
        )
        for candidate, compiled in zip(proposed_candidates, compiled_candidates, strict=True):
            if compiled.valid:
                candidates.append(candidate)
                parameters.append(compiled.parameters)
                teams.append(compiled.packed_team)
            else:
                optimizer.tell(candidate, INVALID_CANDIDATE_LOSS)
        attempts += needed
    return candidates, parameters, teams


class RolloutTeamOptimizerCallback(BaseCallback):
    """Evaluate one NGOpt embodiment per environment and replace it every rollout."""

    def __init__(
        self,
        optimizer,
        search_config: TeamSearchConfig,
        candidates: list,
        parameters: list[dict],
        seed: int,
        total_timesteps: int,
        evaluation_episodes: int,
        incumbent_rescore_interval: int = 5,
        wandb_logger=None,
        label: str = "codesign",
        round_index: int | None = None,
    ):
        super().__init__()
        self.optimizer = optimizer
        self.search_config = search_config
        self.candidates = candidates
        self.parameters = parameters
        self.seed = seed
        self.total_timesteps = total_timesteps
        self.evaluation_episodes = evaluation_episodes
        self.incumbent_rescore_interval = incumbent_rescore_interval
        self.wandb_logger = wandb_logger
        self.label = label
        self.round_index = round_index
        self.rollout_index = 0
        self.rollout_returns = None
        self.best_return = float("-inf")
        self.best_parameters = None
        self.incumbent_return = None
        self.training_battles = 0

    def _on_rollout_start(self) -> None:
        self.rollout_returns = np.zeros(self.model.n_envs, dtype=np.float64)

    def _on_step(self) -> bool:
        self.rollout_returns += np.asarray(self.locals["rewards"], dtype=np.float64)
        self.training_battles += int(np.count_nonzero(self.locals["dones"]))
        return True

    def _on_rollout_end(self) -> None:
        # The controller is unchanged while all candidates receive the same number
        # of complete, deterministic evaluation battles.
        evaluation_returns = evaluate_candidate_population(
            self.model,
            self.training_env,
            self.evaluation_episodes,
            seed=self.seed + self.rollout_index * self.evaluation_episodes,
        )
        for candidate, parameters, evaluation_return in zip(
            self.candidates, self.parameters, evaluation_returns, strict=True
        ):
            self.optimizer.tell(candidate, -float(evaluation_return))
            if evaluation_return > self.best_return:
                self.best_return = float(evaluation_return)
                self.best_parameters = parameters
        if self.wandb_logger is not None:
            self.wandb_logger.log_training(
                self.label,
                self.rollout_index,
                self.num_timesteps,
                {
                    "ngopt/rollout": self.rollout_index + 1,
                    "ngopt/population_size": self.model.n_envs,
                    "ngopt/mean_rollout_return": float(np.mean(self.rollout_returns)),
                    "ngopt/mean_evaluation_return": float(np.mean(evaluation_returns)),
                    "ngopt/best_return": self.best_return,
                    "ngopt/evaluation_episodes_per_candidate": self.evaluation_episodes,
                    "ngopt/controller_version_timesteps": self.num_timesteps,
                },
                self.round_index,
            )
        self.rollout_index += 1
        if (
            self.incumbent_rescore_interval > 0
            and self.rollout_index % self.incumbent_rescore_interval == 0
        ):
            adapter = get_ngopt_parameter_choices(self.search_config)[0]
            incumbent = decode_ngopt_parameter_values(
                self.optimizer.provide_recommendation().value, adapter
            )
            compiled = compile_candidates(
                adapter,
                self.search_config,
                [incumbent],
                [self.seed + self.num_timesteps],
            )[0]
            if compiled.valid:
                for environment_index in range(self.model.n_envs):
                    self.training_env.env_method(
                        "set_learner_team", compiled.packed_team, indices=environment_index
                    )
                self.incumbent_return = float(
                    np.mean(
                        evaluate_candidate_population(
                            self.model,
                            self.training_env,
                            self.evaluation_episodes,
                            seed=self.seed + self.rollout_index * self.evaluation_episodes,
                        )
                    )
                )
                if self.wandb_logger is not None:
                    self.wandb_logger.log_training(
                        self.label,
                        self.rollout_index,
                        self.num_timesteps,
                        {
                            "ngopt/incumbent_rescore_return": self.incumbent_return,
                            "ngopt/controller_version_timesteps": self.num_timesteps,
                        },
                        self.round_index,
                    )
        if self.num_timesteps >= self.total_timesteps:
            return
        self.candidates, self.parameters, teams = sample_team_population(
            self.optimizer,
            self.search_config,
            self.model.n_envs,
            self.seed + self.rollout_index * self.model.n_envs,
        )
        for environment_index, team in enumerate(teams):
            self.training_env.env_method("set_learner_team", team, indices=environment_index)
        self.model._last_obs = self.training_env.reset()
        self.model._last_episode_starts = np.ones(self.model.n_envs, dtype=bool)


def add_compute_arguments(parser: argparse.ArgumentParser) -> None:
    """Add shared PPO, vectorization, NGOpt, and accelerator options."""
    parser.add_argument(
        "--device",
        default="cpu",
        help="PyTorch device, for example cpu, cuda, cuda:6, mps, or auto.",
    )
    parser.add_argument("--num-envs", type=int, default=1)
    parser.add_argument("--vec-env", choices=("dummy", "subproc", "sharded"), default="dummy")
    parser.add_argument("--envs-per-worker", type=int, default=32)
    parser.add_argument("--reset-retries", type=int, default=3)
    parser.add_argument("--challenge-timeout", type=float, default=60.0)
    parser.add_argument(
        "--step-timeout",
        type=float,
        default=60.0,
        help="Restart a stalled sharded worker after this many seconds.",
    )
    parser.add_argument(
        "--step-timeout-reward",
        type=float,
        default=0.0,
        help="Reward returned for transitions recovered after a worker timeout.",
    )
    parser.add_argument(
        "--evaluation-timeout",
        type=float,
        default=200.0,
        help="Terminate an evaluation episode after this many seconds.",
    )
    parser.add_argument(
        "--evaluation-timeout-return",
        type=float,
        default=-10.0,
        help="Return assigned to a timed-out evaluation episode.",
    )
    parser.add_argument("--rollout-steps", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--profile-gpu", action="store_true")
    parser.add_argument(
        "--progress",
        action="store_true",
        help="Display tqdm training progress with recent and average environment FPS.",
    )
    parser.add_argument("--gpu-log-interval", type=int, default=1_000)
    parser.add_argument("--telemetry-log-interval", type=int, default=1_000)
    parser.add_argument("--showdown-host", default=SHOWDOWN_HOST)
    parser.add_argument("--showdown-base-port", type=int, default=SHOWDOWN_PORT)
    parser.add_argument("--showdown-shards", type=int, default=1)
    parser.add_argument("--random-team-pool-size", type=int, default=0)
    parser.add_argument("--learner-team-pool-size", type=int, default=0)
    parser.add_argument("--wandb-project")
    parser.add_argument("--wandb-entity")
    parser.add_argument("--wandb-group")
    parser.add_argument("--wandb-name")
    parser.add_argument("--wandb-mode", choices=("online", "offline", "disabled"), default="online")


def validate_compute_arguments(args: argparse.Namespace) -> None:
    """Reject invalid parallelism and PPO buffer configurations early."""
    defaults = {"envs_per_worker": 32, "telemetry_log_interval": 1_000, "showdown_shards": 1}
    for name in (
        "num_envs",
        "envs_per_worker",
        "reset_retries",
        "challenge_timeout",
        "step_timeout",
        "evaluation_timeout",
        "rollout_steps",
        "batch_size",
        "gpu_log_interval",
        "telemetry_log_interval",
        "showdown_shards",
        "ppo_timesteps",
        "evaluation_episodes",
        "optimizer_evaluation_episodes",
    ):
        if getattr(args, name, defaults.get(name, 1)) < 1:
            raise ValueError(f"--{name.replace('_', '-')} must be at least 1")
    rollout_size = args.num_envs * args.rollout_steps
    if rollout_size < 2 or args.batch_size < 2:
        raise ValueError("PPO requires a rollout buffer and --batch-size of at least 2")
    if args.batch_size > rollout_size:
        raise ValueError(
            f"--batch-size ({args.batch_size}) cannot exceed the rollout buffer size ({args.num_envs} environments x {args.rollout_steps} steps)"
        )
    if rollout_size % args.batch_size:
        raise ValueError(
            f"--batch-size ({args.batch_size}) must divide the rollout buffer size ({rollout_size})"
        )
    if getattr(args, "ppo_timesteps", rollout_size) < rollout_size:
        raise ValueError(
            f"--ppo-timesteps must be at least one complete rollout ({rollout_size} = num_envs x rollout_steps)"
        )
    for name in ("random_team_pool_size", "learner_team_pool_size"):
        if getattr(args, name, 0) < 0:
            raise ValueError(f"--{name.replace('_', '-')} cannot be negative")
    if getattr(args, "evaluation_interval_rollouts", 0) < 0:
        raise ValueError("--evaluation-interval-rollouts cannot be negative")
    if getattr(args, "incumbent_rescore_interval", 0) < 0:
        raise ValueError("--incumbent-rescore-interval cannot be negative")
    ShowdownClusterConfig(
        getattr(args, "showdown_host", SHOWDOWN_HOST),
        getattr(args, "showdown_base_port", SHOWDOWN_PORT),
        getattr(args, "showdown_shards", 1),
    )


def rollout_population_evaluations(total_timesteps: int, num_envs: int, rollout_steps: int) -> int:
    """Return the exact number of teams sampled across all PPO rollouts."""
    rollout_size = num_envs * rollout_steps
    return num_envs * ((total_timesteps + rollout_size - 1) // rollout_size)


def validate_device(device_name: str) -> None:
    """Fail before environment startup when an explicit accelerator is unavailable."""
    if device_name == "auto":
        return
    device = torch.device(device_name)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                f"CUDA device {device_name!r} was requested but this PyTorch build cannot use CUDA"
            )
        index = torch.cuda.current_device() if device.index is None else device.index
        if index >= torch.cuda.device_count():
            raise ValueError(
                f"CUDA device {device_name!r} does not exist; {torch.cuda.device_count()} CUDA devices are available"
            )
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested but is unavailable")


def make_team(
    search_config, parameter_values: dict, seed: int, validate_teams: bool = True
) -> "RetryingTeamDesigner":
    """Build one independently seeded copy of an optimized team."""
    team = RetryingTeamDesigner(search_config, validate_teams=validate_teams)
    team.set_optimizable_parameter_values(parameter_values)
    team.rng = np.random.default_rng(seed)
    return team


def make_vector_env(
    env_factories,
    vec_env: str,
    envs_per_worker: int = 32,
    reset_retries: int = 3,
    step_timeout: float = 60.0,
    step_timeout_reward: float = 0.0,
):
    """Create a monitored in-process or process-parallel SB3 vector environment."""
    if vec_env == "sharded":
        return VecMonitor(
            ShardedVecEnv(
                env_factories,
                envs_per_worker=envs_per_worker,
                reset_retries=reset_retries,
                start_method="spawn",
                step_timeout=step_timeout,
                step_timeout_reward=step_timeout_reward,
            )
        )
    vector_class = DummyVecEnv if vec_env == "dummy" else SubprocVecEnv
    kwargs = {} if vec_env == "dummy" else {"start_method": "spawn"}
    return VecMonitor(vector_class(env_factories, **kwargs))


def showdown_cluster(args: argparse.Namespace) -> ShowdownClusterConfig:
    return ShowdownClusterConfig(args.showdown_host, args.showdown_base_port, args.showdown_shards)


def server_configuration_for(args: argparse.Namespace, environment_index: int):
    return showdown_cluster(args).endpoint_for(environment_index).server_configuration


def make_training_callback(
    args: argparse.Namespace,
    wandb_logger=None,
    label: str | None = None,
    evaluation: int | None = None,
    round_index: int | None = None,
    team_optimizer_callback=None,
    periodic_evaluation_callback=None,
    initial_reset_started_at: float | None = None,
):
    """Create accelerator profiling and W&B callbacks for one PPO learner."""
    callbacks = []
    if initial_reset_started_at is not None:
        callbacks.append(InitialResetCallback(args, wandb_logger, initial_reset_started_at))
    if wandb_logger is not None and wandb_logger.run is not None:
        callbacks.append(WandbRLCallback(wandb_logger, label, evaluation, round_index))
    if args.profile_gpu:
        callbacks.append(
            GPUStatsCallback(args.gpu_log_interval, wandb_logger, label, evaluation, round_index)
        )
    if args.progress:
        callbacks.append(TrainingProgressCallback(args.ppo_timesteps, label or "training", 0))
    if team_optimizer_callback is not None:
        callbacks.append(team_optimizer_callback)
    if periodic_evaluation_callback is not None:
        callbacks.append(periodic_evaluation_callback)
    if not callbacks:
        return None
    return callbacks[0] if len(callbacks) == 1 else CallbackList(callbacks)


def evaluation_replay_dir(output_dir: Path, label: str) -> Path:
    """Create an isolated directory containing only evaluation replays."""
    replay_dir = output_dir / "replays" / label
    replay_dir.mkdir(parents=True, exist_ok=True)
    return replay_dir


def collect_evaluation_replays(replay_dir: Path, previous_replays: set[Path]) -> tuple[Path, ...]:
    """Return one representative replay for an evaluation checkpoint."""
    unique_replays = {}
    for replay_path in sorted(set(replay_dir.glob("*.html")) - previous_replays):
        battle_name = replay_path.name.split(" - ", 1)[-1]
        unique_replays.setdefault(battle_name, replay_path)
    return tuple(list(unique_replays.values())[:1])


def evaluation_account_configurations(
    opponent_name: str, evaluation: int, parallel: bool = False
) -> tuple[AccountConfiguration, AccountConfiguration]:
    """Give a saved evaluation replay meaningful, collision-safe player names."""
    suffix = f"{evaluation + 1:X}" if parallel else ""
    return AccountConfiguration(f"TrainingAgent{suffix}", None), AccountConfiguration(
        f"{opponent_name}{suffix}", None
    )


class WandbEvaluationLogger:
    """Log evaluation metrics and completed Showdown HTML replays to W&B."""

    def __init__(self, args: argparse.Namespace, config: dict, enabled: bool = True):
        self.wandb = None
        self.run = None
        self.lock = Lock()
        self.started_at = time.monotonic()
        self.completed_evaluations = 0
        self.telemetry_log_interval = args.telemetry_log_interval
        if not enabled or args.wandb_project is None:
            return
        try:
            import wandb
        except ImportError:
            raise ImportError(
                "W&B logging requires the rl dependency: pip install -e '.[rl]'"
            ) from None
        self.wandb = wandb
        self.run = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            group=args.wandb_group,
            name=args.wandb_name,
            mode=args.wandb_mode,
            config=config,
            dir=str(args.output_dir),
        )
        self.run.log(
            {
                "system/status": "initialized",
                "system/wall_time_s": 0.0,
                "system/requested_environments": args.num_envs,
                "system/showdown_shards": args.showdown_shards,
            }
        )

    def log(
        self,
        label: str,
        evaluation: int,
        mean_return: float,
        parameters: dict,
        replay_paths: tuple[Path, ...],
        **metrics,
    ) -> None:
        if self.run is None:
            return
        values = {
            "evaluation/index": evaluation,
            "evaluation/mean_return": mean_return,
            "evaluation/label": label,
            "evaluation/team": parameters,
            **metrics,
        }
        if replay_paths:
            values["evaluation/replay"] = self.wandb.Html(
                replay_paths[0].read_text(encoding="utf-8"), inject=False
            )
        with self.lock:
            self.run.log(values)
            if replay_paths:
                artifact = self.wandb.Artifact(
                    f"{self.run.id}-{label}-replays", type="evaluation-replays"
                )
                for replay_path in replay_paths:
                    artifact.add_file(str(replay_path))
                self.run.log_artifact(artifact)

    def log_training(
        self,
        label: str,
        evaluation: int,
        timesteps: int,
        metrics: dict,
        round_index: int | None = None,
    ) -> None:
        if self.run is None:
            return
        values = {
            "rl/label": label,
            "rl/candidate": evaluation,
            "rl/timesteps": timesteps,
            "rl/wall_time_s": time.monotonic() - self.started_at,
            **metrics,
        }
        if round_index is not None:
            values["rl/round"] = round_index + 1
        with self.lock:
            self.run.log(values)

    def log_system(self, status: str, **metrics) -> None:
        if self.run is None:
            return
        values = {
            "system/status": status,
            "system/wall_time_s": time.monotonic() - self.started_at,
            **{f"system/{name}": value for name, value in metrics.items()},
        }
        with self.lock:
            self.run.log(values)

    def log_optimization(
        self,
        label: str,
        evaluation: int,
        mean_return: float,
        best_return: float,
        parameters: dict,
        recommendation: dict,
        duration: float,
        round_index: int | None = None,
    ) -> None:
        if self.run is None:
            return
        with self.lock:
            self.completed_evaluations += 1
            values = {
                "optimization/label": label,
                "optimization/evaluation": evaluation,
                "optimization/completed": self.completed_evaluations,
                "optimization/mean_return": mean_return,
                "optimization/best_return": best_return,
                "optimization/duration_s": duration,
                "optimization/wall_time_s": time.monotonic() - self.started_at,
                "optimization/team": parameters,
                "optimization/recommendation": recommendation,
            }
            if round_index is not None:
                values["optimization/round"] = round_index + 1
            self.run.log(values)

    def log_checkpoint(self, label: str, paths: tuple[Path, ...]) -> None:
        if self.run is None:
            return
        artifact = self.wandb.Artifact(f"{self.run.id}-{label}-controller", type="model")
        for path in paths:
            if path.exists():
                artifact.add_file(str(path))
        with self.lock:
            self.run.log_artifact(artifact)

    def log_final(self, best_return: float, recommendation: dict) -> None:
        if self.run is None:
            return
        with self.lock:
            self.run.summary["optimization/best_return"] = best_return
            self.run.summary["optimization/recommendation"] = recommendation

    def finish(self) -> None:
        if self.run is not None:
            self.run.finish()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.finish()


def wandb_config(args: argparse.Namespace) -> dict:
    """Return a W&B-safe copy of command-line configuration."""
    return {
        name: str(value) if isinstance(value, Path) else value for name, value in vars(args).items()
    }


def is_server_running(host: str = SHOWDOWN_HOST, port: int = SHOWDOWN_PORT) -> bool:
    """Return whether something is accepting connections at the server address."""
    try:
        with socket.create_connection((host, port), timeout=0.2):
            return True
    except OSError:
        return False


@contextmanager
def managed_showdown_server(enabled: bool, showdown_path: Path, verbose: bool = False):
    """Start and stop a local Showdown server when requested and necessary."""
    if not enabled:
        yield
        return

    if is_server_running():
        if verbose:
            print(f"showdown_server=status_existing port={SHOWDOWN_PORT}")
        yield
        return

    showdown_path = showdown_path.resolve()
    executable = showdown_path / "pokemon-showdown"
    if not executable.is_file():
        raise FileNotFoundError(f"Pokemon Showdown launcher not found: {executable}")

    output = None if verbose else subprocess.DEVNULL
    process = subprocess.Popen(
        ["node", str(executable), "start", "--no-security"],
        cwd=showdown_path,
        stdout=output,
        stderr=output,
        start_new_session=True,
    )
    deadline = time.monotonic() + SERVER_STARTUP_TIMEOUT

    try:
        while not is_server_running():
            return_code = process.poll()
            if return_code is not None:
                raise RuntimeError(
                    f"Pokemon Showdown exited during startup with code {return_code}."
                )
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    f"Pokemon Showdown did not open port {SHOWDOWN_PORT} within {SERVER_STARTUP_TIMEOUT:g} seconds."
                )
            time.sleep(0.1)

        if verbose:
            print(f"showdown_server=status_started pid={process.pid} port={SHOWDOWN_PORT}")
        yield
    finally:
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
        if verbose:
            print("showdown_server=status_stopped")


class RetryingTeamDesigner(TeamDesigner):
    """Retry random team details that Showdown rejects."""

    def __init__(
        self, search_config: TeamSearchConfig, max_attempts: int = 10, validate_teams: bool = True
    ):
        super().__init__(search_config, validate_teams=validate_teams)
        self.max_attempts = max_attempts

    def yield_team(self) -> str:
        last_error = None
        for _ in range(self.max_attempts):
            try:
                return super().yield_team()
            except ValueError as error:
                last_error = error

        raise ValueError(f"Could not generate a valid team: {last_error}") from last_error


class ObservationFeaturesExtractor(BaseFeaturesExtractor):
    """Concatenate raw state values while keeping the action mask separate."""

    def __init__(self, observation_space):
        feature_keys = tuple(key for key in observation_space.spaces if key != "action_mask")
        features_dim = sum(
            max(1, int(np.prod(observation_space[key].shape))) for key in feature_keys
        )
        super().__init__(observation_space, features_dim=features_dim)
        self.feature_keys = feature_keys

    def forward(self, observations):
        return torch.cat(
            [
                observations[key].float().reshape(observations[key].shape[0], -1)
                for key in self.feature_keys
            ],
            dim=1,
        )


class MaskedActorCriticPolicy(ActorCriticPolicy):
    """Apply poke-env's action mask to PPO's categorical distribution."""

    def __init__(self, *args, **kwargs):
        super().__init__(
            *args,
            **kwargs,
            net_arch=[64, 64],
            features_extractor_class=ObservationFeaturesExtractor,
        )
        self._action_mask = None

    def forward(self, observations, deterministic=False):
        self._action_mask = observations["action_mask"]
        return super().forward(observations, deterministic)

    def evaluate_actions(self, observations, actions):
        self._action_mask = observations["action_mask"]
        return super().evaluate_actions(observations, actions)

    def _predict(self, observations, deterministic=False):
        self._action_mask = observations["action_mask"]
        return super()._predict(observations, deterministic)

    def extract_features(self, observations, features_extractor=None):
        """Bypass SB3's automatic MultiDiscrete one-hot conversion."""
        if self.share_features_extractor:
            extractor = features_extractor or self.features_extractor
            return extractor(observations)
        return (self.pi_features_extractor(observations), self.vf_features_extractor(observations))

    def get_distribution(self, observations):
        self._action_mask = observations["action_mask"]
        features = self.pi_features_extractor(observations)
        latent_policy = self.mlp_extractor.forward_actor(features)
        return self._get_action_dist_from_latent(latent_policy)

    def predict_values(self, observations):
        features = self.vf_features_extractor(observations)
        latent_value = self.mlp_extractor.forward_critic(features)
        return self.value_net(latent_value)

    def _get_action_dist_from_latent(self, latent_policy):
        action_logits = self.action_net(latent_policy)
        valid_actions = self._action_mask == 1
        no_legal_actions = ~valid_actions.any(dim=-1)
        if no_legal_actions.any():
            valid_actions = valid_actions.clone()
            valid_actions[no_legal_actions, 0] = True
        return self.action_dist.proba_distribution(
            action_logits=action_logits.masked_fill(~valid_actions, float("-inf"))
        )


def make_search_config(format_config: FormatConfig) -> TeamSearchConfig:
    """Create the optimized learner configuration for the selected format."""
    return make_all_attribute_search_config(
        format_config,
        team_size=TEAM_SIZE,
        types_allowed=("water",) if "monotype" in format_config.format_id else None,
    )


@lru_cache(maxsize=None)
def get_ngopt_parameter_choices(
    search_config: TeamSearchConfig,
) -> tuple[TeamDesignVectorAdapter]:
    """Return the adapter used by the bounded NGOpt representation."""
    return (TeamDesignVectorAdapter(search_config),)


def make_ngopt_parameterization(search_config: TeamSearchConfig):
    """Expose all optimized team attributes through one bounded vector."""
    (team_adapter,) = get_ngopt_parameter_choices(search_config)
    return ng.p.Array(shape=(team_adapter.dimension,), lower=0.0, upper=1.0)


def decode_ngopt_parameter_values(
    parameter_values: np.ndarray, team_adapter: TeamDesignVectorAdapter
) -> dict:
    """Decode the bounded NGOpt vector through the canonical team schema."""
    return team_adapter.decode(parameter_values)


def team_species_are_trackable(parameter_values: dict, generation: int) -> bool:
    """Return whether poke-env's fuzzy identifier matching can distinguish every Pokemon."""
    pokedex = GenData.from_gen(generation).pokedex
    species = [pokemon["species"] for pokemon in parameter_values.values()]
    base_species = [
        to_id_str(pokedex.get(to_id_str(name), {}).get("baseSpecies", name)) for name in species
    ]
    for index, base_name in enumerate(base_species):
        for other_index, other_name in enumerate(species):
            if index != other_index and (
                base_name == to_id_str(other_name)
                or base_name in {to_id_str(part) for part in other_name.split("-")}
            ):
                return False
    return True


def make_control_env(
    environment_class: type[BaselineOpponentEnv],
    learner_team: TeamDesigner,
    observation_encoder: HumanVisibleBattleEncoder,
    save_replays: bool | str = False,
    account_configuration1: AccountConfiguration | None = None,
    account_configuration2: AccountConfiguration | None = None,
    **env_kwargs,
):
    """Create the selected baseline environment for one learner team."""
    return Monitor(
        environment_class(
            team=learner_team,
            observation_encoder=observation_encoder,
            account_configuration1=account_configuration1,
            account_configuration2=account_configuration2,
            save_replays=save_replays,
            **env_kwargs,
        )
    )


def make_control_training_env(
    environment_class: type[BaselineOpponentEnv],
    learner_config: TeamSearchConfig,
    parameter_values: dict,
    observation_encoder: HumanVisibleBattleEncoder,
    args: argparse.Namespace,
    seed: int,
    learner_team_pool: RandomTeamPool | None = None,
    opponent_team_pool: RandomTeamPool | None = None,
    num_envs: int | None = None,
    vec_env: str | None = None,
    save_replays: bool | str = False,
    account_configuration1: AccountConfiguration | None = None,
    account_configuration2: AccountConfiguration | None = None,
    learner_teams: list[str] | None = None,
):
    """Create independently seeded vector workers for PPO data collection."""
    num_envs = num_envs or args.num_envs
    vec_env = vec_env or args.vec_env

    def make_env(rank: int):
        def initialize():
            learner_team = (
                learner_teams[rank]
                if learner_teams is not None
                else make_team(learner_config, parameter_values, seed + rank)
                if learner_team_pool is None
                else learner_team_pool.builder(seed + rank)
            )
            return environment_class(
                team=learner_team,
                observation_encoder=observation_encoder,
                opponent_team=None
                if opponent_team_pool is None
                else opponent_team_pool.builder(seed + rank),
                server_configuration=server_configuration_for(args, rank),
                challenge_timeout=args.challenge_timeout,
                save_replays=save_replays,
                account_configuration1=account_configuration1,
                account_configuration2=account_configuration2,
            )

        return initialize

    return make_vector_env(
        [make_env(rank) for rank in range(num_envs)],
        vec_env,
        args.envs_per_worker,
        args.reset_retries,
        args.step_timeout,
        args.step_timeout_reward,
    )


def evaluate_controller_result(
    model: PPO, env_factory, episodes: int, timeout: float, timeout_return: float = -10.0
):
    """Return detailed deterministic evaluation results."""
    return evaluate_isolated(
        env_factory,
        lambda observation: model.predict(observation, deterministic=True)[0],
        episodes,
        timeout,
        timeout_return,
    )


def evaluate_controller(
    model: PPO, env_factory, episodes: int, timeout: float, timeout_return: float = -10.0
) -> tuple[float, int]:
    """Compatibility wrapper returning mean return and timeout count."""
    result = evaluate_controller_result(model, env_factory, episodes, timeout, timeout_return)
    return result.mean_return, result.timed_out_episodes


def evaluate_vector_controller(model: PPO, env: VecEnv, episodes: int) -> float:
    """Return mean deterministic return from a one-worker vector environment."""
    episode_returns = []
    for _ in range(episodes):
        observation = env.reset()
        done = False
        episode_return = 0.0
        while not done:
            action, _ = model.predict(observation, deterministic=True)
            observation, reward, dones, _ = env.step(action)
            episode_return += float(reward[0])
            done = bool(dones[0])
        episode_returns.append(episode_return)
    return float(np.mean(episode_returns))


def watch_saved_controller(
    output_dir: Path,
    games: int,
    turn_delay: float,
    start_delay: float,
    environment_class: type[BaselineOpponentEnv],
    learner_config: TeamSearchConfig,
    observation_encoder: HumanVisibleBattleEncoder,
    device: str,
) -> None:
    """Play slowed exhibition games with the saved best team and controller."""
    controller_path = output_dir / "best_controller"
    team_path = output_dir / "best_team.json"
    schema_path = output_dir / "observation_schema.json"
    if (
        not controller_path.with_suffix(".zip").exists()
        or not team_path.exists()
        or not schema_path.exists()
    ):
        raise FileNotFoundError(
            f"No compatible saved controller, team, and observation schema found in {output_dir}. Run training first."
        )
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    if schema != observation_encoder.get_schema():
        raise ValueError("The saved controller uses a different observation schema. Retrain it.")

    learner_team = RetryingTeamDesigner(learner_config)
    learner_team.set_optimizable_parameter_values(json.loads(team_path.read_text(encoding="utf-8")))
    replay_dir = output_dir / "replays"
    env = make_control_env(
        environment_class,
        learner_team,
        observation_encoder,
        save_replays=str(replay_dir),
        account_configuration1=AccountConfiguration("CodesignRL", None),
        account_configuration2=AccountConfiguration("BaselineOpponent", None),
    )
    model = PPO.load(controller_path, env=env, device=device)

    try:
        for game in range(1, games + 1):
            observation, _ = env.reset()
            battle_env = env.unwrapped.env
            battle_tag = battle_env.battle1.battle_tag
            print(f"watch_game={game}/{games} url=http://localhost:8000/{battle_tag}")
            time.sleep(start_delay)

            terminated = False
            truncated = False
            episode_return = 0.0
            while not terminated and not truncated:
                action, _ = model.predict(observation, deterministic=True)
                observation, reward, terminated, truncated, _ = env.step(action)
                episode_return += float(reward)
                time.sleep(turn_delay)

            print(f"watch_game={game}/{games} return={episode_return:.3f}")
    finally:
        env.close()

    print(f"replays={replay_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ppo-timesteps", type=int, default=2048)
    parser.add_argument("--evaluation-episodes", type=int, default=8)
    parser.add_argument(
        "--optimizer-evaluation-episodes",
        type=int,
        default=2,
        help="Complete deterministic battles used to score each NGOpt candidate.",
    )
    parser.add_argument(
        "--incumbent-rescore-interval",
        type=int,
        default=5,
        help="Rescore NGOpt's incumbent every N PPO rollouts; zero disables it.",
    )
    parser.add_argument(
        "--evaluation-interval-rollouts",
        type=int,
        default=0,
        help="Evaluate after every N completed PPO rollouts; zero disables periodic evaluation.",
    )
    parser.add_argument("--seed", type=int, default=0)
    add_compute_arguments(parser)
    parser.add_argument("--output-dir", type=Path, default=Path("runs/minimal_codesign"))
    parser.add_argument(
        "--environment",
        choices=ENVIRONMENTS,
        default="random",
        help="Baseline environment and opponent policy.",
    )
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument(
        "--launch-server",
        action="store_true",
        help="Start and stop the bundled local Pokemon Showdown server.",
    )
    parser.add_argument("--watch", action="store_true")
    parser.add_argument("--watch-games", type=int, default=3)
    parser.add_argument("--turn-delay", type=float, default=1.0)
    parser.add_argument("--watch-start-delay", type=float, default=5.0)
    return parser.parse_args()


def run_experiment(
    args: argparse.Namespace,
    environment_class: type[BaselineOpponentEnv],
    format_config: FormatConfig,
    wandb_logger: WandbEvaluationLogger,
) -> None:
    """Run training or watch mode against an available Showdown server."""
    started = time.monotonic()
    with startup_stage(args, wandb_logger, "observation_setup"):
        learner_config = make_search_config(format_config)
        observation_encoder = HumanVisibleBattleEncoder.from_showdown(
            format_config, observation_config=OBSERVATION_CONFIG
        )
    if args.watch:
        watch_saved_controller(
            args.output_dir,
            args.watch_games,
            args.turn_delay,
            args.watch_start_delay,
            environment_class,
            learner_config,
            observation_encoder,
            args.device,
        )
        return

    (args.output_dir / "observation_schema.json").write_text(
        json.dumps(observation_encoder.get_schema(), indent=2) + "\n", encoding="utf-8"
    )
    team_adapter = TeamDesignVectorAdapter(learner_config)
    (args.output_dir / "team_design_schema.json").write_text(
        json.dumps(team_adapter.schema(), indent=2) + "\n", encoding="utf-8"
    )

    pool_size = args.random_team_pool_size or (
        max(args.num_envs, 1_024) if args.vec_env == "sharded" else 0
    )
    opponent_team_pool = None
    if pool_size:
        with startup_stage(args, wandb_logger, "opponent_team_pool", teams=pool_size):
            opponent_team_pool = RandomTeamPool.from_showdown(
                format_config.showdown_path, environment_class.OPPONENT_TEAM_FORMAT, pool_size
            )

    rollout_size = args.num_envs * args.rollout_steps
    rollout_count = (args.ppo_timesteps + rollout_size - 1) // rollout_size
    population_evaluations = rollout_population_evaluations(
        args.ppo_timesteps, args.num_envs, args.rollout_steps
    )
    with startup_stage(args, wandb_logger, "learner_population", teams=args.num_envs):
        parametrization = make_ngopt_parameterization(learner_config)
        parametrization.random_state.seed(args.seed)
        optimizer = ng.optimizers.NGOpt(
            parametrization=parametrization,
            budget=population_evaluations,
            num_workers=args.num_envs,
        )
        candidates, parameters, teams = sample_team_population(
            optimizer, learner_config, args.num_envs, args.seed
        )
    with startup_stage(args, wandb_logger, "environment_workers", environments=args.num_envs):
        training_env = make_control_training_env(
            environment_class,
            learner_config,
            parameters[0],
            observation_encoder,
            args,
            args.seed,
            opponent_team_pool=opponent_team_pool,
            learner_teams=teams,
        )
    team_optimizer_callback = RolloutTeamOptimizerCallback(
        optimizer,
        learner_config,
        candidates,
        parameters,
        args.seed,
        args.ppo_timesteps,
        evaluation_episodes=args.optimizer_evaluation_episodes,
        incumbent_rescore_interval=args.incumbent_rescore_interval,
        wandb_logger=wandb_logger,
        label="codesign",
    )
    model = PPO(
        MaskedActorCriticPolicy,
        training_env,
        n_steps=args.rollout_steps,
        batch_size=args.batch_size,
        learning_rate=3e-4,
        gamma=0.99,
        ent_coef=0.01,
        seed=args.seed,
        device=args.device,
        verbose=int(args.verbose),
    )
    if args.verbose:
        print(
            f"environment={args.environment} format={format_config.format_id} team_size={TEAM_SIZE} ppo_timesteps={args.ppo_timesteps} "
            f"rollouts={rollout_count} teams_per_rollout={args.num_envs} ng_evaluations={population_evaluations} device={args.device} "
            f"num_envs={args.num_envs} vec_env={args.vec_env} rollout_steps={args.rollout_steps} batch_size={args.batch_size}"
        )

    def evaluate_periodically(current_model: PPO, completed_rollouts: int, timesteps: int) -> None:
        recommendation = decode_ngopt_parameter_values(
            optimizer.provide_recommendation().value, *get_ngopt_parameter_choices(learner_config)
        )
        label = f"rollout-{completed_rollouts}"
        replay_dir = evaluation_replay_dir(args.output_dir, label)
        previous_replays = set(replay_dir.glob("*.html"))
        evaluation_accounts = evaluation_account_configurations(
            EVALUATION_OPPONENT_NAMES[environment_class], completed_rollouts
        )
        evaluation_seed = args.seed + population_evaluations + completed_rollouts

        def evaluation_env_factory(episode: int):
            seed = evaluation_seed + episode
            return make_control_env(
                environment_class,
                make_team(learner_config, recommendation, seed),
                observation_encoder,
                opponent_team=None
                if opponent_team_pool is None
                else opponent_team_pool.builder(seed),
                server_configuration=server_configuration_for(args, args.num_envs),
                challenge_timeout=args.challenge_timeout,
                save_replays=str(replay_dir),
                account_configuration1=evaluation_accounts[0],
                account_configuration2=evaluation_accounts[1],
            )

        if args.verbose:
            print(
                f"status=evaluating rollout={completed_rollouts} timesteps={timesteps} episodes={args.evaluation_episodes}",
                flush=True,
            )
        mean_return, timed_out_episodes = evaluate_controller(
            current_model,
            evaluation_env_factory,
            args.evaluation_episodes,
            args.evaluation_timeout,
            args.evaluation_timeout_return,
        )
        replay_paths = collect_evaluation_replays(replay_dir, previous_replays)
        evaluation_metrics = {
            "evaluation/timesteps": timesteps,
            "evaluation/rollout": completed_rollouts,
            "evaluation/timed_out_episodes": timed_out_episodes,
        }
        wandb_logger.log(
            label,
            completed_rollouts,
            mean_return,
            recommendation,
            replay_paths,
            **evaluation_metrics,
        )
        print(
            f"evaluation_rollout={completed_rollouts} timesteps={timesteps} mean_return={mean_return:.3f}",
            flush=True,
        )

    periodic_evaluation_callback = None
    if args.evaluation_interval_rollouts:
        periodic_evaluation_callback = PeriodicEvaluationCallback(
            args.evaluation_interval_rollouts, rollout_count, evaluate_periodically
        )
    try:
        initial_reset_started_at = begin_startup_stage(
            args, wandb_logger, "initial_reset", environments=args.num_envs
        )
        callback = make_training_callback(
            args,
            wandb_logger,
            "codesign",
            0,
            team_optimizer_callback=team_optimizer_callback,
            periodic_evaluation_callback=periodic_evaluation_callback,
            initial_reset_started_at=initial_reset_started_at,
        )
        model.learn(total_timesteps=args.ppo_timesteps, callback=callback)
    finally:
        training_env.close()

    recommendation = decode_ngopt_parameter_values(
        optimizer.provide_recommendation().value, *get_ngopt_parameter_choices(learner_config)
    )
    replay_dir = evaluation_replay_dir(args.output_dir, "final")
    previous_replays = set(replay_dir.glob("*.html"))
    evaluation_accounts = evaluation_account_configurations(
        EVALUATION_OPPONENT_NAMES[environment_class], 0
    )

    def evaluation_env_factory(episode: int):
        seed = args.seed + population_evaluations + episode
        return make_control_env(
            environment_class,
            make_team(learner_config, recommendation, seed),
            observation_encoder,
            opponent_team=None if opponent_team_pool is None else opponent_team_pool.builder(seed),
            server_configuration=server_configuration_for(args, args.num_envs),
            challenge_timeout=args.challenge_timeout,
            save_replays=str(replay_dir),
            account_configuration1=evaluation_accounts[0],
            account_configuration2=evaluation_accounts[1],
        )

    evaluation_result = evaluate_controller_result(
        model,
        evaluation_env_factory,
        args.evaluation_episodes,
        args.evaluation_timeout,
        args.evaluation_timeout_return,
    )
    mean_return = evaluation_result.mean_return
    timed_out_episodes = evaluation_result.timed_out_episodes
    replay_paths = collect_evaluation_replays(replay_dir, previous_replays)
    model.save(args.output_dir / "best_controller")
    write_run_artifacts(
        args.output_dir,
        team_parameters=recommendation,
        team_design_schema=team_adapter.schema(),
        observation_schema=observation_encoder.get_schema(),
        metrics=RunMetrics(
            mean_return=mean_return,
            evaluation_battles=args.evaluation_episodes,
            training_battles=team_optimizer_callback.training_battles,
            environment_transitions=args.ppo_timesteps,
            win_rate=evaluation_result.win_rate,
            mean_episode_length=evaluation_result.mean_episode_length,
            mean_turn_count=evaluation_result.mean_turn_count,
            wall_time_seconds=time.monotonic() - started,
        ),
        metadata={
            "method": "ppo_ngopt",
            "timed_out_episodes": timed_out_episodes,
            "optimizer_evaluations": population_evaluations,
            "incumbent_rescore_return": team_optimizer_callback.incumbent_return,
        },
    )
    wandb_logger.log(
        "final",
        rollout_count - 1,
        mean_return,
        recommendation,
        replay_paths,
        **{"evaluation/timed_out_episodes": timed_out_episodes},
    )
    wandb_logger.log_final(mean_return, recommendation)
    wandb_logger.log_checkpoint(
        "final",
        (
            args.output_dir / "best_controller.zip",
            args.output_dir / "best_team.json",
            args.output_dir / "observation_schema.json",
            args.output_dir / "team_design_schema.json",
        ),
    )
    print(f"evaluated_return={mean_return:.3f} best_team={recommendation}")


def main() -> None:
    args = parse_args()
    validate_compute_arguments(args)
    validate_device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    environment_class = ENVIRONMENTS[args.environment]
    format_config = FormatConfig.from_showdown(environment_class.BATTLE_FORMAT)
    wandb_logger = WandbEvaluationLogger(args, wandb_config(args), enabled=not args.watch)

    try:
        with managed_showdown_cluster(
            args.launch_server, format_config.showdown_path, showdown_cluster(args), args.verbose
        ):
            run_experiment(args, environment_class, format_config, wandb_logger)
    finally:
        wandb_logger.finish()
