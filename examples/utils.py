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
from gymnasium.wrappers.vector import DictInfoToList
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecEnv, VecMonitor

from pokemon_codesign_env.configs import (
    AttributeMode,
    FormatConfig,
    ObservationConfig,
    SelectionMode,
    TeamSearchConfig,
)
from pokemon_codesign_env.envs import MaxBasePowerPlayerEnv, RandomPlayerEnv, SimpleHeuristicsPlayerEnv
import pokemon_codesign_env as pce
from pokemon_codesign_env.envs.base import BaselineOpponentEnv
from pokemon_codesign_env.gym_environment_factory import OPPONENTS
from pokemon_codesign_env.observations import HumanVisibleBattleEncoder
from pokemon_codesign_env.sharded_vec_env import ShardedVecEnv
from pokemon_codesign_env.team_pool import RandomTeamPool, validate_team_batch
from pokemon_codesign_env.spaces import TeamDesignSpace
from pokemon_codesign_env.team_designer import BatchedTeamDesigner, TeamDesigner

TEAM_SIZE = 6
OBSERVATION_CONFIG = ObservationConfig(team_size=TEAM_SIZE)
INVALID_CANDIDATE_LOSS = 1_000_000.0
SHOWDOWN_HOST = "127.0.0.1"
SHOWDOWN_PORT = 8000
SERVER_STARTUP_TIMEOUT = 120.0
ENVIRONMENTS = {"random": RandomPlayerEnv, "max-base-power": MaxBasePowerPlayerEnv, "simple-heuristics": SimpleHeuristicsPlayerEnv}
EVALUATION_OPPONENT_NAMES = {
    RandomPlayerEnv: "RandomPlayer",
    MaxBasePowerPlayerEnv: "MaxPowerPlayer",
    SimpleHeuristicsPlayerEnv: "SimpleHeuristic",
}


def begin_startup_stage(args: argparse.Namespace, wandb_logger, stage: str, **metrics) -> float:
    """Report the beginning of a potentially slow startup stage."""
    if args.verbose:
        details = " ".join(f"{name}={value}" for name, value in metrics.items())
        print(f"startup_stage={stage} status=starting{f' {details}' if details else ''}", flush=True)
    wandb_logger.log_system(f"{stage}_starting", **metrics)
    return time.monotonic()


def finish_startup_stage(args: argparse.Namespace, wandb_logger, stage: str, started_at: float, **metrics) -> None:
    """Report completion and elapsed time for one startup stage."""
    duration = time.monotonic() - started_at
    if args.verbose:
        details = " ".join(f"{name}={value}" for name, value in metrics.items())
        print(f"startup_stage={stage} status=ready duration_s={duration:.3f}{f' {details}' if details else ''}", flush=True)
    wandb_logger.log_system(f"{stage}_ready", startup_stage_s=duration, **metrics)


@contextmanager
def startup_stage(args: argparse.Namespace, wandb_logger, stage: str, **metrics):
    """Time and report a synchronous startup stage."""
    started_at = begin_startup_stage(args, wandb_logger, stage, **metrics)
    try:
        yield
    except Exception:
        if args.verbose:
            print(f"startup_stage={stage} status=failed duration_s={time.monotonic() - started_at:.3f}", flush=True)
        wandb_logger.log_system(f"{stage}_failed", startup_stage_s=time.monotonic() - started_at, **metrics)
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
        finish_startup_stage(self.args, self.wandb_logger, "initial_reset", self.started_at, environments=self.model.n_envs)

    def _on_step(self) -> bool:
        return True


class GPUStatsCallback(BaseCallback):
    """Periodically report CUDA utilization and allocator memory."""

    def __init__(self, log_interval: int, wandb_logger=None, label: str | None = None, evaluation: int | None = None, round_index: int | None = None):
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
        print(f"training_device={device} gpu_name={properties.name} gpu_memory_mib={properties.total_memory / 2**20:.0f}")

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
            metrics = {"gpu/memory_mib": allocated, "gpu/reserved_mib": reserved, "gpu/peak_mib": peak}
            if utilization is not None:
                metrics["gpu/utilization_percent"] = utilization
            self.wandb_logger.log_training(self.label, self.evaluation, self.num_timesteps, metrics, self.round_index)
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
            "rollout_started", label=self.label, candidate=self.evaluation, ready_environments=self.model.n_envs,
            rollout_size=self.model.n_envs * self.model.n_steps, round_index=self.round_index,
        )

    def _log(self) -> None:
        metrics = {}
        for name, value in self.model.logger.name_to_value.items():
            if isinstance(value, (int, float, np.integer, np.floating)):
                metrics[f"rl/{name}"] = float(value)
        self.wandb_logger.log_training(self.label, self.evaluation, self.num_timesteps, metrics, self.round_index)

    def _on_step(self) -> bool:
        dones = self.locals.get("dones")
        if dones is not None:
            self.completed_episodes += int(np.sum(dones))
        infos = self.locals.get("infos") or ()
        recovered_infos = [info for info in infos if info.get("sharded_worker_restarted")]
        if recovered_infos:
            self.recovered_environments += len(recovered_infos)
            self.restarted_workers += len({info["sharded_worker_index"] for info in recovered_infos})
        interval = self.wandb_logger.telemetry_log_interval
        if self.num_timesteps >= self._next_telemetry:
            now = time.monotonic()
            elapsed = max(now - self.started_at, 1e-9)
            interval_elapsed = max(now - self._last_telemetry_at, 1e-9)
            fps = (self.num_timesteps - self._last_telemetry_timesteps) / interval_elapsed
            rollout_size = self.model.n_envs * self.model.n_steps
            metrics = {
                "rl/rollout/vector_steps": self.n_calls, "rl/rollout/transitions_per_second": self.num_timesteps / elapsed,
                "rl/rollout/fps": fps, "rl/rollout/average_fps": self.num_timesteps / elapsed,
                "rl/rollout/completed_episodes": self.completed_episodes,
                "rl/rollout/buffer_fill": ((self.num_timesteps - 1) % rollout_size + 1) / rollout_size,
                "rl/recovery/environments": self.recovered_environments, "rl/recovery/worker_restarts": self.restarted_workers,
            }
            self.wandb_logger.log_training(self.label, self.evaluation, self.num_timesteps, metrics, self.round_index)
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
        total_timesteps = ((self.requested_timesteps + rollout_size - 1) // rollout_size) * rollout_size
        self.started_at = time.monotonic()
        self._last_measurement_at = self.started_at
        self.progress = tqdm(total=total_timesteps, desc=self.label, unit="step", position=self.position, dynamic_ncols=True)

    def _on_step(self) -> bool:
        increment = self.num_timesteps - self.progress.n
        if increment > 0:
            self.progress.update(increment)
        now = time.monotonic()
        measurement_elapsed = now - self._last_measurement_at
        if measurement_elapsed >= 1.0 or self.num_timesteps >= self.progress.total:
            fps = (self.num_timesteps - self._last_measurement_timesteps) / max(measurement_elapsed, 1e-9)
            average_fps = self.num_timesteps / max(now - self.started_at, 1e-9)
            self.progress.set_postfix(fps=f"{fps:,.0f}", average_fps=f"{average_fps:,.0f}", refresh=False)
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
            completed_rollouts == 0 or completed_rollouts >= self.total_rollouts or completed_rollouts % self.interval_rollouts
            or completed_rollouts == self.last_evaluation_rollout
        )
        if should_skip:
            return
        self.evaluator(self.model, completed_rollouts, self.num_timesteps)
        self.last_evaluation_rollout = completed_rollouts


def sample_team_population(optimizer, embodiment_space, population_size: int):
    """Ask NGOpt for one design per environment, telling it about the ones it cannot build.

    The design space decides what a design is: `from_index` decodes a point of `flat_space` or
    raises, and `validate_embodiments` asks Showdown whether the result can actually be built.
    A candidate that fails either is scored as invalid so the optimizer learns to avoid it.
    """
    candidates, embodiments = [], []
    while len(embodiments) < population_size:
        proposed = [optimizer.ask() for _ in range(population_size - len(embodiments))]
        decoded, kept = [], []
        for candidate in proposed:
            try:
                decoded.append(embodiment_space.from_index(candidate.value))
                kept.append(candidate)
            except ValueError:
                optimizer.tell(candidate, INVALID_CANDIDATE_LOSS)
        if not kept:
            continue
        for candidate, embodiment, reason in zip(kept, decoded, _screen(embodiment_space, decoded), strict=True):
            if reason is None:
                candidates.append(candidate)
                embodiments.append(embodiment)
            else:
                optimizer.tell(candidate, INVALID_CANDIDATE_LOSS)
    return candidates, embodiments


def _screen(embodiment_space, embodiments) -> list:
    """Ask Showdown which of these designs it can build, in one batch."""
    format_config = embodiment_space.search_config.format
    teams = BatchedTeamDesigner.serialize_parameter_values(
        embodiment_space.search_config, embodiments, list(range(len(embodiments)))
    )
    return [None if problems is None else "; ".join(problems)
            for problems in validate_team_batch(format_config.showdown_path, format_config.format_id, teams)]


class RolloutTeamOptimizerCallback(BaseCallback):
    """Score one NGOpt embodiment per environment and replace it every rollout."""

    def __init__(
        self, optimizer, embodiment_space, candidates: list, parameters: list[dict], seed: int, total_timesteps: int,
        wandb_logger=None, label: str = "codesign", round_index: int | None = None,
    ):
        super().__init__()
        self.optimizer = optimizer
        self.embodiment_space = embodiment_space
        self.candidates = candidates
        self.parameters = parameters
        self.seed = seed
        self.total_timesteps = total_timesteps
        self.wandb_logger = wandb_logger
        self.label = label
        self.round_index = round_index
        self.rollout_index = 0
        self.rollout_returns = None
        self.best_return = float("-inf")
        self.best_parameters = None

    def _on_rollout_start(self) -> None:
        self.rollout_returns = np.zeros(self.model.n_envs, dtype=np.float64)

    def _on_step(self) -> bool:
        self.rollout_returns += np.asarray(self.locals["rewards"], dtype=np.float64)
        return True

    def _on_rollout_end(self) -> None:
        for candidate, parameters, rollout_return in zip(self.candidates, self.parameters, self.rollout_returns, strict=True):
            self.optimizer.tell(candidate, -float(rollout_return))
            if rollout_return > self.best_return:
                self.best_return = float(rollout_return)
                self.best_parameters = parameters
        if self.wandb_logger is not None:
            self.wandb_logger.log_training(self.label, self.rollout_index, self.num_timesteps, {
                "ngopt/rollout": self.rollout_index + 1, "ngopt/population_size": self.model.n_envs,
                "ngopt/mean_return": float(np.mean(self.rollout_returns)), "ngopt/best_return": self.best_return,
            }, self.round_index)
        self.rollout_index += 1
        if self.num_timesteps >= self.total_timesteps:
            return
        self.candidates, self.parameters = sample_team_population(
            self.optimizer, self.embodiment_space, self.model.n_envs
        )
        # The designs reach the environments through reset, which is where a codesign environment
        # takes one, rather than by installing compiled teams behind its back.
        self.training_env.set_embodiment(self.parameters)
        self.model._last_obs = self.training_env.reset()
        self.model._last_episode_starts = np.ones(self.model.n_envs, dtype=bool)


def add_compute_arguments(parser: argparse.ArgumentParser) -> None:
    """Add shared PPO, vectorization, NGOpt, and accelerator options."""
    parser.add_argument("--device", default="cpu", help="PyTorch device, for example cpu, cuda, cuda:6, mps, or auto.")
    parser.add_argument("--num-envs", type=int, default=1)
    parser.add_argument("--vec-env", choices=("dummy", "subproc", "sharded"), default="dummy")
    parser.add_argument("--envs-per-worker", type=int, default=32)
    parser.add_argument("--reset-retries", type=int, default=3)
    parser.add_argument("--challenge-timeout", type=float, default=60.0)
    parser.add_argument("--step-timeout", type=float, default=60.0, help="Restart a stalled sharded worker after this many seconds.")
    parser.add_argument("--step-timeout-reward", type=float, default=0.0, help="Reward returned for transitions recovered after a worker timeout.")
    parser.add_argument("--evaluation-timeout", type=float, default=200.0, help="Terminate an evaluation episode after this many seconds.")
    parser.add_argument("--evaluation-timeout-return", type=float, default=-10.0, help="Return assigned to a timed-out evaluation episode.")
    parser.add_argument("--rollout-steps", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--profile-gpu", action="store_true")
    parser.add_argument("--progress", action="store_true", help="Display tqdm training progress with recent and average environment FPS.")
    parser.add_argument("--gpu-log-interval", type=int, default=1_000)
    parser.add_argument("--telemetry-log-interval", type=int, default=1_000)
    parser.add_argument(
        "--showdown-shards", type=int, default=0,
        help="Showdown servers to launch, or 0 to use one per 32 environments.",
    )
    parser.add_argument("--wandb-project")
    parser.add_argument("--wandb-entity")
    parser.add_argument("--wandb-group")
    parser.add_argument("--wandb-name")
    parser.add_argument("--wandb-mode", choices=("online", "offline", "disabled"), default="online")


def validate_compute_arguments(args: argparse.Namespace) -> None:
    """Reject invalid parallelism and PPO buffer configurations early."""
    defaults = {"envs_per_worker": 32, "telemetry_log_interval": 1_000}
    for name in (
        "num_envs", "envs_per_worker", "reset_retries", "challenge_timeout", "step_timeout", "evaluation_timeout", "rollout_steps", "batch_size",
        "gpu_log_interval",
        "telemetry_log_interval",
        "ppo_timesteps", "evaluation_episodes",
    ):
        if getattr(args, name, defaults.get(name, 1)) < 1:
            raise ValueError(f"--{name.replace('_', '-')} must be at least 1")
    rollout_size = args.num_envs * args.rollout_steps
    if rollout_size < 2 or args.batch_size < 2:
        raise ValueError("PPO requires a rollout buffer and --batch-size of at least 2")
    if args.batch_size > rollout_size:
        raise ValueError(f"--batch-size ({args.batch_size}) cannot exceed the rollout buffer size ({args.num_envs} environments x {args.rollout_steps} steps)")
    if rollout_size % args.batch_size:
        raise ValueError(f"--batch-size ({args.batch_size}) must divide the rollout buffer size ({rollout_size})")
    if getattr(args, "ppo_timesteps", rollout_size) < rollout_size:
        raise ValueError(f"--ppo-timesteps must be at least one complete rollout ({rollout_size} = num_envs x rollout_steps)")
    if getattr(args, "evaluation_interval_rollouts", 0) < 0:
        raise ValueError("--evaluation-interval-rollouts cannot be negative")
    if getattr(args, "showdown_shards", 0) < 0:
        raise ValueError("--showdown-shards cannot be negative")


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
            raise RuntimeError(f"CUDA device {device_name!r} was requested but this PyTorch build cannot use CUDA")
        index = torch.cuda.current_device() if device.index is None else device.index
        if index >= torch.cuda.device_count():
            raise ValueError(f"CUDA device {device_name!r} does not exist; {torch.cuda.device_count()} CUDA devices are available")
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested but is unavailable")


class SB3VecEnvAdapter(VecEnv):
    """Present a codesign vector environment through the interface Stable-Baselines3 expects.

    The benchmark speaks the codesign API: reset takes an embodiment and returns observations,
    infos and one rejection flag per environment, and step returns terminations and truncations
    separately with infos as one dictionary of arrays. Stable-Baselines3 wants observations alone,
    a single `dones`, and a list of dictionaries. The translation lives here so the benchmark does
    not depend on any one training framework.
    """

    def __init__(self, env):
        self.env = env
        self._actions = None
        self._pending_embodiment = None
        super().__init__(env.num_envs, env.single_observation_space, env.single_action_space)

    def set_embodiment(self, embodiment) -> None:
        """Battle these designs from the next reset onward, one per environment."""
        self._pending_embodiment = embodiment

    @property
    def embodiment_rejected(self):
        return self.env.embodiment_rejected

    def _as_list(self, vector_infos: dict) -> list[dict]:
        """Split Gymnasium's vectorized infos into one dictionary per environment.

        Each key holds one value per environment beside a `_key` mask saying which environments
        actually reported it. A nested dictionary, such as the info of a finished episode, is
        split the same way.
        """
        infos = [{} for _ in range(self.num_envs)]
        self._scatter(vector_infos, infos)
        return infos

    def _scatter(self, vector_infos: dict, infos: list[dict]) -> None:
        for key, values in vector_infos.items():
            if key.startswith("_"):
                continue
            present = vector_infos.get(f"_{key}")
            if isinstance(values, dict):
                nested = [{} for _ in range(self.num_envs)]
                self._scatter(values, nested)
                values = nested
            for index in range(self.num_envs):
                if present is None or present[index]:
                    infos[index][key] = values[index]

    def reset(self):
        seeds = [seed for seed in self._seeds if seed is not None]
        observations, vector_infos, _rejected = self.env.reset(
            seed=seeds[0] if seeds else None, embodiment=self._pending_embodiment
        )
        self._pending_embodiment = None
        self.reset_infos = self._as_list(vector_infos)
        self._reset_seeds()
        self._reset_options()
        return observations

    def step_async(self, actions) -> None:
        self._actions = actions

    def step_wait(self):
        observations, rewards, terminations, truncations, vector_infos = self.env.step(self._actions)
        dones = np.logical_or(terminations, truncations)
        infos = self._as_list(vector_infos)
        for index, info in enumerate(infos):
            info["TimeLimit.truncated"] = bool(truncations[index] and not terminations[index])
            # Gymnasium reports the observation that ended an episode as final_obs under same-step
            # autoreset; Stable-Baselines3 looks for terminal_observation.
            if dones[index] and info.get("final_obs") is not None:
                info["terminal_observation"] = info["final_obs"]
        return observations, rewards, dones, infos

    def close(self) -> None:
        self.env.close()

    # Gymnasium's VectorWrapper does not forward attribute access, so these reach the environments
    # through the vector environment underneath the wrappers.
    def get_attr(self, attr_name: str, indices=None) -> list:
        return self.env.unwrapped.get_attr(attr_name, indices=indices)

    def set_attr(self, attr_name: str, value, indices=None) -> None:
        self.env.unwrapped.set_attr(attr_name, value, indices=indices)

    def env_method(self, method_name: str, *method_args, indices=None, **method_kwargs) -> list:
        return self.env.unwrapped.env_method(method_name, *method_args, indices=indices, **method_kwargs)

    def env_is_wrapped(self, wrapper_class, indices=None) -> list[bool]:
        return self.env.unwrapped.env_is_wrapped(wrapper_class, indices=indices)

    def get_images(self):
        return self.env.unwrapped.get_images()


def make_vector_env(
    env_factories, vec_env: str, envs_per_worker: int = 32, reset_retries: int = 3, step_timeout: float = 60.0,
    step_timeout_reward: float = 0.0,
):
    """Create a monitored in-process or process-parallel SB3 vector environment."""
    if vec_env == "sharded":
        return VecMonitor(SB3VecEnvAdapter(
            ShardedVecEnv(
                env_factories, envs_per_worker=envs_per_worker, reset_retries=reset_retries, start_method="spawn",
                step_timeout=step_timeout, step_timeout_reward=step_timeout_reward,
            )
        ))
    vector_class = DummyVecEnv if vec_env == "dummy" else SubprocVecEnv
    kwargs = {} if vec_env == "dummy" else {"start_method": "spawn"}
    return VecMonitor(vector_class(env_factories, **kwargs))


def make_training_callback(
    args: argparse.Namespace, wandb_logger=None, label: str | None = None,
    evaluation: int | None = None, round_index: int | None = None, team_optimizer_callback=None, periodic_evaluation_callback=None,
    initial_reset_started_at: float | None = None,
):
    """Create accelerator profiling and W&B callbacks for one PPO learner."""
    callbacks = []
    if initial_reset_started_at is not None:
        callbacks.append(InitialResetCallback(args, wandb_logger, initial_reset_started_at))
    if wandb_logger is not None and wandb_logger.run is not None:
        callbacks.append(WandbRLCallback(wandb_logger, label, evaluation, round_index))
    if args.profile_gpu:
        callbacks.append(GPUStatsCallback(args.gpu_log_interval, wandb_logger, label, evaluation, round_index))
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
    replay_dir = output_dir / "evaluation_replays" / label
    replay_dir.mkdir(parents=True, exist_ok=True)
    return replay_dir


def collect_evaluation_replays(replay_dir: Path, previous_replays: set[Path]) -> tuple[Path, ...]:
    """Return one representative replay for an evaluation checkpoint."""
    unique_replays = {}
    for replay_path in sorted(set(replay_dir.glob("*.html")) - previous_replays):
        battle_name = replay_path.name.split(" - ", 1)[-1]
        unique_replays.setdefault(battle_name, replay_path)
    return tuple(list(unique_replays.values())[:1])


def evaluation_account_configurations(opponent_name: str, evaluation: int, parallel: bool = False) -> tuple[AccountConfiguration, AccountConfiguration]:
    """Give a saved evaluation replay meaningful, collision-safe player names."""
    suffix = f"{evaluation + 1:X}" if parallel else ""
    return AccountConfiguration(f"TrainingAgent{suffix}", None), AccountConfiguration(f"{opponent_name}{suffix}", None)


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
            raise ImportError("W&B logging requires the rl dependency: pip install -e '.[rl]'") from None
        self.wandb = wandb
        self.run = wandb.init(
            project=args.wandb_project, entity=args.wandb_entity, group=args.wandb_group,
            name=args.wandb_name, mode=args.wandb_mode, config=config, dir=str(args.output_dir),
        )
        self.run.log({
            "system/status": "initialized", "system/wall_time_s": 0.0,
            "system/requested_environments": args.num_envs, "system/showdown_shards": args.showdown_shards,
        })

    def log(self, label: str, evaluation: int, mean_return: float, parameters: dict, replay_paths: tuple[Path, ...], **metrics) -> None:
        if self.run is None:
            return
        values = {"evaluation/index": evaluation, "evaluation/mean_return": mean_return, "evaluation/label": label, "evaluation/team": parameters, **metrics}
        if replay_paths:
            values["evaluation/replay"] = self.wandb.Html(replay_paths[0].read_text(encoding="utf-8"), inject=False)
        with self.lock:
            self.run.log(values)
            if replay_paths:
                artifact = self.wandb.Artifact(f"{self.run.id}-{label}-replays", type="evaluation-replays")
                for replay_path in replay_paths:
                    artifact.add_file(str(replay_path))
                self.run.log_artifact(artifact)

    def log_training(self, label: str, evaluation: int, timesteps: int, metrics: dict, round_index: int | None = None) -> None:
        if self.run is None:
            return
        values = {"rl/label": label, "rl/candidate": evaluation, "rl/timesteps": timesteps, "rl/wall_time_s": time.monotonic() - self.started_at, **metrics}
        if round_index is not None:
            values["rl/round"] = round_index + 1
        with self.lock:
            self.run.log(values)

    def log_system(self, status: str, **metrics) -> None:
        if self.run is None:
            return
        values = {
            "system/status": status, "system/wall_time_s": time.monotonic() - self.started_at,
            **{f"system/{name}": value for name, value in metrics.items()},
        }
        with self.lock:
            self.run.log(values)

    def log_optimization(
        self, label: str, evaluation: int, mean_return: float, best_return: float, parameters: dict,
        recommendation: dict, duration: float, round_index: int | None = None,
    ) -> None:
        if self.run is None:
            return
        with self.lock:
            self.completed_evaluations += 1
            values = {
                "optimization/label": label, "optimization/evaluation": evaluation, "optimization/completed": self.completed_evaluations,
                "optimization/mean_return": mean_return, "optimization/best_return": best_return, "optimization/duration_s": duration,
                "optimization/wall_time_s": time.monotonic() - self.started_at, "optimization/team": parameters,
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
    return {name: str(value) if isinstance(value, Path) else value for name, value in vars(args).items()}


class RetryingTeamDesigner(TeamDesigner):
    """Retry random team details that Showdown rejects."""

    def __init__(self, search_config: TeamSearchConfig, max_attempts: int = 10, validate_teams: bool = True):
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
        features_dim = sum(max(1, int(np.prod(observation_space[key].shape))) for key in feature_keys)
        super().__init__(observation_space, features_dim=features_dim)
        self.feature_keys = feature_keys

    def forward(self, observations):
        return torch.cat([observations[key].float().reshape(observations[key].shape[0], -1) for key in self.feature_keys], dim=1)


class MaskedActorCriticPolicy(ActorCriticPolicy):
    """Apply poke-env's action mask to PPO's categorical distribution."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args,**kwargs,net_arch=[64, 64],features_extractor_class=ObservationFeaturesExtractor,)
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
        return self.action_dist.proba_distribution(action_logits=action_logits.masked_fill(~valid_actions, float("-inf")))


def make_search_config(format_config: FormatConfig) -> TeamSearchConfig:
    """Create the optimized learner configuration for the selected format."""
    return TeamSearchConfig(
        format=format_config,
        team_size=TEAM_SIZE,
        species=SelectionMode.OPTIMIZED,
        natures=AttributeMode.OPTIMIZED,
        generations_allowed=tuple(range(1, format_config.generation + 1)),
        types_allowed=("water",) if "monotype" in format_config.format_id else None,
    )


def make_ngopt_parameterization(embodiment_space):
    """Search the design space in the flat representation it already provides."""
    nvec = embodiment_space.flat_space.nvec
    return ng.p.Array(
        shape=(len(nvec),), lower=np.zeros(len(nvec), dtype=np.int64), upper=np.asarray(nvec, dtype=np.int64) - 1
    ).set_integer_casting()


# pce.make selects the opponent by name; the examples still choose by class.
OPPONENT_NAMES = {environment_class: name for name, environment_class in OPPONENTS.items()}


def make_control_env(
    environment_class: type[BaselineOpponentEnv],
    learner_config: TeamSearchConfig,
    args: argparse.Namespace,
    save_replays: bool | str = False,
    account_configuration1: AccountConfiguration | None = None,
    account_configuration2: AccountConfiguration | None = None,
    **env_kwargs,
):
    """Create one codesign environment, which launches and owns the Showdown server it needs."""
    env, base_embodiment = pce.make(
        opponent=OPPONENT_NAMES[environment_class],
        format_config=learner_config.format,
        embodiment_config=learner_config,
        challenge_timeout=args.challenge_timeout,
        save_replays=save_replays,
        account_configuration1=account_configuration1,
        account_configuration2=account_configuration2,
        **env_kwargs,
    )
    return Monitor(env), base_embodiment


def make_control_training_env(
    environment_class: type[BaselineOpponentEnv],
    learner_config: TeamSearchConfig,
    args: argparse.Namespace,
    num_envs: int | None = None,
    save_replays: bool | str = False,
    account_configuration1: AccountConfiguration | None = None,
    account_configuration2: AccountConfiguration | None = None,
    **env_kwargs,
):
    """Create the vector environment PPO collects from, and one base design per environment.

    The environments launch and own the Showdown servers they battle on, so nothing here has to
    start a server or be told where one is running.
    """
    num_envs = num_envs or args.num_envs
    vector_env, base_embodiments = pce.make_vec(
        opponent=OPPONENT_NAMES[environment_class],
        num_envs=num_envs,
        format_config=learner_config.format,
        embodiment_config=learner_config,
        shards=args.showdown_shards or None,
        envs_per_worker=args.envs_per_worker,
        reset_retries=args.reset_retries,
        step_timeout=args.step_timeout,
        step_timeout_reward=args.step_timeout_reward,
        challenge_timeout=args.challenge_timeout,
        save_replays=save_replays,
        account_configuration1=account_configuration1,
        account_configuration2=account_configuration2,
        **env_kwargs,
    )
    return VecMonitor(SB3VecEnvAdapter(vector_env)), base_embodiments


def evaluate_recommendation(
    model: PPO, environment_class, learner_config: TeamSearchConfig, args: argparse.Namespace, embodiment: dict,
    replay_dir: Path, accounts, episodes: int,
) -> tuple[float, int]:
    """Score one design over several episodes in an environment that is reused between them.

    An environment owns the Showdown server it battles on, so building one per episode would start
    and stop a server per episode. One environment plays them all instead.
    """
    evaluation_env, _base = make_control_training_env(
        environment_class, learner_config, args, num_envs=1, save_replays=str(replay_dir),
        account_configuration1=accounts[0], account_configuration2=accounts[1],
    )
    try:
        evaluation_env.set_embodiment(embodiment)
        return evaluate_vector_controller(model, evaluation_env, episodes), 0
    finally:
        evaluation_env.close()


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
    if (not controller_path.with_suffix(".zip").exists() or not team_path.exists() or not schema_path.exists()):
        raise FileNotFoundError(f"No compatible saved controller, team, and observation schema found in {output_dir}. Run training first.")
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
        "--evaluation-interval-rollouts", type=int, default=0,
        help="Evaluate after every N completed PPO rollouts; zero disables periodic evaluation.",
    )
    parser.add_argument("--seed", type=int, default=0)
    add_compute_arguments(parser)
    parser.add_argument("--output-dir", type=Path, default=Path("runs/minimal_codesign"))
    parser.add_argument("--environment", choices=ENVIRONMENTS, default="random", help="Baseline environment and opponent policy.")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--watch", action="store_true")
    parser.add_argument("--watch-games", type=int, default=3)
    parser.add_argument("--turn-delay", type=float, default=1.0)
    parser.add_argument("--watch-start-delay", type=float, default=5.0)
    return parser.parse_args()


def run_experiment(
    args: argparse.Namespace, environment_class: type[BaselineOpponentEnv], format_config: FormatConfig, wandb_logger: WandbEvaluationLogger
) -> None:
    """Run training or watch mode against an available Showdown server."""
    with startup_stage(args, wandb_logger, "observation_setup"):
        learner_config = make_search_config(format_config)
        observation_encoder = HumanVisibleBattleEncoder.from_showdown(format_config, observation_config=OBSERVATION_CONFIG)
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

    (args.output_dir / "observation_schema.json").write_text(json.dumps(observation_encoder.get_schema(), indent=2) + "\n", encoding="utf-8")

    rollout_size = args.num_envs * args.rollout_steps
    rollout_count = (args.ppo_timesteps + rollout_size - 1) // rollout_size
    population_evaluations = rollout_population_evaluations(args.ppo_timesteps, args.num_envs, args.rollout_steps)
    embodiment_space = TeamDesignSpace(learner_config, seed=args.seed)
    with startup_stage(args, wandb_logger, "learner_population", teams=args.num_envs):
        parametrization = make_ngopt_parameterization(embodiment_space)
        parametrization.random_state.seed(args.seed)
        optimizer = ng.optimizers.NGOpt(parametrization=parametrization, budget=population_evaluations, num_workers=args.num_envs)
        candidates, parameters = sample_team_population(optimizer, embodiment_space, args.num_envs)
    with startup_stage(args, wandb_logger, "environment_workers", environments=args.num_envs):
        training_env, _base_embodiments = make_control_training_env(environment_class, learner_config, args)
        training_env.set_embodiment(parameters)
    team_optimizer_callback = RolloutTeamOptimizerCallback(
        optimizer, embodiment_space, candidates, parameters, args.seed, args.ppo_timesteps, wandb_logger, "codesign"
    )
    model = PPO(
        MaskedActorCriticPolicy, training_env, n_steps=args.rollout_steps, batch_size=args.batch_size, learning_rate=3e-4,
        gamma=0.99, ent_coef=0.01, seed=args.seed, device=args.device, verbose=int(args.verbose),
    )
    if args.verbose:
        print(
            f"environment={args.environment} format={format_config.format_id} team_size={TEAM_SIZE} ppo_timesteps={args.ppo_timesteps} "
            f"rollouts={rollout_count} teams_per_rollout={args.num_envs} ng_evaluations={population_evaluations} device={args.device} "
            f"num_envs={args.num_envs} vec_env={args.vec_env} rollout_steps={args.rollout_steps} batch_size={args.batch_size}"
        )

    def evaluate_periodically(current_model: PPO, completed_rollouts: int, timesteps: int) -> None:
        recommendation = embodiment_space.from_index(optimizer.provide_recommendation().value)
        label = f"rollout-{completed_rollouts}"
        replay_dir = evaluation_replay_dir(args.output_dir, label)
        previous_replays = set(replay_dir.glob("*.html"))
        evaluation_accounts = evaluation_account_configurations(EVALUATION_OPPONENT_NAMES[environment_class], completed_rollouts)
        evaluation_seed = args.seed + population_evaluations + completed_rollouts
        if args.verbose:
            print(f"status=evaluating rollout={completed_rollouts} timesteps={timesteps} episodes={args.evaluation_episodes}", flush=True)
        mean_return, timed_out_episodes = evaluate_recommendation(
            current_model, environment_class, learner_config, args, recommendation, replay_dir, evaluation_accounts,
            args.evaluation_episodes,
        )
        replay_paths = collect_evaluation_replays(replay_dir, previous_replays)
        evaluation_metrics = {
            "evaluation/timesteps": timesteps, "evaluation/rollout": completed_rollouts, "evaluation/timed_out_episodes": timed_out_episodes,
        }
        wandb_logger.log(label, completed_rollouts, mean_return, recommendation, replay_paths, **evaluation_metrics)
        print(f"evaluation_rollout={completed_rollouts} timesteps={timesteps} mean_return={mean_return:.3f}", flush=True)

    periodic_evaluation_callback = None
    if args.evaluation_interval_rollouts:
        periodic_evaluation_callback = PeriodicEvaluationCallback(args.evaluation_interval_rollouts, rollout_count, evaluate_periodically)
    try:
        initial_reset_started_at = begin_startup_stage(args, wandb_logger, "initial_reset", environments=args.num_envs)
        callback = make_training_callback(
            args, wandb_logger, "codesign", 0, team_optimizer_callback=team_optimizer_callback, periodic_evaluation_callback=periodic_evaluation_callback,
            initial_reset_started_at=initial_reset_started_at,
        )
        model.learn(total_timesteps=args.ppo_timesteps, callback=callback)
    finally:
        training_env.close()

    recommendation = embodiment_space.from_index(optimizer.provide_recommendation().value)
    replay_dir = evaluation_replay_dir(args.output_dir, "final")
    previous_replays = set(replay_dir.glob("*.html"))
    evaluation_accounts = evaluation_account_configurations(EVALUATION_OPPONENT_NAMES[environment_class], 0)
    mean_return, timed_out_episodes = evaluate_recommendation(
        model, environment_class, learner_config, args, recommendation, replay_dir, evaluation_accounts, args.evaluation_episodes
    )
    replay_paths = collect_evaluation_replays(replay_dir, previous_replays)
    model.save(args.output_dir / "best_controller")
    (args.output_dir / "best_team.json").write_text(json.dumps(recommendation, indent=2) + "\n")
    wandb_logger.log("final", rollout_count - 1, mean_return, recommendation, replay_paths, **{"evaluation/timed_out_episodes": timed_out_episodes})
    wandb_logger.log_final(mean_return, recommendation)
    wandb_logger.log_checkpoint(
        "final", (args.output_dir / "best_controller.zip", args.output_dir / "best_team.json", args.output_dir / "observation_schema.json")
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
        # The environments launch and own the Showdown servers they battle on, so there is no
        # server to start here and none to point them at.
        run_experiment(args, environment_class, format_config, wandb_logger)
    finally:
        wandb_logger.finish()
