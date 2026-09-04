"""Train FastTD3 with phase-projected Pokemon embodiment and battle actions."""

from __future__ import annotations

import json
import shutil
import sys
import time
import types
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from pokemon_codesign_env.baseline_adapters import (
    TeamDesignVectorAdapter,
    make_all_attribute_search_config,
)
from pokemon_codesign_env.baseline_adapters.fasttd3 import project_tensor_actions
from pokemon_codesign_env.baseline_artifacts import RunMetrics, write_run_artifacts
from pokemon_codesign_env.configs import FormatConfig, ObservationConfig
from pokemon_codesign_env.fasttd3_factory import ENVIRONMENTS, action_layout
from pokemon_codesign_env.fasttd3_tracking import (
    FastTD3RunTracker,
    executed_action_array,
)
from pokemon_codesign_env.fasttd3_vector_env import FastTD3ShardedVectorEnv
from pokemon_codesign_env.observations import HumanVisibleBattleEncoder

BASELINE_ROOT = Path(__file__).resolve().parents[2] / "codesign-baselines" / "fasttd3_rl"
SOURCE_ROOT = BASELINE_ROOT / "fast_td3"
sys.path.insert(0, str(SOURCE_ROOT))

from hyperparams import BaseArgs  # noqa: E402


@dataclass
class PokemonFastTD3Args(BaseArgs):
    env_name: str = "pokemon-codesign"
    environment: str = "random"
    showdown_host: str = "127.0.0.1"
    showdown_port: int = 8000
    challenge_timeout: float = 60.0
    max_episode_steps: int = 1_000
    device: str = "auto"
    num_envs: int = 8
    num_eval_envs: int = 2
    total_timesteps: int = 100_000
    buffer_size: int = 50_000
    batch_size: int = 1_024
    learning_starts: int = 100
    obs_normalization: bool = False
    compile: bool = False
    amp: bool = False
    render_interval: int = 0
    v_min: float = -100.0
    v_max: float = 100.0
    use_wandb: bool = False
    exp_name: str = "pokemon_fasttd3"
    output_dir: Path = Path("runs/pokemon_fasttd3")
    vec_env: str = "sharded"
    envs_per_worker: int = 8
    showdown_shards: int = 1


_args: PokemonFastTD3Args | None = None
_layout: dict[str, int] | None = None
_training_env = None
_evaluation_env = None
_started = None


def get_args():
    global _args, _layout, _started
    import tyro

    _args = tyro.cli(PokemonFastTD3Args)
    requested = _args.device
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    _args.device = str(torch.device(requested))
    _args.cuda = requested != "cpu"
    _args.device_rank = torch.device(requested).index or 0
    if _args.obs_normalization:
        raise ValueError("Pokemon phase and legality projection requires --no-obs-normalization")
    _layout = action_layout(_args.environment)
    if _args.vec_env not in {"sync", "sharded"}:
        raise ValueError("vec-env must be 'sync' or 'sharded'")
    if _args.showdown_shards < 1:
        raise ValueError("showdown-shards must be positive")
    _started = time.monotonic()
    return _args


def _make_envs(*_unused, **_unused_kwargs):
    global _training_env, _evaluation_env
    if _args is None:
        raise RuntimeError("Arguments must be parsed before environments are created")
    from codesign.gymnasium_env import GymnasiumVectorEnv

    def environment_kwargs(rank, *, evaluation=False):
        return json.dumps(
            {
                "environment": _args.environment,
                "showdown_host": _args.showdown_host,
                "showdown_port": _args.showdown_port + rank % _args.showdown_shards,
                "challenge_timeout": _args.challenge_timeout,
                "save_replays": str(_args.output_dir / "replays") if evaluation else False,
            }
        )

    common = dict(
        env_name=_args.env_name,
        device=torch.device(_args.device),
        env_factory="pokemon_codesign_env.fasttd3_factory:make_env",
        max_episode_steps=_args.max_episode_steps,
    )

    base_vector_env = (
        GymnasiumVectorEnv if _args.vec_env == "sync" else FastTD3ShardedVectorEnv
    )

    class ExecutedActionVectorEnv(base_vector_env):
        """Expose the exact environment-retained actions to replay insertion."""

        last_executed_actions = None

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.tracker = FastTD3RunTracker(self.num_envs)

        def step(self, actions):
            result = super().step(actions)
            raw_infos = result[3]["gymnasium"]
            executed = executed_action_array(raw_infos, self.num_envs)
            self.last_executed_actions = torch.as_tensor(
                executed, device=self.sim_device, dtype=torch.float32
            )
            self.tracker.update(
                raw_infos,
                result[1].detach().cpu().numpy(),
                result[2].detach().cpu().numpy(),
            )
            return result

    def build(num_envs, seed, *, evaluation=False):
        if _args.vec_env == "sync":
            return ExecutedActionVectorEnv(
                num_envs=num_envs,
                seed=seed,
                env_kwargs=environment_kwargs(0, evaluation=evaluation),
                normalize_actions=False,
                **common,
            )
        return ExecutedActionVectorEnv(
            num_envs=num_envs,
            seed=seed,
            env_kwargs_by_rank=[
                environment_kwargs(rank, evaluation=evaluation) for rank in range(num_envs)
            ],
            envs_per_worker=_args.envs_per_worker,
            step_timeout=_args.challenge_timeout,
            **common,
        )

    training = build(_args.num_envs, _args.seed)
    evaluation = build(_args.num_eval_envs, _args.seed + 10_000, evaluation=True)
    _training_env = training
    _evaluation_env = evaluation
    return training, evaluation, evaluation


def _write_artifacts():
    if _args is None or _training_env is None:
        return
    tracker = (
        _evaluation_env.tracker
        if _evaluation_env is not None and _evaluation_env.tracker.best_parameters is not None
        else _training_env.tracker
    )
    environment_class = ENVIRONMENTS[_args.environment]
    format_config = FormatConfig.from_showdown(environment_class.BATTLE_FORMAT)
    adapter = TeamDesignVectorAdapter(make_all_attribute_search_config(format_config))
    encoder = HumanVisibleBattleEncoder.from_showdown(
        format_config, observation_config=ObservationConfig(team_size=6)
    )
    parameters = tracker.best_parameters
    if parameters is None:
        parameters = adapter.decode(np.full(adapter.dimension, 0.5))
    else:
        parameters = adapter.decode(parameters)
    source_checkpoint = Path("models") / (
        f"{_args.env_name}__{_args.exp_name}__{_args.seed}_final.pt"
    )
    _args.output_dir.mkdir(parents=True, exist_ok=True)
    if source_checkpoint.is_file():
        shutil.copy2(source_checkpoint, _args.output_dir / "best_controller.pt")
    write_run_artifacts(
        _args.output_dir,
        team_parameters=parameters,
        team_design_schema=adapter.schema(),
        observation_schema=encoder.get_schema(),
        metrics=RunMetrics(
            mean_return=tracker.mean_return if tracker.mean_return is not None else 0.0,
            evaluation_battles=len(
                _evaluation_env.tracker.completed_returns if _evaluation_env else []
            ),
            training_battles=len(_training_env.tracker.completed_returns),
            environment_transitions=_training_env.tracker.environment_transitions,
            invalid_candidate_rate=_training_env.tracker.invalid_candidate_rate,
            wall_time_seconds=None if _started is None else time.monotonic() - _started,
        ),
        metadata={
            "method": "fasttd3",
            "best_episode_return": (
                tracker.best_return if np.isfinite(tracker.best_return) else None
            ),
        },
    )


def main():
    if not SOURCE_ROOT.is_dir():
        raise RuntimeError(f"FastTD3 baseline not found at {SOURCE_ROOT}")
    import fast_td3 as upstream_models
    import train as upstream_train
    from fast_td3_utils import SimpleReplayBuffer

    upstream_actor = upstream_models.Actor

    class PokemonActor(upstream_actor):
        def _project(self, observations, actions):
            if _layout is None:
                raise RuntimeError("Pokemon action layout has not been initialized")
            phases = observations[:, -1] > 0.5
            projected = project_tensor_actions(actions, phases, _layout["embodiment_dimension"])
            start = _layout["action_mask_offset"]
            stop = start + _layout["battle_action_count"]
            legal = observations[:, start:stop] > 0.5
            battle = projected[:, _layout["embodiment_dimension"] :] * legal
            return torch.cat((projected[:, : _layout["embodiment_dimension"]], battle), dim=-1)

        def forward(self, observations):
            return self._project(observations, super().forward(observations))

        def explore(self, obs, dones=None, deterministic=False):
            actions = super().explore(obs, dones, deterministic)
            return self._project(obs, actions)

    upstream_models.Actor = PokemonActor

    class ExecutedActionReplayBuffer(SimpleReplayBuffer):
        def extend(self, transition):
            if _training_env is None or _training_env.last_executed_actions is None:
                raise RuntimeError("No executed Pokemon actions are available for replay")
            transition["actions"] = _training_env.last_executed_actions
            return super().extend(transition)

    upstream_train.SimpleReplayBuffer = ExecutedActionReplayBuffer
    environment_shim = types.ModuleType("environments.mujoco_playground_env")
    environment_shim.make_env = _make_envs
    sys.modules[environment_shim.__name__] = environment_shim
    upstream_train.get_args = get_args
    upstream_train.main()
    _write_artifacts()


if __name__ == "__main__":
    main()
