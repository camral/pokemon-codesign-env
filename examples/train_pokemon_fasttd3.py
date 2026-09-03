"""Train FastTD3 with phase-projected Pokemon embodiment and battle actions."""

from __future__ import annotations

import json
import sys
import types
from dataclasses import dataclass
from pathlib import Path

import torch

from pokemon_codesign_env.baseline_adapters.fasttd3 import project_tensor_actions
from pokemon_codesign_env.fasttd3_factory import action_layout

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


_args: PokemonFastTD3Args | None = None
_layout: dict[str, int] | None = None


def get_args():
    global _args, _layout
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
    return _args


def _make_envs(*_unused, **_unused_kwargs):
    if _args is None:
        raise RuntimeError("Arguments must be parsed before environments are created")
    from codesign.gymnasium_env import GymnasiumVectorEnv

    kwargs = json.dumps(
        {
            "environment": _args.environment,
            "showdown_host": _args.showdown_host,
            "showdown_port": _args.showdown_port,
            "challenge_timeout": _args.challenge_timeout,
        }
    )
    common = dict(
        env_name=_args.env_name,
        seed=_args.seed,
        device=torch.device(_args.device),
        env_factory="pokemon_codesign_env.fasttd3_factory:make_env",
        env_kwargs=kwargs,
        max_episode_steps=_args.max_episode_steps,
        normalize_actions=False,
    )
    training = GymnasiumVectorEnv(num_envs=_args.num_envs, **common)
    evaluation = GymnasiumVectorEnv(
        num_envs=_args.num_eval_envs, **{**common, "seed": _args.seed + 10_000}
    )
    return training, evaluation, evaluation


def main():
    if not SOURCE_ROOT.is_dir():
        raise RuntimeError(f"FastTD3 baseline not found at {SOURCE_ROOT}")
    import fast_td3 as upstream_models
    import train as upstream_train

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

        def explore(self, observations, dones=None, deterministic=False):
            actions = super().explore(observations, dones, deterministic)
            return self._project(observations, actions)

    upstream_models.Actor = PokemonActor
    environment_shim = types.ModuleType("environments.mujoco_playground_env")
    environment_shim.make_env = _make_envs
    sys.modules[environment_shim.__name__] = environment_shim
    upstream_train.get_args = get_args
    upstream_train.main()


if __name__ == "__main__":
    main()
