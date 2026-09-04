"""Pokemon backend for the repaired upstream House of Dextra lifecycle."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
from functools import partial
from typing import Any

import numpy as np
from gymnasium import spaces
from poke_env.environment import SinglesEnv
from poke_env.ps_client import ServerConfiguration

from pokemon_codesign_env.baseline_adapters import (
    PokemonTeamGraphAdapter,
    TeamDesignVectorAdapter,
    make_all_attribute_search_config,
)
from pokemon_codesign_env.configs import FormatConfig, ObservationConfig
from pokemon_codesign_env.fasttd3_factory import ENVIRONMENTS
from pokemon_codesign_env.observations import HumanVisibleBattleEncoder
from pokemon_codesign_env.sharded_vec_env import ShardedVecEnv
from pokemon_codesign_env.team_designer import TeamDesigner


def _make_battle_environment(config: dict, design: dict, rank: int):
    environment_class = ENVIRONMENTS[config["environment"]]
    format_config = FormatConfig.from_showdown(environment_class.BATTLE_FORMAT)
    adapter = TeamDesignVectorAdapter(make_all_attribute_search_config(format_config))
    designer = TeamDesigner(adapter.search_config, validate_teams=True)
    designer.rng = np.random.default_rng(int(config["seed"]) + rank)
    designer.set_optimizable_parameter_values(design)
    encoder = HumanVisibleBattleEncoder.from_showdown(
        format_config, observation_config=ObservationConfig(team_size=6)
    )
    port = int(config["showdown_port"]) + rank % int(config["showdown_shards"])
    server = ServerConfiguration(
        f"ws://{config['showdown_host']}:{port}/showdown/websocket",
        "https://play.pokemonshowdown.com/action.php?",
    )
    return environment_class(
        team=designer.yield_team(),
        observation_encoder=encoder,
        server_configuration=server,
        challenge_timeout=float(config["challenge_timeout"]),
        save_replays=config.get("save_replays", False),
    )


class PokemonControllerBatch:
    """Sharded, auto-resetting Pokémon battles with continuous action scores."""

    def __init__(
        self,
        problem: PokemonHouseCodesignProblem,
        designs: Sequence[dict],
        *,
        training: bool,
        seed: int,
    ):
        self.problem = problem
        self.designs = list(designs)
        self.action_size = problem.action_count
        self.observation_size = problem.observation_dimension
        runtime = {
            **problem.runtime_config,
            "seed": seed,
            "save_replays": (
                problem.runtime_config.get("save_replays", False) if not training else False
            ),
        }
        factories = [
            partial(_make_battle_environment, runtime, design, rank)
            for rank, design in enumerate(self.designs)
        ]
        self.env = ShardedVecEnv(
            factories,
            envs_per_worker=int(runtime["envs_per_worker"]),
            reset_retries=int(runtime["reset_retries"]),
            step_timeout=float(runtime["step_timeout"]),
            step_timeout_reward=float(runtime["step_timeout_reward"]),
            start_method=runtime["start_method"],
        )
        self.env.seed(seed)
        self._action_mask = np.ones((len(designs), self.action_size), dtype=bool)

    def _observation(self, observation: Mapping[str, Any]) -> np.ndarray:
        self._action_mask = np.asarray(observation["action_mask"], dtype=bool)
        return np.concatenate(
            [
                np.asarray(observation[key], dtype=np.float32).reshape(len(self.designs), -1)
                for key in self.problem.observation_keys
            ],
            axis=1,
        )

    def reset(self) -> np.ndarray:
        return self._observation(self.env.reset())

    def step(self, actions):
        scores = np.asarray(actions, dtype=np.float32)
        expected = (len(self.designs), self.action_size)
        if scores.shape != expected:
            raise ValueError(f"Expected action scores with shape {expected}, got {scores.shape}")
        legal = self._action_mask.copy()
        no_legal = ~legal.any(axis=1)
        legal[no_legal, 0] = True
        selected = np.argmax(np.where(legal, scores, -np.inf), axis=1).astype(np.int64)
        observation, rewards, dones, infos = self.env.step(selected)
        return (
            self._observation(observation),
            rewards.astype(np.float32),
            dones,
            {
                "selected_battle_action": selected,
                "raw_infos": infos,
            },
        )

    def close(self) -> None:
        self.env.close()


class PokemonHouseCodesignProblem:
    """One House controller family for a fixed Pokémon battle-format ABI."""

    def __init__(self, config: Mapping[str, Any]):
        environment = str(config.get("environment", "random"))
        if environment not in ENVIRONMENTS:
            raise ValueError(f"Unknown Pokemon environment {environment!r}")
        environment_class = ENVIRONMENTS[environment]
        self.format_config = FormatConfig.from_showdown(environment_class.BATTLE_FORMAT)
        self.vector_adapter = TeamDesignVectorAdapter(
            make_all_attribute_search_config(self.format_config)
        )
        self.graph_adapter = PokemonTeamGraphAdapter(self.vector_adapter)
        self.family_name = self.format_config.format_id
        self.encoder = HumanVisibleBattleEncoder.from_showdown(
            self.format_config, observation_config=ObservationConfig(team_size=6)
        )
        self.action_count = SinglesEnv.get_action_space_size(self.format_config.generation)
        battle_space = spaces.Dict(
            {
                **self.encoder.observation_space.spaces,
                "action_mask": spaces.MultiBinary(self.action_count),
            }
        )
        self.observation_keys = tuple(key for key in battle_space.spaces if key != "action_mask")
        self.observation_dimension = sum(
            int(np.prod(battle_space[key].shape)) for key in self.observation_keys
        )
        shards = int(config.get("showdown_shards", 1))
        if shards < 1:
            raise ValueError("showdown_shards must be positive")
        self.runtime_config = {
            "environment": environment,
            "showdown_host": str(config.get("showdown_host", "127.0.0.1")),
            "showdown_port": int(config.get("showdown_port", 8000)),
            "showdown_shards": shards,
            "challenge_timeout": float(config.get("challenge_timeout", 60.0)),
            "step_timeout": float(config.get("step_timeout", 120.0)),
            "step_timeout_reward": float(config.get("step_timeout_reward", 0.0)),
            "envs_per_worker": int(config.get("envs_per_worker", 8)),
            "reset_retries": int(config.get("reset_retries", 3)),
            "start_method": str(config.get("start_method", "spawn")),
            "save_replays": config.get("save_replays", False),
        }

    def families(self) -> Sequence[str]:
        return [self.family_name]

    def _check_family(self, family: str) -> None:
        if family != self.family_name:
            raise ValueError(f"Unknown controller family {family!r}")

    def sample_design(self, family: str, rng: np.random.Generator) -> dict:
        self._check_family(family)
        return self.graph_adapter.initial_design(rng)

    def initial_design(self, family: str, rng: np.random.Generator) -> dict:
        return self.sample_design(family, rng)

    def mutate(self, design: dict, family: str, rng: np.random.Generator) -> dict:
        self._check_family(family)
        return self.graph_adapter.mutate(design, rng)

    def family(self, design: dict) -> str:
        self.vector_adapter.designer.set_optimizable_parameter_values(design)
        return self.family_name

    def encode(self, design: dict) -> tuple[np.ndarray, np.ndarray]:
        return self.graph_adapter.encode(design)

    def morphology(self, designs: Sequence[dict]) -> np.ndarray:
        return np.stack([self.graph_adapter.encode(design)[0].reshape(-1) for design in designs])

    def action_mask(self, designs: Sequence[dict]) -> np.ndarray:
        return np.ones((len(designs), self.action_count), dtype=np.float32)

    def make_controller_batch(
        self, designs: Sequence[dict], *, training: bool, seed: int
    ) -> PokemonControllerBatch:
        return PokemonControllerBatch(self, designs, training=training, seed=seed)

    def serialize(self, design: dict) -> dict:
        return self.graph_adapter.serialize(design)

    def deserialize(self, payload: Any) -> dict:
        design = deepcopy(payload)
        self.vector_adapter.designer.set_optimizable_parameter_values(design)
        return design

    def schema(self) -> dict:
        return {
            "family": self.family_name,
            "observation": self.encoder.get_schema(),
            "graph": self.graph_adapter.schema(),
        }


def make_problem(config: Mapping[str, Any]) -> PokemonHouseCodesignProblem:
    return PokemonHouseCodesignProblem(config)
