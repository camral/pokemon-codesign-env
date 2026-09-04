"""Pickle-safe isolated Pokemon evaluator for CMA-ES workers."""

from __future__ import annotations

import numpy as np
from gymnasium import spaces
from poke_env.environment import SinglesEnv
from poke_env.ps_client import ServerConfiguration

from pokemon_codesign_env.baseline_adapters import (
    TeamDesignVectorAdapter,
    make_all_attribute_search_config,
)
from pokemon_codesign_env.compact_controller import CompactBattleController
from pokemon_codesign_env.configs import FormatConfig, ObservationConfig
from pokemon_codesign_env.fasttd3_factory import ENVIRONMENTS
from pokemon_codesign_env.observations import HumanVisibleBattleEncoder
from pokemon_codesign_env.team_designer import TeamDesigner


class PokemonCMAEvaluator:
    def __init__(
        self,
        environment: str,
        seed: int,
        evaluation_repeats: int = 1,
        invalid_reward: float = -1_000_000,
        showdown_host: str = "127.0.0.1",
        showdown_port: int = 8000,
        challenge_timeout: float = 60.0,
    ):
        if evaluation_repeats < 1:
            raise ValueError("evaluation_repeats must be positive")
        self.environment = environment
        self.seed = seed
        self.evaluation_repeats = evaluation_repeats
        self.invalid_reward = invalid_reward
        self.showdown_host = showdown_host
        self.showdown_port = showdown_port
        self.challenge_timeout = challenge_timeout
        self.evaluation_index = 0

    @property
    def evaluation_seed(self) -> int:
        """Repeat the identical opponent-seed panel for every candidate."""

        return self.seed + self.evaluation_index % self.evaluation_repeats

    def __call__(self, design, policy, device=None):
        del device
        environment_class = ENVIRONMENTS[self.environment]
        format_config = FormatConfig.from_showdown(environment_class.BATTLE_FORMAT)
        search_config = make_all_attribute_search_config(format_config)
        adapter = TeamDesignVectorAdapter(search_config)
        encoder = HumanVisibleBattleEncoder.from_showdown(
            format_config, observation_config=ObservationConfig(team_size=6)
        )
        action_count = SinglesEnv.get_action_space_size(format_config.generation)
        observation_space = spaces.Dict(
            {**encoder.observation_space.spaces, "action_mask": spaces.MultiBinary(action_count)}
        )
        controller = CompactBattleController(observation_space, action_count)
        try:
            evaluation_seed = self.evaluation_seed
            parameters = adapter.decode(np.clip(design, 0, 1))
            designer = TeamDesigner(search_config, validate_teams=True)
            designer.rng = np.random.default_rng(evaluation_seed)
            designer.set_optimizable_parameter_values(parameters)
            packed_team = designer.yield_team()
            controller.set_parameters(policy)
        except (RuntimeError, TypeError, ValueError):
            return self.invalid_reward

        server = ServerConfiguration(
            f"ws://{self.showdown_host}:{self.showdown_port}/showdown/websocket",
            "https://play.pokemonshowdown.com/action.php?",
        )
        env = environment_class(
            team=packed_team,
            observation_encoder=encoder,
            server_configuration=server,
            challenge_timeout=self.challenge_timeout,
        )
        try:
            observation, _ = env.reset(seed=evaluation_seed)
            terminated = truncated = False
            score = 0.0
            while not terminated and not truncated:
                observation, reward, terminated, truncated, _ = env.step(
                    controller.predict(observation)
                )
                score += float(reward)
        finally:
            env.close()
        self.evaluation_index += 1
        return score
