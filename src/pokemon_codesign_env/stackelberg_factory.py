"""Factory for the Pokemon Stackelberg PPO problem backend."""

from __future__ import annotations

from gymnasium import spaces
from poke_env.environment import SinglesEnv
from poke_env.ps_client import ServerConfiguration

from pokemon_codesign_env.baseline_adapters import (
    TeamDesignVectorAdapter,
    make_all_attribute_search_config,
)
from pokemon_codesign_env.configs import FormatConfig, ObservationConfig
from pokemon_codesign_env.fasttd3_factory import ENVIRONMENTS
from pokemon_codesign_env.observations import HumanVisibleBattleEncoder
from pokemon_codesign_env.stackelberg_problem import PokemonStackelbergProblem


def make_problem(config):
    environment_class = ENVIRONMENTS[config.get("environment", "random")]
    format_config = FormatConfig.from_showdown(environment_class.BATTLE_FORMAT)
    adapter = TeamDesignVectorAdapter(make_all_attribute_search_config(format_config))
    encoder = HumanVisibleBattleEncoder.from_showdown(
        format_config, observation_config=ObservationConfig(team_size=6)
    )
    action_count = SinglesEnv.get_action_space_size(format_config.generation)
    battle_space = spaces.Dict(
        {**encoder.observation_space.spaces, "action_mask": spaces.MultiBinary(action_count)}
    )
    host = config.get("showdown_host", "127.0.0.1")
    port = int(config.get("showdown_port", 8000))
    server = ServerConfiguration(
        f"ws://{host}:{port}/showdown/websocket",
        "https://play.pokemonshowdown.com/action.php?",
    )
    runtime_options = {"save_replays": config.get("save_replays", False)}

    def make_battle_env(team):
        return environment_class(
            team=team,
            observation_encoder=encoder,
            server_configuration=server,
            challenge_timeout=float(config.get("challenge_timeout", 60)),
            save_replays=runtime_options["save_replays"],
        )

    problem = PokemonStackelbergProblem(adapter, battle_space, action_count, make_battle_env)
    problem.runtime_options = runtime_options
    return problem
