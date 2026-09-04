"""Factory consumed by the upstream FastTD3 Gymnasium adapter."""

from __future__ import annotations

import numpy as np
from gymnasium import spaces
from poke_env.environment import SinglesEnv
from poke_env.ps_client import ServerConfiguration

from pokemon_codesign_env.baseline_adapters import (
    TeamDesignVectorAdapter,
    make_all_attribute_search_config,
)
from pokemon_codesign_env.configs import FormatConfig, ObservationConfig
from pokemon_codesign_env.envs import (
    MaxBasePowerPlayerEnv,
    RandomPlayerEnv,
    SimpleHeuristicsPlayerEnv,
)
from pokemon_codesign_env.envs.fasttd3_codesign import FastTD3PokemonCodesignEnv
from pokemon_codesign_env.observations import HumanVisibleBattleEncoder

ENVIRONMENTS = {
    "random": RandomPlayerEnv,
    "max-base-power": MaxBasePowerPlayerEnv,
    "simple-heuristics": SimpleHeuristicsPlayerEnv,
}


def make_env(
    *,
    environment: str = "random",
    showdown_host: str = "127.0.0.1",
    showdown_port: int = 8000,
    challenge_timeout: float = 60.0,
    save_replays: bool | str = False,
    device: str | None = None,
):
    """Build one lazy design-first Pokemon environment."""

    del device
    environment_class = ENVIRONMENTS[environment]
    format_config = FormatConfig.from_showdown(environment_class.BATTLE_FORMAT)
    search_config = make_all_attribute_search_config(format_config)
    adapter = TeamDesignVectorAdapter(search_config)
    encoder = HumanVisibleBattleEncoder.from_showdown(
        format_config, observation_config=ObservationConfig(team_size=6)
    )
    action_count = SinglesEnv.get_action_space_size(format_config.generation)
    battle_space = spaces.Dict(
        {**encoder.observation_space.spaces, "action_mask": spaces.MultiBinary(action_count)}
    )
    server = ServerConfiguration(
        f"ws://{showdown_host}:{showdown_port}/showdown/websocket",
        "https://play.pokemonshowdown.com/action.php?",
    )

    def make_battle_env(packed_team: str):
        return environment_class(
            team=packed_team,
            observation_encoder=encoder,
            server_configuration=server,
            challenge_timeout=challenge_timeout,
            save_replays=save_replays,
        )

    return FastTD3PokemonCodesignEnv(adapter, battle_space, action_count, make_battle_env)


def action_layout(environment: str = "random") -> dict[str, int]:
    """Return offsets needed by the phase-projecting FastTD3 actor."""

    environment_class = ENVIRONMENTS[environment]
    format_config = FormatConfig.from_showdown(environment_class.BATTLE_FORMAT)
    adapter = TeamDesignVectorAdapter(make_all_attribute_search_config(format_config))
    encoder = HumanVisibleBattleEncoder.from_showdown(
        format_config, observation_config=ObservationConfig(team_size=6)
    )
    action_count = SinglesEnv.get_action_space_size(format_config.generation)
    battle_space = spaces.Dict(
        {**encoder.observation_space.spaces, "action_mask": spaces.MultiBinary(action_count)}
    )
    mask_offset = 0
    for key, space in battle_space.spaces.items():
        if key == "action_mask":
            break
        mask_offset += int(np.prod(space.shape))
    return {
        "embodiment_dimension": adapter.dimension,
        "battle_action_count": action_count,
        "action_mask_offset": mask_offset,
    }
