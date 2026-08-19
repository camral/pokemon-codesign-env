from poke_env.player import MaxBasePowerPlayer

from examples.envs.base import BaselineOpponentEnv
from pokemon_codesign_env.configs import ObservationConfig


class MaxBasePowerPlayerEnv(BaselineOpponentEnv):
    """Gen 6 OU against the maximum-base-power baseline and a random team."""

    BATTLE_FORMAT = "gen6ou"
    OPPONENT_TEAM_FORMAT = "gen6battlefactory"
    OPPONENT_CLASS = MaxBasePowerPlayer
    OBSERVATION_CONFIG = ObservationConfig(max_effects=8, max_fields=6, max_side_conditions=10)
