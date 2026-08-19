from poke_env.player import RandomPlayer

from examples.envs.base import BaselineOpponentEnv
from pokemon_codesign_env.configs import ObservationConfig


class RandomPlayerEnv(BaselineOpponentEnv):
    """Gen 4 Custom Game against random actions and a random team."""

    BATTLE_FORMAT = "gen4customgame"
    OPPONENT_TEAM_FORMAT = "gen4randombattle"
    OPPONENT_CLASS = RandomPlayer
    OBSERVATION_CONFIG = ObservationConfig()
