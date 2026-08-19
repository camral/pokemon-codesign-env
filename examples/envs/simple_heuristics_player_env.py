from poke_env.player import SimpleHeuristicsPlayer

from examples.envs.base import BaselineOpponentEnv
from pokemon_codesign_env.configs import ObservationConfig


class SimpleHeuristicsPlayerEnv(BaselineOpponentEnv):
    """Gen 9 Monotype against the heuristic baseline and a random team."""

    BATTLE_FORMAT = "gen9monotype"
    OPPONENT_TEAM_FORMAT = "gen9monotyperandombattle"
    OPPONENT_CLASS = SimpleHeuristicsPlayer
    OBSERVATION_CONFIG = ObservationConfig()
