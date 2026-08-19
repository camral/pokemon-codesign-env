from poke_env.player import SimpleHeuristicsPlayer

from pokemon_codesign_env.envs.base import BaselineOpponentEnv


class SimpleHeuristicsPlayerEnv(BaselineOpponentEnv):
    """Gen 9 Monotype against the heuristic baseline and a random team."""

    BATTLE_FORMAT = "gen9monotype"
    OPPONENT_TEAM_FORMAT = "gen9monotyperandombattle"
    OPPONENT_CLASS = SimpleHeuristicsPlayer
