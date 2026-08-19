from poke_env.player import RandomPlayer

from pokemon_codesign_env.envs.base import BaselineOpponentEnv


class RandomPlayerEnv(BaselineOpponentEnv):
    """Gen 4 Custom Game against random actions and a random team."""

    BATTLE_FORMAT = "gen4customgame"
    OPPONENT_TEAM_FORMAT = "gen4randombattle"
    OPPONENT_CLASS = RandomPlayer
