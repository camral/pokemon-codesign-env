from poke_env.player import MaxBasePowerPlayer

from pokemon_codesign_env.envs.base import BaselineOpponentEnv


class MaxBasePowerPlayerEnv(BaselineOpponentEnv):
    """Gen 6 OU against the maximum-base-power baseline and a random team."""

    BATTLE_FORMAT = "gen6ou"
    OPPONENT_TEAM_FORMAT = "gen6battlefactory"
    OPPONENT_CLASS = MaxBasePowerPlayer
