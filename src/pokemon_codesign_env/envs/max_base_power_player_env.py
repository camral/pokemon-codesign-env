from poke_env.player import MaxBasePowerPlayer

from pokemon_codesign_env.envs.base import BaselineOpponentEnv


class MaxBasePowerPlayerEnv(BaselineOpponentEnv):
    """Gen 6 Custom Game against the maximum-base-power baseline and a random team."""

    BATTLE_FORMAT = "gen6customgame"
    OPPONENT_TEAM_FORMAT = "gen6randombattle"
    OPPONENT_CLASS = MaxBasePowerPlayer
