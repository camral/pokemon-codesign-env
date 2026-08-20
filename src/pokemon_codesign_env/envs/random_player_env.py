import random

from poke_env.battle import Pokemon
from poke_env.player import RandomPlayer

from pokemon_codesign_env.envs.base import BaselineOpponentEnv


class SwitchWeightedRandomPlayer(RandomPlayer):
    """Random player that switches with a fixed probability when moves are available."""

    def __init__(self, switch_probability: float = 0.2, **kwargs):
        if not 0 <= switch_probability <= 1:
            raise ValueError("switch_probability must be between 0 and 1")
        self.switch_probability = switch_probability
        super().__init__(**kwargs)

    def choose_move(self, battle):
        orders = battle.valid_orders
        if not orders:
            return self.choose_default_move()
        switch_orders = [order for order in orders if isinstance(order.order, Pokemon)]
        move_orders = [order for order in orders if not isinstance(order.order, Pokemon)]
        if not move_orders:
            return random.choice(switch_orders)
        if switch_orders and random.random() < self.switch_probability:
            return random.choice(switch_orders)
        return random.choice(move_orders)


class RandomPlayerEnv(BaselineOpponentEnv):
    """Gen 4 Custom Game against switch-weighted random actions and a random team."""

    BATTLE_FORMAT = "gen4customgame"
    OPPONENT_TEAM_FORMAT = "gen4randombattle"
    OPPONENT_CLASS = SwitchWeightedRandomPlayer
