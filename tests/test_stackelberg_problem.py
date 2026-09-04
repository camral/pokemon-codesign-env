import numpy as np
from gymnasium import spaces

from pokemon_codesign_env.stackelberg_problem import PokemonStackelbergProblem


class _Adapter:
    dimension = 12
    search_config = type("Config", (), {"team_size": 6})()

    def decode(self, design):
        return {"design": design.tolist()}


class _Battle:
    battle1 = type("BattleState", (), {"won": True, "turn": 9})()

    def reset(self, seed=None):
        return {"state": [0.0], "action_mask": [0, 1, 1]}, {}

    def step(self, action):
        return {"state": [1.0], "action_mask": [1, 0, 1]}, 1, True, False, {}

    def close(self):
        pass


def test_follower_node_scores_map_to_one_masked_global_action(monkeypatch):
    monkeypatch.setattr(
        "pokemon_codesign_env.stackelberg_problem.TeamDesigner",
        lambda *_args, **_kwargs: type(
            "Designer",
            (),
            {
                "rng": None,
                "set_optimizable_parameter_values": lambda self, values: None,
                "yield_team": lambda self: "packed",
            },
        )(),
    )
    battle_space = spaces.Dict(
        {"state": spaces.Box(-1, 1, (1,)), "action_mask": spaces.MultiBinary(3)}
    )
    problem = PokemonStackelbergProblem(_Adapter(), battle_space, 3, lambda team: _Battle())
    observation = problem.reset(np.full((6, 2), 0.5), seed=2)
    assert observation.shape == (6, 4)

    control = np.zeros((6, 3))
    control[:, 0] = 100
    control[:, 2] = 1
    _, _, terminated, _, info = problem.step(control)

    assert terminated
    assert info["selected_battle_action"] == 2
    assert info["won"] is True
    assert info["turn"] == 9
    assert problem.edges.shape == (2, 30)
