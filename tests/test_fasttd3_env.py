import numpy as np
from gymnasium import spaces

from pokemon_codesign_env.envs.fasttd3_codesign import FastTD3PokemonCodesignEnv


class _Adapter:
    dimension = 2
    search_config = object()

    def decode(self, vector):
        return {"decoded": vector.tolist()}


class _Battle:
    def __init__(self):
        self.closed = False

    def reset(self):
        return {"state": np.array([0.25]), "action_mask": np.array([0, 1, 1])}, {}

    def step(self, action):
        return (
            {"state": np.array([0.5]), "action_mask": np.array([1, 0, 1])},
            2.0,
            True,
            False,
            {"raw": True},
        )

    def close(self):
        self.closed = True


def test_reset_is_lazy_and_actions_are_projected(monkeypatch):
    created = []
    battle_space = spaces.Dict(
        {"state": spaces.Box(-1, 1, (1,)), "action_mask": spaces.MultiBinary(3)}
    )
    monkeypatch.setattr(
        "pokemon_codesign_env.envs.fasttd3_codesign.TeamDesigner",
        lambda *_args, **_kwargs: type(
            "Designer",
            (),
            {
                "set_optimizable_parameter_values": lambda self, values: None,
                "yield_team": lambda self: "packed",
            },
        )(),
    )
    env = FastTD3PokemonCodesignEnv(
        _Adapter(), battle_space, 3, lambda team: created.append(_Battle()) or created[-1]
    )

    observation, info = env.reset()
    assert not created
    assert observation[-1] == 1
    assert info["phase"] == "design"

    _, reward, terminated, _, info = env.step(np.array([-1, 1, 0.8, 0.7, 0.6]))
    assert len(created) == 1
    assert reward == 0
    assert not terminated
    np.testing.assert_array_equal(info["executed_actions"], [-1, 1, 0, 0, 0])

    _, reward, terminated, _, info = env.step(np.array([0.9, 0.8, 1.0, 0.5, -0.5]))
    assert reward == 2
    assert terminated
    assert info["selected_battle_action"] == 1
    np.testing.assert_array_equal(info["executed_actions"], [0, 0, 0, 0.5, -0.5])
