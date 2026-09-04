from types import SimpleNamespace

import numpy as np

from pokemon_codesign_env.envs.stackelberg_codesign import LazyStackelbergPokemonEnv


class _Problem:
    design = np.full((6, 2), 0.5)
    design_low = np.zeros((6, 2))
    design_high = np.ones((6, 2))
    fixed_features = np.eye(6)
    edges = np.asarray([[0, 1], [1, 0]])
    control_action_dim = 3
    observation_dimension = 1

    def close(self):
        pass

    def reset(self, design, seed):
        self.last_design = design
        return np.zeros((6, 1))

    def step(self, control):
        return np.ones((6, 1)), 4.0, True, False, {"won": True, "turn": 7}


def test_lazy_stackelberg_tracks_selected_team_and_episode(monkeypatch):
    problem = _Problem()
    monkeypatch.setattr(
        "pokemon_codesign_env.envs.stackelberg_codesign._load_factory",
        lambda _path: lambda _config: problem,
    )
    cfg = SimpleNamespace(env_specs={"factory": "fake:factory"}, seed=3)
    env = LazyStackelbergPokemonEnv(cfg)
    action = np.zeros((6, 6))

    _, reward, done, _, info = env.step(action)
    assert reward == 0
    assert not done
    assert info["use_transform_action"]
    np.testing.assert_array_equal(info["selected_design"], 0.5)

    _, reward, done, _, info = env.step(action)
    assert reward == 4
    assert done
    assert info["selected_design"].shape == (6, 2)
    assert env.completed_returns == [4.0]
    assert env.wins == [True]
    assert env.turn_counts == [7]
    np.testing.assert_array_equal(env.best_design, 0.5)
