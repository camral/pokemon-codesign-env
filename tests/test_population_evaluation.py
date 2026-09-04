import numpy as np

from pokemon_codesign_env.baseline_adapters.evaluation import evaluate_candidate_population


class _ConstantModel:
    def predict(self, observations, deterministic):
        assert deterministic
        return np.zeros(len(observations), dtype=np.int64), None


class _AutoResetVectorEnv:
    num_envs = 2

    def __init__(self):
        self.lengths = np.array([2, 3])
        self.rewards = np.array([1.0, 2.0])
        self.steps = np.zeros(2, dtype=np.int64)
        self.seed_calls = []

    def env_method(self, name, value, indices):
        self.seed_calls.append((name, value, indices))

    def reset(self):
        self.steps[:] = 0
        return np.zeros((2, 1))

    def step(self, actions):
        self.steps += 1
        dones = self.steps == self.lengths
        self.steps[dones] = 0
        return np.zeros((2, 1)), self.rewards.copy(), dones, [{}, {}]


def test_each_candidate_gets_the_same_number_of_complete_battles():
    returns = evaluate_candidate_population(_ConstantModel(), _AutoResetVectorEnv(), episodes=2)

    np.testing.assert_array_equal(returns, [2.0, 6.0])


def test_population_evaluation_installs_same_seed_panel():
    env = _AutoResetVectorEnv()

    evaluate_candidate_population(_ConstantModel(), env, episodes=1, seed=40)

    assert env.seed_calls[:2] == [
        ("set_reset_seed_sequence", 40, 0),
        ("set_reset_seed_sequence", 40, 1),
    ]
    assert env.seed_calls[-2:] == [
        ("set_reset_seed_sequence", None, 0),
        ("set_reset_seed_sequence", None, 1),
    ]
