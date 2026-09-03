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
