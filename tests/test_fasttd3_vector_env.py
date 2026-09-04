import numpy as np
import torch
from gymnasium import spaces

from pokemon_codesign_env.fasttd3_vector_env import FastTD3ShardedVectorEnv


class _VectorEnv:
    observation_space = spaces.Box(-1, 1, (2,), dtype=np.float32)
    action_space = spaces.Box(-1, 1, (2,), dtype=np.float32)

    def seed(self, seed):
        self.seed_value = seed

    def reset(self):
        return np.asarray([[0.0, 0.0], [0.0, 0.0]], dtype=np.float32)

    def step(self, actions):
        self.actions = actions
        return (
            np.asarray([[9.0, 9.0], [2.0, 2.0]], dtype=np.float32),
            np.asarray([1.0, 2.0]),
            np.asarray([True, False]),
            (
                {
                    "terminal_observation": np.asarray([1.0, 1.0]),
                    "TimeLimit.truncated": True,
                    "executed_actions": actions[0],
                },
                {"executed_actions": actions[1]},
            ),
        )

    def close(self):
        self.closed = True


def _adapter():
    adapter = FastTD3ShardedVectorEnv.__new__(FastTD3ShardedVectorEnv)
    adapter.envs = _VectorEnv()
    adapter.num_envs = 2
    adapter.sim_device = torch.device("cpu")
    adapter._action_low = np.full(2, -1, dtype=np.float32)
    adapter._action_high = np.full(2, 1, dtype=np.float32)
    adapter._initial_seed = 12
    return adapter


def test_sharded_adapter_preserves_terminal_observation_and_infos():
    adapter = _adapter()
    reset = adapter.reset()
    assert adapter.envs.seed_value == 12
    assert reset.shape == (2, 2)

    observations, rewards, dones, infos = adapter.step(torch.tensor([[2.0, 0.5], [-2.0, -0.5]]))

    torch.testing.assert_close(observations, torch.tensor([[9.0, 9.0], [2.0, 2.0]]))
    torch.testing.assert_close(rewards, torch.tensor([1.0, 2.0]))
    torch.testing.assert_close(dones, torch.tensor([True, False]))
    torch.testing.assert_close(infos["observations"]["raw"]["obs"][0], torch.tensor([1.0, 1.0]))
    assert infos["time_outs"].tolist() == [True, False]
    np.testing.assert_array_equal(adapter.envs.actions, [[1.0, 0.5], [-1.0, -0.5]])
