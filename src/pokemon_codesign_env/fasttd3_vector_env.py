"""FastTD3 tensor adapter over the repository's sharded Gymnasium collector."""

from __future__ import annotations

import json
from functools import partial
from importlib import import_module

import gymnasium as gym
import numpy as np
import torch

from pokemon_codesign_env.sharded_vec_env import ShardedVecEnv


def _make_timed_environment(factory_path: str, kwargs_json: str, max_episode_steps: int):
    module_name, attribute = factory_path.split(":", 1)
    factory = getattr(import_module(module_name), attribute)
    env = factory(**json.loads(kwargs_json))
    env = gym.wrappers.TimeLimit(env, max_episode_steps=max_episode_steps)
    if (
        not isinstance(env.observation_space, gym.spaces.Box)
        or len(env.observation_space.shape) != 1
    ):
        env = gym.wrappers.FlattenObservation(env)
    return env


class FastTD3ShardedVectorEnv:
    """Expose `ShardedVecEnv` through the small API expected by FastTD3."""

    asymmetric_obs = False
    num_privileged_obs = 0

    def __init__(
        self,
        *,
        num_envs: int,
        seed: int,
        device: torch.device,
        env_factory: str,
        env_kwargs_by_rank: list[str],
        max_episode_steps: int,
        envs_per_worker: int = 8,
        step_timeout: float = 60.0,
        start_method: str = "spawn",
    ):
        if len(env_kwargs_by_rank) != num_envs:
            raise ValueError("env_kwargs_by_rank must contain one entry per environment")
        factories = [
            partial(_make_timed_environment, env_factory, kwargs, max_episode_steps)
            for kwargs in env_kwargs_by_rank
        ]
        self.envs = ShardedVecEnv(
            factories,
            envs_per_worker=envs_per_worker,
            step_timeout=step_timeout,
            start_method=start_method,
        )
        observation_space = self.envs.observation_space
        action_space = self.envs.action_space
        if not isinstance(observation_space, gym.spaces.Box) or len(observation_space.shape) != 1:
            raise TypeError("FastTD3 requires one-dimensional Box observations")
        if not isinstance(action_space, gym.spaces.Box) or len(action_space.shape) != 1:
            raise TypeError("FastTD3 requires one-dimensional Box actions")
        self.num_envs = num_envs
        self.sim_device = device
        self.num_obs = int(np.prod(observation_space.shape))
        self.num_actions = int(np.prod(action_space.shape))
        self.max_episode_steps = max_episode_steps
        self._action_low = action_space.low.astype(np.float32)
        self._action_high = action_space.high.astype(np.float32)
        self._initial_seed = seed

    def _tensor(self, value, dtype):
        return torch.as_tensor(value, device=self.sim_device, dtype=dtype)

    def reset(self):
        if self._initial_seed is not None:
            self.envs.seed(self._initial_seed)
            self._initial_seed = None
        return self._tensor(self.envs.reset(), torch.float32)

    def step(self, actions):
        numpy_actions = np.clip(actions.detach().cpu().numpy(), self._action_low, self._action_high)
        observations, rewards, dones, raw_infos = self.envs.step(numpy_actions)
        true_observations = np.asarray(observations).copy()
        truncations = np.zeros(self.num_envs, dtype=bool)
        for index, info in enumerate(raw_infos):
            truncations[index] = bool(info.get("TimeLimit.truncated", False))
            if dones[index] and "terminal_observation" in info:
                true_observations[index] = info["terminal_observation"]
        return (
            self._tensor(observations, torch.float32),
            self._tensor(rewards, torch.float32),
            self._tensor(dones, torch.bool),
            {
                "time_outs": self._tensor(truncations, torch.bool),
                "observations": {"raw": {"obs": self._tensor(true_observations, torch.float32)}},
                "gymnasium": raw_infos,
            },
        )

    def close(self):
        self.envs.close()
