"""Lazy design-first adapter for upstream Stackelberg PPO."""

from __future__ import annotations

from importlib import import_module
from types import SimpleNamespace

import numpy as np


def _load_factory(path: str):
    module, name = path.split(":", 1)
    return getattr(import_module(module), name)


class LazyStackelbergPokemonEnv:
    """Expose six Pokemon nodes without starting a pre-design battle."""

    def __init__(self, cfg, agent=None):
        specs = dict(cfg.env_specs)
        self.cfg = cfg
        self.agent = agent
        self.problem = _load_factory(specs["factory"])(specs.get("problem", {}))
        self._seed = int(getattr(cfg, "seed", 0))
        self.np_random = np.random.RandomState(self._seed)
        self.initial_design = np.asarray(self.problem.design, dtype=np.float64)
        self.design_low = np.broadcast_to(self.problem.design_low, self.initial_design.shape).copy()
        self.design_high = np.broadcast_to(
            self.problem.design_high, self.initial_design.shape
        ).copy()
        self.num_nodes, self.attr_design_dim = self.initial_design.shape
        self.fixed_features = np.asarray(self.problem.fixed_features, dtype=np.float64)
        self.attr_fixed_dim = self.fixed_features.shape[1]
        self.edges = np.asarray(self.problem.edges, dtype=np.int64)
        self.control_action_dim = int(self.problem.control_action_dim)
        self.sim_obs_dim = int(self.problem.observation_dimension)
        self.skel_num_action = 1
        self.robot = SimpleNamespace(bodies=list(range(self.num_nodes)))
        self.end_reward = 0.0
        self.design = self.initial_design.copy()
        self.stage = "attribute_transform"
        self._sim_obs = np.zeros((self.num_nodes, self.sim_obs_dim), dtype=np.float64)
        self._distances = self._shortest_path_distances()
        self._depths = self._distances[0].astype(np.int64)
        self.reset_tracking()

    def seed(self, seed=None):
        self._seed = self._seed if seed is None else int(seed)
        self.np_random = np.random.RandomState(self._seed)
        return [self._seed]

    def reset(self):
        self.problem.close()
        self.design = self.initial_design.copy()
        self.stage = "attribute_transform"
        self._sim_obs.fill(0)
        return self._get_obs()

    def reset_tracking(self):
        self.episode_return = 0.0
        self.completed_returns = []
        self.wins = []
        self.turn_counts = []
        self.episode_lengths = []
        self.current_episode_length = 0
        self.environment_transitions = 0
        self.best_return = -np.inf
        self.best_design = None

    def step(self, action):
        action = np.asarray(action, dtype=np.float64)
        expected = (self.num_nodes, self.control_action_dim + self.attr_design_dim + 1)
        if action.shape != expected:
            raise ValueError(f"Expected action shape {expected}, got {action.shape}")
        if self.stage == "attribute_transform":
            raw_design = np.clip((action[:, self.control_action_dim : -1] + 1.0) / 2.0, 0.0, 1.0)
            self.design = self.design_low + raw_design * (self.design_high - self.design_low)
            episode_seed = int(self.np_random.randint(0, 2**31 - 1))
            self._sim_obs = self.problem.reset(self.design, episode_seed)
            self.stage = "execution"
            self.environment_transitions += 1
            return (
                self._get_obs(),
                0.0,
                False,
                False,
                {
                    **self._info(True),
                    "selected_design": self.design.copy(),
                },
            )
        self._sim_obs, reward, terminated, truncated, info = self.problem.step(
            action[:, : self.control_action_dim]
        )
        self.environment_transitions += 1
        self.current_episode_length += 1
        self.episode_return += float(reward)
        if terminated or truncated:
            self.completed_returns.append(self.episode_return)
            self.episode_lengths.append(self.current_episode_length)
            won = info.get("won", info.get("win"))
            turns = info.get("turn", info.get("turn_count"))
            self.wins.append(None if won is None else bool(won))
            self.turn_counts.append(None if turns is None else int(turns))
            if self.episode_return > self.best_return:
                self.best_return = self.episode_return
                self.best_design = self.design.copy()
            self.episode_return = 0.0
            self.current_episode_length = 0
        return (
            self._get_obs(),
            reward,
            terminated,
            truncated,
            {**dict(info), **self._info(False), "selected_design": self.design.copy()},
        )

    def _info(self, transform):
        return {
            "stage": "attribute_transform" if transform else "execution",
            "use_transform_action": transform,
            "reward_ctrl": 0.0,
        }

    def _get_obs(self):
        observation = np.concatenate((self.fixed_features, self._sim_obs, self.design), axis=1)
        return [
            observation,
            self.edges.copy(),
            np.asarray([1 if self.stage == "attribute_transform" else 2]),
            np.asarray([self.num_nodes]),
            np.arange(self.num_nodes),
            self._depths.copy(),
            np.zeros(self.num_nodes),
            self._distances.copy(),
            np.zeros((self.num_nodes, 1)),
        ]

    def _shortest_path_distances(self):
        distances = np.full((self.num_nodes, self.num_nodes), np.inf)
        np.fill_diagonal(distances, 0)
        for source, target in self.edges.T:
            distances[source, target] = distances[target, source] = 1
        for middle in range(self.num_nodes):
            distances = np.minimum(
                distances, distances[:, middle, None] + distances[None, middle, :]
            )
        distances[~np.isfinite(distances)] = self.num_nodes
        return distances
