"""Evaluation helpers shared by population-based Pokemon baselines."""

from __future__ import annotations

import numpy as np


def evaluate_candidate_population(model, env, episodes: int) -> np.ndarray:
    """Evaluate every vector slot for the same number of complete battles."""

    if episodes < 1:
        raise ValueError("episodes must be at least one")
    observations = env.reset()
    episode_counts = np.zeros(env.num_envs, dtype=np.int64)
    return_sums = np.zeros(env.num_envs, dtype=np.float64)
    while np.any(episode_counts < episodes):
        actions, _ = model.predict(observations, deterministic=True)
        observations, rewards, dones, _ = env.step(actions)
        active = episode_counts < episodes
        return_sums[active] += np.asarray(rewards, dtype=np.float64)[active]
        episode_counts += np.asarray(dones, dtype=np.int64) * active
    return return_sums / episodes
