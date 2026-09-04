"""Per-environment accounting for the FastTD3 integration."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


def vector_info_value(raw_infos, key: str, index: int):
    """Read one value from Gymnasium vector-dict or list-of-dicts infos."""

    if isinstance(raw_infos, (list, tuple)):
        return raw_infos[index].get(key)
    mask = raw_infos.get(f"_{key}")
    values = None if mask is not None and not bool(mask[index]) else raw_infos.get(key)
    if values is not None:
        return values[index]
    final_infos = raw_infos.get("final_info")
    final_mask = raw_infos.get("_final_info")
    if final_infos is not None and (final_mask is None or bool(final_mask[index])):
        final_info = final_infos[index]
        if final_info is not None:
            return final_info.get(key)
    return None


def executed_action_array(raw_infos, num_envs: int) -> np.ndarray:
    actions = [vector_info_value(raw_infos, "executed_actions", index) for index in range(num_envs)]
    if any(action is None for action in actions):
        raise KeyError("Pokemon environments must return executed_actions for every environment")
    return np.asarray(actions, dtype=np.float32)


@dataclass
class FastTD3RunTracker:
    num_envs: int
    episode_returns: np.ndarray = field(init=False)
    active_parameters: list[dict | None] = field(init=False)
    completed_returns: list[float] = field(default_factory=list)
    best_return: float = -np.inf
    best_parameters: dict | None = None
    invalid_designs: int = 0
    designs: int = 0
    environment_transitions: int = 0

    def __post_init__(self):
        self.episode_returns = np.zeros(self.num_envs, dtype=np.float64)
        self.active_parameters = [None] * self.num_envs

    def update(self, raw_infos, rewards, dones) -> None:
        rewards = np.asarray(rewards, dtype=np.float64)
        dones = np.asarray(dones, dtype=bool)
        self.episode_returns += rewards
        self.environment_transitions += self.num_envs
        for index in range(self.num_envs):
            if vector_info_value(raw_infos, "phase", index) == "design":
                self.designs += 1
                if vector_info_value(raw_infos, "invalid_design", index):
                    self.invalid_designs += 1
                else:
                    design = vector_info_value(raw_infos, "selected_design", index)
                    if design is not None:
                        self.active_parameters[index] = np.asarray(design).copy()
            if dones[index]:
                episode_return = float(self.episode_returns[index])
                self.completed_returns.append(episode_return)
                if self.active_parameters[index] is not None and episode_return > self.best_return:
                    self.best_return = episode_return
                    self.best_parameters = self.active_parameters[index]
                self.episode_returns[index] = 0.0
                self.active_parameters[index] = None

    @property
    def invalid_candidate_rate(self) -> float:
        return self.invalid_designs / self.designs if self.designs else 0.0

    @property
    def mean_return(self) -> float | None:
        return float(np.mean(self.completed_returns)) if self.completed_returns else None
