"""Native Pokemon trajectory storage for Transformer Transformer."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

POKEMON_MODALITIES = (
    "pokemon",
    "move",
    "ev",
    "iv",
    "battle_state",
    "opponent_task",
    "control",
    "action_mask",
)


class PokemonTrajectoryWriter:
    def __init__(self, design_dimension: int, observation_dimension: int, action_count: int):
        self.design_dimension = design_dimension
        self.observation_dimension = observation_dimension
        self.action_count = action_count
        self.rows = []
        self.team_parameters = []
        self.tasks = []

    def add_episode(self, design, parameters, task, transitions):
        episode_id = len(self.team_parameters)
        self.team_parameters.append(parameters)
        self.tasks.append(task)
        for step, (observation, mask, action, reward) in enumerate(transitions):
            self.rows.append((episode_id, step, design, observation, mask, action, reward))

    def write(self, path: Path, *, schema: dict):
        import zarr

        if not self.rows:
            raise ValueError("Cannot write an empty trajectory dataset")
        root = zarr.open_group(str(path), mode="w")
        episode_ids, steps, designs, observations, masks, actions, rewards = zip(
            *self.rows, strict=True
        )
        arrays = {
            "episode_id": np.asarray(episode_ids, dtype=np.int64),
            "step": np.asarray(steps, dtype=np.int64),
            "design": np.asarray(designs, dtype=np.float32),
            "battle_state": np.asarray(observations, dtype=np.float32),
            "action_mask": np.asarray(masks, dtype=np.bool_),
            "control": np.asarray(actions, dtype=np.int64),
            "reward": np.asarray(rewards, dtype=np.float32),
        }
        for name, array in arrays.items():
            root.create_array(name, data=array)
        root.attrs["schema"] = json.dumps(schema, sort_keys=True)
        root.attrs["modalities"] = list(POKEMON_MODALITIES)
        root.attrs["team_parameters"] = json.dumps(self.team_parameters, sort_keys=True)
        root.attrs["opponent_tasks"] = json.dumps(self.tasks, sort_keys=True)


def flatten_battle_observation(observation: dict, keys: tuple[str, ...]) -> np.ndarray:
    return np.concatenate(
        [np.asarray(observation[key], dtype=np.float32).reshape(-1) for key in keys]
    )
