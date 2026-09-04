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


def held_out_episode_indices(episodes: int, fraction: float, seed: int) -> set[int]:
    """Choose a reproducible non-empty split while preserving training data."""

    if episodes < 2:
        raise ValueError("At least two episodes are required for train/held-out splits")
    if not 0 < fraction < 1:
        raise ValueError("held-out fraction must be strictly between zero and one")
    count = min(max(round(episodes * fraction), 1), episodes - 1)
    selected = np.random.default_rng(seed).choice(episodes, size=count, replace=False)
    return {int(index) for index in selected}


def encode_team_modalities(parameters: dict, vocabularies: dict) -> tuple[np.ndarray, ...]:
    """Encode one canonical TeamDesigner team with dataset vocabularies."""

    categorical = ("species", "item", "nature", "ability", "gender", "tera_type")
    pokemon_tokens = np.zeros((6, 7), dtype=np.int64)
    move_tokens = np.zeros((6, 4), dtype=np.int64)
    evs = np.zeros((6, 6), dtype=np.int64)
    ivs = np.zeros_like(evs)
    stats = ("hp", "atk", "def", "spa", "spd", "spe")
    for slot in range(6):
        pokemon = parameters[f"pokemon_{slot}"]
        pokemon_tokens[slot, :6] = [
            vocabularies[attribute].get(pokemon.get(attribute), 0) for attribute in categorical
        ]
        pokemon_tokens[slot, 6] = int(pokemon["happiness"])
        move_tokens[slot, : len(pokemon["moves"][:4])] = [
            vocabularies["move"].get(move, 0) for move in pokemon["moves"][:4]
        ]
        evs[slot] = [pokemon["evs"][stat] for stat in stats]
        ivs[slot] = [pokemon["ivs"][stat] for stat in stats]
    return pokemon_tokens, move_tokens, evs, ivs


def native_team_features(
    pokemon: np.ndarray,
    move: np.ndarray,
    ev: np.ndarray,
    iv: np.ndarray,
    scales: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Flatten native modalities and consistently scale each feature column."""

    arrays = [np.asarray(value, dtype=np.float32) for value in (pokemon, move, ev, iv)]
    unbatched = arrays[0].ndim == 2
    if unbatched:
        arrays = [value[None, ...] for value in arrays]
    features = np.concatenate([value.reshape(len(value), -1) for value in arrays], axis=1)
    if scales is None:
        scales = np.maximum(features.max(axis=0), 1.0)
    scales = np.asarray(scales, dtype=np.float32)
    if scales.shape != (features.shape[1],):
        raise ValueError(f"Expected {features.shape[1]} native scales, got {scales.shape}")
    normalized = features / scales
    return (normalized[0] if unbatched else normalized), scales


class PokemonTrajectoryWriter:
    def __init__(self, design_dimension: int, observation_dimension: int, action_count: int):
        self.design_dimension = design_dimension
        self.observation_dimension = observation_dimension
        self.action_count = action_count
        self.rows = []
        self.team_parameters = []
        self.tasks = []

    def add_episode(self, design, parameters, task, transitions, *, held_out: bool = False):
        episode_id = len(self.team_parameters)
        self.team_parameters.append(parameters)
        self.tasks.append(task)
        for step, (observation, mask, action, reward) in enumerate(transitions):
            self.rows.append(
                (episode_id, step, design, observation, mask, action, reward, held_out)
            )

    def write(self, path: Path, *, schema: dict):
        import zarr

        if not self.rows:
            raise ValueError("Cannot write an empty trajectory dataset")
        root = zarr.open_group(str(path), mode="w")
        episode_ids, steps, designs, observations, masks, actions, rewards, held_out = zip(
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
            "held_out": np.asarray(held_out, dtype=np.bool_),
        }
        episode_arrays, vocabularies = self._episode_modalities()
        arrays.update(episode_arrays)
        for name, array in arrays.items():
            root.create_dataset(name, data=array, shape=array.shape, dtype=array.dtype)
        root.attrs["schema"] = json.dumps(schema, sort_keys=True)
        root.attrs["modalities"] = list(POKEMON_MODALITIES)
        root.attrs["team_parameters"] = json.dumps(self.team_parameters, sort_keys=True)
        root.attrs["opponent_tasks"] = json.dumps(self.tasks, sort_keys=True)
        root.attrs["vocabularies"] = json.dumps(vocabularies, sort_keys=True)

    def _episode_modalities(self):
        categorical = ("species", "item", "nature", "ability", "gender", "tera_type")
        vocabularies = {}
        for attribute in (*categorical, "move", "opponent_task"):
            if attribute == "move":
                values = {
                    move
                    for team in self.team_parameters
                    for pokemon in team.values()
                    for move in pokemon["moves"]
                }
            elif attribute == "opponent_task":
                values = set(self.tasks)
            else:
                values = {
                    pokemon.get(attribute)
                    for team in self.team_parameters
                    for pokemon in team.values()
                    if pokemon.get(attribute) is not None
                }
            vocabularies[attribute] = {
                value: index + 1 for index, value in enumerate(sorted(values))
            }

        encoded = [encode_team_modalities(team, vocabularies) for team in self.team_parameters]
        pokemon_tokens, move_tokens, evs, ivs = (
            np.stack([episode[index] for episode in encoded]) for index in range(4)
        )
        task_tokens = np.asarray(
            [vocabularies["opponent_task"][task] for task in self.tasks], dtype=np.int64
        )
        return {
            "pokemon": pokemon_tokens,
            "move": move_tokens,
            "ev": evs,
            "iv": ivs,
            "opponent_task": task_tokens,
        }, vocabularies


def flatten_battle_observation(observation: dict, keys: tuple[str, ...]) -> np.ndarray:
    return np.concatenate(
        [np.asarray(observation[key], dtype=np.float32).reshape(-1) for key in keys]
    )
