"""Small fixed-topology masked controller suitable for CMA-ES."""

from __future__ import annotations

import numpy as np


class CompactBattleController:
    def __init__(self, observation_space, action_count: int, hidden_size: int = 16):
        self.feature_keys = tuple(key for key in observation_space.spaces if key != "action_mask")
        self.input_size = sum(
            int(np.prod(observation_space[key].shape)) for key in self.feature_keys
        )
        self.action_count = action_count
        self.hidden_size = hidden_size
        self.parameter_count = (
            self.input_size * hidden_size + hidden_size + hidden_size * action_count + action_count
        )
        self.set_parameters(np.zeros(self.parameter_count))

    def set_parameters(self, flat_parameters):
        flat = np.asarray(flat_parameters, dtype=np.float64)
        if flat.shape != (self.parameter_count,):
            raise ValueError(
                f"Expected {self.parameter_count} controller parameters, got {flat.shape}"
            )
        offset = 0
        size = self.input_size * self.hidden_size
        self.input_weights = flat[offset : offset + size].reshape(self.input_size, self.hidden_size)
        offset += size
        self.input_bias = flat[offset : offset + self.hidden_size]
        offset += self.hidden_size
        size = self.hidden_size * self.action_count
        self.output_weights = flat[offset : offset + size].reshape(
            self.hidden_size, self.action_count
        )
        offset += size
        self.output_bias = flat[offset : offset + self.action_count]

    def predict(self, observation) -> int:
        features = np.concatenate(
            [
                np.asarray(observation[key], dtype=np.float64).reshape(-1)
                for key in self.feature_keys
            ]
        )
        hidden = np.tanh(features @ self.input_weights + self.input_bias)
        scores = hidden @ self.output_weights + self.output_bias
        legal = np.asarray(observation["action_mask"], dtype=bool)
        if not legal.any():
            legal = legal.copy()
            legal[0] = True
        return int(np.argmax(np.where(legal, scores, -np.inf)))
