"""Design-first Pokemon environment for FastTD3."""

from __future__ import annotations

from collections.abc import Callable

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from pokemon_codesign_env.baseline_adapters.fasttd3 import project_actions
from pokemon_codesign_env.baseline_adapters.team_design import TeamDesignVectorAdapter
from pokemon_codesign_env.team_designer import TeamDesigner


class FastTD3PokemonCodesignEnv(gym.Env):
    """Concatenate team design and continuous battle-score actions.

    Reset is deliberately simulator-free. The first step builds the selected
    team and creates the battle environment. Every later step retains that team.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        adapter: TeamDesignVectorAdapter,
        battle_observation_space: spaces.Dict,
        battle_action_count: int,
        battle_env_factory: Callable[[str], gym.Env],
        *,
        invalid_design_reward: float = -100.0,
    ):
        super().__init__()
        if battle_action_count < 1:
            raise ValueError("battle_action_count must be positive")
        self.adapter = adapter
        self.battle_observation_space = battle_observation_space
        self.battle_action_count = battle_action_count
        self.battle_env_factory = battle_env_factory
        self.invalid_design_reward = invalid_design_reward
        self.embodiment_dimension = adapter.dimension
        self.action_space = spaces.Box(
            -1.0,
            1.0,
            shape=(self.embodiment_dimension + battle_action_count,),
            dtype=np.float32,
        )
        self._observation_keys = tuple(battle_observation_space.spaces)
        self._battle_observation_dimension = sum(
            int(np.prod(battle_observation_space[key].shape)) for key in self._observation_keys
        )
        # Battle state, selected normalized design, and one design-phase flag.
        self.observation_space = spaces.Box(
            -np.inf,
            np.inf,
            shape=(self._battle_observation_dimension + self.embodiment_dimension + 1,),
            dtype=np.float32,
        )
        self._battle_env = None
        self._battle_observation = None
        self._design = np.zeros(self.embodiment_dimension, dtype=np.float32)
        self._design_phase = True
        self._parameters = None
        self._candidate_id = None

    def _flat_observation(self) -> np.ndarray:
        if self._battle_observation is None:
            battle = np.zeros(self._battle_observation_dimension, dtype=np.float32)
        else:
            battle = np.concatenate(
                [
                    np.asarray(self._battle_observation[key], dtype=np.float32).reshape(-1)
                    for key in self._observation_keys
                ]
            )
        return np.concatenate(
            (battle, self._design, np.asarray([float(self._design_phase)], dtype=np.float32))
        )

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if self._battle_env is not None:
            self._battle_env.close()
        self._battle_env = None
        self._battle_observation = None
        self._design = np.zeros(self.embodiment_dimension, dtype=np.float32)
        self._design_phase = True
        self._parameters = None
        self._candidate_id = None
        return self._flat_observation(), {"phase": "design"}

    def step(self, action):
        proposed = np.asarray(action, dtype=np.float32)
        if proposed.shape != self.action_space.shape:
            raise ValueError(
                f"Expected action shape {self.action_space.shape}, got {proposed.shape}"
            )
        if self._design_phase:
            return self._design_step(proposed)
        return self._battle_step(proposed)

    def _design_step(self, proposed):
        executed = project_actions(proposed, True, self.embodiment_dimension)
        self._design = np.clip((executed[: self.embodiment_dimension] + 1.0) / 2.0, 0.0, 1.0)
        try:
            parameters = self.adapter.decode(self._design)
            self._parameters = parameters
            candidate_id = getattr(self.adapter, "candidate_id", None)
            self._candidate_id = candidate_id(parameters) if candidate_id else None
            designer = TeamDesigner(self.adapter.search_config, validate_teams=True)
            designer.rng = self.np_random
            designer.set_optimizable_parameter_values(parameters)
            packed_team = designer.yield_team()
            self._battle_env = self.battle_env_factory(packed_team)
            self._battle_observation, reset_info = self._battle_env.reset()
        except (RuntimeError, ValueError) as error:
            info = {
                "phase": "design",
                "invalid_design": True,
                "validation_error": str(error),
                "executed_actions": executed,
                "selected_battle_action": -1,
                "selected_design": self._design.copy(),
                "candidate_id": self._candidate_id,
            }
            return self._flat_observation(), self.invalid_design_reward, True, False, info
        self._design_phase = False
        info = {
            **dict(reset_info or {}),
            "phase": "design",
            "invalid_design": False,
            "executed_actions": executed,
            "selected_battle_action": -1,
            "selected_design": self._design.copy(),
            "candidate_id": self._candidate_id,
        }
        return self._flat_observation(), 0.0, False, False, info

    def _battle_step(self, proposed):
        if self._battle_env is None or self._battle_observation is None:
            raise RuntimeError("A battle action cannot execute before embodiment selection")
        executed = project_actions(proposed, False, self.embodiment_dimension)
        scores = executed[self.embodiment_dimension :]
        legal = np.asarray(self._battle_observation["action_mask"], dtype=bool)
        if legal.shape != (self.battle_action_count,):
            raise ValueError("The Pokemon action mask has the wrong shape")
        if not legal.any():
            legal = legal.copy()
            legal[0] = True
        scores[~legal] = 0.0
        selected_action = int(np.argmax(np.where(legal, scores, -np.inf)))
        self._battle_observation, reward, terminated, truncated, raw_info = self._battle_env.step(
            selected_action
        )
        info = {
            **dict(raw_info or {}),
            "phase": "battle",
            "executed_actions": executed,
            "selected_battle_action": selected_action,
            "selected_design": self._design.copy(),
            "candidate_id": self._candidate_id,
        }
        return self._flat_observation(), float(reward), terminated, truncated, info

    def close(self):
        if self._battle_env is not None:
            self._battle_env.close()
            self._battle_env = None
