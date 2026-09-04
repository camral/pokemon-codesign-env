"""Six-node Pokemon backend for the Stackelberg PPO baseline."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from pokemon_codesign_env.baseline_adapters.team_design import TeamDesignVectorAdapter
from pokemon_codesign_env.team_designer import TeamDesigner


class PokemonStackelbergProblem:
    """Leader-selected team followed by global masked battle control."""

    def __init__(
        self,
        adapter: TeamDesignVectorAdapter,
        battle_observation_space,
        battle_action_count: int,
        battle_env_factory: Callable[[str], object],
    ):
        if adapter.search_config.team_size != 6 or adapter.dimension % 6:
            raise ValueError("Stackelberg PPO requires six equal Pokemon design blocks")
        self.adapter = adapter
        self.battle_observation_space = battle_observation_space
        self.battle_env_factory = battle_env_factory
        self.control_action_dim = battle_action_count
        self.design_width = adapter.dimension // 6
        self.design = np.full((6, self.design_width), 0.5, dtype=np.float64)
        self.design_low = np.zeros_like(self.design)
        self.design_high = np.ones_like(self.design)
        self.fixed_features = np.eye(6, dtype=np.float64)
        self.edges = np.asarray(
            [
                [source for source in range(6) for target in range(6) if source != target],
                [target for source in range(6) for target in range(6) if source != target],
            ],
            dtype=np.int64,
        )
        self._observation_keys = tuple(battle_observation_space.spaces)
        self.observation_dimension = sum(
            int(np.prod(battle_observation_space[key].shape)) for key in self._observation_keys
        )
        self._env = None
        self._observation = None

    def _node_observation(self):
        flat = np.concatenate(
            [
                np.asarray(self._observation[key], dtype=np.float64).reshape(-1)
                for key in self._observation_keys
            ]
        )
        return np.repeat(flat[None, :], 6, axis=0)

    def reset(self, design: np.ndarray, seed: int | None = None) -> np.ndarray:
        if self._env is not None:
            self._env.close()
        design = np.clip(np.asarray(design, dtype=np.float64), 0.0, 1.0)
        if design.shape != self.design.shape:
            raise ValueError(f"Expected design shape {self.design.shape}, got {design.shape}")
        parameters = self.adapter.decode(design.reshape(-1))
        designer = TeamDesigner(self.adapter.search_config, validate_teams=True)
        designer.rng = np.random.default_rng(seed)
        designer.set_optimizable_parameter_values(parameters)
        self._env = self.battle_env_factory(designer.yield_team())
        self._observation, _ = self._env.reset(seed=seed)
        self.design = design.copy()
        return self._node_observation()

    def step(self, control: np.ndarray):
        if self._env is None or self._observation is None:
            raise RuntimeError("The leader must select a design before follower control")
        control = np.asarray(control, dtype=np.float64)
        if control.shape != (6, self.control_action_dim):
            raise ValueError(
                f"Expected control shape {(6, self.control_action_dim)}, got {control.shape}"
            )
        # Every node votes on the one global Showdown action; averaging is
        # permutation-neutral and preserves a differentiable global score head.
        scores = control.mean(axis=0)
        legal = np.asarray(self._observation["action_mask"], dtype=bool)
        if not legal.any():
            legal = legal.copy()
            legal[0] = True
        selected_action = int(np.argmax(np.where(legal, scores, -np.inf)))
        observation, reward, terminated, truncated, info = self._env.step(np.int64(selected_action))
        self._observation = observation
        return (
            self._node_observation(),
            float(reward),
            bool(terminated),
            bool(truncated),
            {**dict(info or {}), "selected_battle_action": selected_action},
        )

    def close(self):
        if self._env is not None:
            self._env.close()
            self._env = None
