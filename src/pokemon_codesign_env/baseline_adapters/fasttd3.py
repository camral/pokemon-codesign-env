"""Phase-dependent action projection required by Pokemon FastTD3."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def project_actions(
    actions: np.ndarray,
    design_phase: bool | np.ndarray,
    embodiment_dimension: int,
) -> np.ndarray:
    """Zero the action block that cannot execute in the current environment phase."""

    projected = np.asarray(actions).copy()
    if projected.ndim not in {1, 2}:
        raise ValueError("actions must be a vector or a batch of vectors")
    if not 0 < embodiment_dimension < projected.shape[-1]:
        raise ValueError("embodiment_dimension must split the action vector")

    single = projected.ndim == 1
    batch = projected[None, :] if single else projected
    phases = np.asarray(design_phase, dtype=bool)
    if phases.ndim == 0:
        phases = np.full(batch.shape[0], bool(phases))
    if phases.shape != (batch.shape[0],):
        raise ValueError("design_phase must contain one flag per action")

    batch[phases, embodiment_dimension:] = 0
    batch[~phases, :embodiment_dimension] = 0
    return batch[0] if single else batch


def project_tensor_actions(actions, design_phase, embodiment_dimension: int):
    """Differentiably apply the same projection to actor and target-policy tensors."""

    import torch

    if not isinstance(actions, torch.Tensor):
        raise TypeError("actions must be a torch.Tensor")
    if actions.ndim not in {1, 2}:
        raise ValueError("actions must be a vector or a batch of vectors")
    if not 0 < embodiment_dimension < actions.shape[-1]:
        raise ValueError("embodiment_dimension must split the action vector")

    single = actions.ndim == 1
    batch = actions.unsqueeze(0) if single else actions
    phases = torch.as_tensor(design_phase, dtype=torch.bool, device=actions.device)
    if phases.ndim == 0:
        phases = phases.expand(batch.shape[0])
    if tuple(phases.shape) != (batch.shape[0],):
        raise ValueError("design_phase must contain one flag per action")

    embodiment_mask = phases[:, None].expand(-1, embodiment_dimension)
    battle_mask = (~phases)[:, None].expand(-1, batch.shape[-1] - embodiment_dimension)
    mask = torch.cat((embodiment_mask, battle_mask), dim=-1).to(dtype=actions.dtype)
    projected = batch * mask
    return projected[0] if single else projected


def replay_actions(proposed_actions: np.ndarray, infos: dict | Sequence[dict]) -> np.ndarray:
    """Extract the post-discard actions that must be written to replay."""

    proposed = np.asarray(proposed_actions)
    if isinstance(infos, dict):
        if "executed_actions" not in infos:
            raise KeyError("Vector infos do not contain executed_actions")
        executed = np.asarray(infos["executed_actions"], dtype=proposed.dtype)
    else:
        try:
            executed = np.asarray(
                [info["executed_actions"] for info in infos], dtype=proposed.dtype
            )
        except KeyError as error:
            raise KeyError("An environment info does not contain executed_actions") from error
    if executed.shape != proposed.shape:
        raise ValueError(f"Executed actions have shape {executed.shape}; expected {proposed.shape}")
    return executed
