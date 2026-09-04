"""Train Pokemon control and team-generation heads over one Transformer backbone."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as functional

from pokemon_codesign_env.transformer_data import native_team_features
from pokemon_codesign_env.transformer_model import PokemonTransformerTransformer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("runs/pokemon_transformer/model.pt"))
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    import zarr

    root = zarr.open_group(str(args.dataset), mode="r")
    design = np.asarray(root["design"])
    state = np.asarray(root["battle_state"])
    masks = np.asarray(root["action_mask"])
    actions = np.asarray(root["control"])
    episode_ids = np.asarray(root["episode_id"])
    held_out = np.asarray(root["held_out"], dtype=bool)
    episode_tasks = json.loads(root.attrs["opponent_tasks"])
    task_vocabulary = {task: index for index, task in enumerate(sorted(set(episode_tasks)))}
    tasks = np.asarray(
        [task_vocabulary[episode_tasks[index]] for index in episode_ids], dtype=np.int64
    )
    native_arrays = (
        np.asarray(root["pokemon"])[episode_ids],
        np.asarray(root["move"])[episode_ids],
        np.asarray(root["ev"])[episode_ids],
        np.asarray(root["iv"])[episode_ids],
    )
    training_indices = np.flatnonzero(~held_out)
    held_out_indices = np.flatnonzero(held_out)
    if not len(training_indices) or not len(held_out_indices):
        raise ValueError("Dataset must contain both training and held-out trajectories")
    _, native_scales = native_team_features(*(value[training_indices] for value in native_arrays))
    native_features, _ = native_team_features(*native_arrays, scales=native_scales)
    device = torch.device(args.device)
    model = PokemonTransformerTransformer(
        design.shape[1],
        state.shape[1],
        masks.shape[1],
        task_count=len(task_vocabulary),
        native_dimension=native_features.shape[1],
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    rng = np.random.default_rng(0)
    for _ in range(args.epochs):
        permutation = rng.permutation(training_indices)
        for start in range(0, len(permutation), args.batch_size):
            indices = permutation[start : start + args.batch_size]
            batch_design = torch.as_tensor(design[indices], device=device)
            logits, generated = model(
                batch_design,
                torch.as_tensor(state[indices], device=device),
                torch.as_tensor(tasks[indices], device=device),
                torch.as_tensor(masks[indices], device=device),
                torch.as_tensor(native_features[indices], device=device),
            )
            control_loss = functional.cross_entropy(
                logits, torch.as_tensor(actions[indices], device=device)
            )
            design_loss = functional.mse_loss(generated, batch_design)
            loss = control_loss + design_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        held_logits, held_design = model(
            torch.as_tensor(design[held_out_indices], device=device),
            torch.as_tensor(state[held_out_indices], device=device),
            torch.as_tensor(tasks[held_out_indices], device=device),
            torch.as_tensor(masks[held_out_indices], device=device),
            torch.as_tensor(native_features[held_out_indices], device=device),
        )
        held_control_loss = functional.cross_entropy(
            held_logits, torch.as_tensor(actions[held_out_indices], device=device)
        )
        held_design_loss = functional.mse_loss(
            held_design, torch.as_tensor(design[held_out_indices], device=device)
        )
    torch.save(
        {
            "configuration": model.configuration,
            "state_dict": model.state_dict(),
            "task_vocabulary": task_vocabulary,
            "native_vocabularies": json.loads(root.attrs["vocabularies"]),
            "native_scales": native_scales.tolist(),
            "training_metrics": {
                "training_transitions": len(training_indices),
                "held_out_transitions": len(held_out_indices),
                "training_episodes": len(set(episode_ids[training_indices].tolist())),
                "held_out_episodes": len(set(episode_ids[held_out_indices].tolist())),
            },
        },
        args.output,
    )
    (args.output.parent / "metrics.json").write_text(
        json.dumps(
            {
                "held_out_control_loss": float(held_control_loss.item()),
                "held_out_design_loss": float(held_design_loss.item()),
                "training_transitions": len(training_indices),
                "held_out_transitions": len(held_out_indices),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
