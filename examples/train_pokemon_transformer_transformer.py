"""Train Pokemon control and team-generation heads over one Transformer backbone."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as functional

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
    tasks = np.zeros_like(episode_ids)
    device = torch.device(args.device)
    model = PokemonTransformerTransformer(
        design.shape[1], state.shape[1], masks.shape[1], task_count=1
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    rng = np.random.default_rng(0)
    for _ in range(args.epochs):
        permutation = rng.permutation(len(design))
        for start in range(0, len(design), args.batch_size):
            indices = permutation[start : start + args.batch_size]
            batch_design = torch.as_tensor(design[indices], device=device)
            logits, generated = model(
                batch_design,
                torch.as_tensor(state[indices], device=device),
                torch.as_tensor(tasks[indices], device=device),
                torch.as_tensor(masks[indices], device=device),
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
    torch.save(
        {"configuration": model.configuration, "state_dict": model.state_dict()}, args.output
    )


if __name__ == "__main__":
    main()
