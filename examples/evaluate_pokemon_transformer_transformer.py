"""Generate, validate, and evaluate Pokemon Transformer Transformer teams."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch

from pokemon_codesign_env.baseline_adapters import (
    TeamDesignVectorAdapter,
    make_all_attribute_search_config,
)
from pokemon_codesign_env.baseline_artifacts import RunMetrics, write_run_artifacts
from pokemon_codesign_env.configs import FormatConfig, ObservationConfig
from pokemon_codesign_env.fasttd3_factory import ENVIRONMENTS
from pokemon_codesign_env.observations import HumanVisibleBattleEncoder
from pokemon_codesign_env.team_designer import TeamDesigner
from pokemon_codesign_env.transformer_data import flatten_battle_observation
from pokemon_codesign_env.transformer_model import PokemonTransformerTransformer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--environment", choices=ENVIRONMENTS, default="random")
    parser.add_argument("--candidates", type=int, default=16)
    parser.add_argument("--episodes", type=int, default=4)
    parser.add_argument("--design-noise", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output-dir", type=Path, default=Path("runs/pokemon_transformerer_eval"))
    args = parser.parse_args()
    device = torch.device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model = PokemonTransformerTransformer(**checkpoint["configuration"]).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    task_vocabulary = checkpoint.get("task_vocabulary", {args.environment: 0})
    if args.environment not in task_vocabulary:
        raise ValueError(
            f"Checkpoint has no task token for {args.environment!r}; available: "
            f"{sorted(task_vocabulary)}"
        )
    task_id = int(task_vocabulary[args.environment])
    environment_class = ENVIRONMENTS[args.environment]
    format_config = FormatConfig.from_showdown(environment_class.BATTLE_FORMAT)
    adapter = TeamDesignVectorAdapter(make_all_attribute_search_config(format_config))
    encoder = HumanVisibleBattleEncoder.from_showdown(
        format_config, observation_config=ObservationConfig(team_size=6)
    )
    keys = tuple(encoder.observation_space.spaces)
    rng = np.random.default_rng(args.seed)
    with torch.no_grad():
        centre = model.generate_design(torch.full((1,), task_id, dtype=torch.long, device=device))[
            0
        ]
    candidates = np.clip(
        centre.cpu().numpy()[None, :]
        + rng.normal(0, args.design_noise, size=(args.candidates, adapter.dimension)),
        0,
        1,
    )
    started = time.monotonic()
    evaluated = []
    for candidate_index, design in enumerate(candidates):
        parameters = adapter.decode(design)
        designer = TeamDesigner(adapter.search_config, validate_teams=True)
        designer.rng = np.random.default_rng(args.seed + candidate_index)
        designer.set_optimizable_parameter_values(parameters)
        try:
            team = designer.yield_team()
        except ValueError:
            continue
        returns = []
        for episode in range(args.episodes):
            env = environment_class(team=team, observation_encoder=encoder)
            try:
                observation, _ = env.reset(
                    seed=args.seed + candidate_index * args.episodes + episode
                )
                terminated = truncated = False
                episode_return = 0.0
                while not terminated and not truncated:
                    state = flatten_battle_observation(observation, keys)
                    with torch.no_grad():
                        logits, _ = model(
                            torch.as_tensor(design[None, :], dtype=torch.float32, device=device),
                            torch.as_tensor(state[None, :], device=device),
                            torch.full((1,), task_id, dtype=torch.long, device=device),
                            torch.as_tensor(observation["action_mask"][None, :], device=device),
                        )
                    action = int(logits.argmax(dim=-1).item())
                    observation, reward, terminated, truncated, _ = env.step(action)
                    episode_return += float(reward)
                returns.append(episode_return)
            finally:
                env.close()
        evaluated.append((float(np.mean(returns)), parameters))
    if not evaluated:
        raise RuntimeError("No generated candidate passed Pokemon Showdown validation")
    best_return, best_parameters = max(evaluated, key=lambda item: item[0])
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, args.output_dir / "best_controller.pt")
    write_run_artifacts(
        args.output_dir,
        team_parameters=best_parameters,
        team_design_schema=adapter.schema(),
        observation_schema=encoder.get_schema(),
        metrics=RunMetrics(
            mean_return=best_return,
            evaluation_battles=len(evaluated) * args.episodes,
            invalid_candidate_rate=1 - len(evaluated) / args.candidates,
            wall_time_seconds=time.monotonic() - started,
        ),
        metadata={"method": "transformer_transformer"},
    )


if __name__ == "__main__":
    main()
