"""Generate, validate, and evaluate Pokemon Transformer Transformer teams."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch
from poke_env.ps_client import ServerConfiguration

from pokemon_codesign_env.baseline_adapters import (
    TeamDesignVectorAdapter,
    make_all_attribute_search_config,
)
from pokemon_codesign_env.baseline_artifacts import RunMetrics, write_run_artifacts
from pokemon_codesign_env.configs import FormatConfig, ObservationConfig
from pokemon_codesign_env.evaluation import SeededResetEnv, evaluate_isolated
from pokemon_codesign_env.fasttd3_factory import ENVIRONMENTS
from pokemon_codesign_env.observations import HumanVisibleBattleEncoder
from pokemon_codesign_env.team_designer import TeamDesigner
from pokemon_codesign_env.transformer_data import (
    encode_team_modalities,
    flatten_battle_observation,
    native_team_features,
)
from pokemon_codesign_env.transformer_model import PokemonTransformerTransformer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--environment", choices=ENVIRONMENTS, default="random")
    parser.add_argument("--candidates", type=int, default=16)
    parser.add_argument("--episodes", type=int, default=4, help="Selection battles per candidate")
    parser.add_argument("--held-out-episodes", type=int, default=16)
    parser.add_argument("--design-noise", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--showdown-host", default="127.0.0.1")
    parser.add_argument("--showdown-port", type=int, default=8000)
    parser.add_argument("--evaluation-timeout", type=float, default=60.0)
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
    server = ServerConfiguration(
        f"ws://{args.showdown_host}:{args.showdown_port}/showdown/websocket",
        "https://play.pokemonshowdown.com/action.php?",
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
    invalid_candidates = 0

    native_vocabularies = checkpoint.get("native_vocabularies")
    native_scales = checkpoint.get("native_scales")

    def action_selector(design, parameters):
        native_team = None
        if model.native_input is not None:
            if native_vocabularies is None or native_scales is None:
                raise ValueError("Native-feature checkpoint metadata is missing")
            native_team, _ = native_team_features(
                *encode_team_modalities(parameters, native_vocabularies),
                scales=np.asarray(native_scales),
            )

        def select(observation):
            state = flatten_battle_observation(observation, keys)
            with torch.no_grad():
                logits, _ = model(
                    torch.as_tensor(design[None, :], dtype=torch.float32, device=device),
                    torch.as_tensor(state[None, :], device=device),
                    torch.full((1,), task_id, dtype=torch.long, device=device),
                    torch.as_tensor(observation["action_mask"][None, :], device=device),
                    None
                    if native_team is None
                    else torch.as_tensor(native_team[None, :], device=device),
                )
            return int(logits.argmax(dim=-1).item())

        return select

    def environment_factory(parameters, seed_base, replay_path=False):
        def make(episode):
            evaluation_seed = seed_base + episode
            designer = TeamDesigner(adapter.search_config, validate_teams=True)
            designer.rng = np.random.default_rng(evaluation_seed)
            designer.set_optimizable_parameter_values(parameters)
            team = designer.yield_team()
            return SeededResetEnv(
                environment_class(
                    team=team,
                    observation_encoder=encoder,
                    server_configuration=server,
                    challenge_timeout=args.evaluation_timeout,
                    save_replays=replay_path,
                ),
                evaluation_seed,
            )

        return make

    for design in candidates:
        parameters = adapter.decode(design)
        designer = TeamDesigner(adapter.search_config, validate_teams=True)
        designer.rng = np.random.default_rng(args.seed + 10_000)
        designer.set_optimizable_parameter_values(parameters)
        try:
            designer.yield_team()
        except (RuntimeError, ValueError):
            invalid_candidates += 1
            continue
        selection = evaluate_isolated(
            environment_factory(parameters, args.seed + 10_000),
            action_selector(design, parameters),
            args.episodes,
            args.evaluation_timeout,
        )
        evaluated.append((selection.mean_return, parameters, design))
    if not evaluated:
        raise RuntimeError("No generated candidate passed Pokemon Showdown validation")
    selection_return, best_parameters, best_design = max(evaluated, key=lambda item: item[0])
    args.output_dir.mkdir(parents=True, exist_ok=True)
    replay_dir = args.output_dir / "replays"
    replay_dir.mkdir(exist_ok=True)
    held_out = evaluate_isolated(
        environment_factory(best_parameters, args.seed + 100_000, str(replay_dir)),
        action_selector(best_design, best_parameters),
        args.held_out_episodes,
        args.evaluation_timeout,
    )
    training_metrics = checkpoint.get("training_metrics", {})
    torch.save(checkpoint, args.output_dir / "best_controller.pt")
    write_run_artifacts(
        args.output_dir,
        team_parameters=best_parameters,
        team_design_schema=adapter.schema(),
        observation_schema=encoder.get_schema(),
        metrics=RunMetrics(
            mean_return=held_out.mean_return,
            evaluation_battles=args.held_out_episodes,
            training_battles=int(training_metrics.get("training_episodes", 0)),
            environment_transitions=int(training_metrics.get("training_transitions", 0)),
            invalid_candidate_rate=invalid_candidates / args.candidates,
            win_rate=held_out.win_rate,
            mean_episode_length=held_out.mean_episode_length,
            mean_turn_count=held_out.mean_turn_count,
            wall_time_seconds=time.monotonic() - started,
        ),
        metadata={
            "method": "transformer_transformer",
            "selection_battles": len(evaluated) * args.episodes,
            "selection_mean_return": selection_return,
            "timed_out_evaluation_battles": held_out.timed_out_episodes,
        },
    )


if __name__ == "__main__":
    main()
