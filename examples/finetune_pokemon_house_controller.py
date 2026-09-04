"""Fine-tune and evaluate House of Dextra's controller on its winning Pokemon team."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

from pokemon_codesign_env.baseline_artifacts import RunMetrics, write_run_artifacts
from pokemon_codesign_env.configs import FormatConfig
from pokemon_codesign_env.fasttd3_factory import ENVIRONMENTS
from pokemon_codesign_env.house_problem import make_problem, train_house_family
from pokemon_codesign_env.scaling import ShowdownClusterConfig, managed_showdown_cluster


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--design", type=Path, required=True)
    parser.add_argument("--controller-manifest", type=Path, required=True)
    parser.add_argument("--num-actors", type=int, default=2048)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--horizon", type=int, default=32)
    parser.add_argument("--minibatch-size", type=int, default=8192)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--evaluation-episodes", type=int, default=16)
    parser.add_argument("--evaluation-actors", type=int, default=1)
    parser.add_argument("--evaluation-max-steps", type=int, default=100000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--launch-server", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument(
        "--baseline-root",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "codesign-baselines",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("runs/pokemon_house/finetuned"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sys.path.insert(0, str(args.baseline_root.resolve()))
    from house_of_dextra.general_codesign import train_controller
    from house_of_dextra.general_codesign.controller import (
        MANIFEST_VERSION,
        ControllerEvaluator,
        load_manifest,
        resolve_checkpoint,
    )

    started = time.monotonic()
    base_manifest = load_manifest(args.controller_manifest)
    config = dict(base_manifest.get("problem_config", {}))
    problem = make_problem(config)
    payload = json.loads(args.design.read_text(encoding="utf-8"))
    design = problem.deserialize(payload.get("design", payload))
    family = problem.family(design)
    if family not in base_manifest["families"]:
        raise SystemExit(f"manifest has no base controller for family {family!r}")
    environment = config.get("environment", "random")
    format_config = FormatConfig.from_showdown(ENVIRONMENTS[environment].BATTLE_FORMAT)
    cluster = ShowdownClusterConfig(
        config.get("showdown_host", "127.0.0.1"),
        int(config.get("showdown_port", 8000)),
        int(config.get("showdown_shards", 1)),
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with managed_showdown_cluster(
        args.launch_server, format_config.showdown_path, cluster, args.verbose
    ):
        zero_shot = ControllerEvaluator(
            problem,
            base_manifest,
            device=args.device,
            episodes=args.evaluation_episodes,
            actors_per_design=args.evaluation_actors,
            max_steps=args.evaluation_max_steps,
            seed=args.seed + 100_000,
        ).evaluate([design])[0]
        entry = train_house_family(
            train_controller,
            problem=problem,
            family=family,
            output_dir=args.output_dir,
            num_designs=1,
            num_actors=args.num_actors,
            epochs=args.epochs,
            horizon=args.horizon,
            minibatch_size=args.minibatch_size,
            learning_rate=args.learning_rate,
            seed=args.seed,
            device=args.device,
            recurrent=bool(base_manifest["families"][family]["training"]["recurrent"]),
            initial_checkpoint=resolve_checkpoint(base_manifest, family),
            fixed_design=design,
        )
        fine_manifest_payload = {
            "format_version": MANIFEST_VERSION,
            "problem": "pokemon_codesign_env.house_problem:make_problem",
            "problem_config": config,
            "design": problem.serialize(design),
            "families": {family: entry},
        }
        fine_manifest_path = args.output_dir / "manifest.json"
        fine_manifest_path.write_text(
            json.dumps(fine_manifest_payload, indent=2) + "\n", encoding="utf-8"
        )
        fine_manifest = load_manifest(fine_manifest_path)
        problem.runtime_config["save_replays"] = str(args.output_dir / "replays")
        fine_tuned = ControllerEvaluator(
            problem,
            fine_manifest,
            device=args.device,
            episodes=args.evaluation_episodes,
            actors_per_design=args.evaluation_actors,
            max_steps=args.evaluation_max_steps,
            seed=args.seed + 200_000,
        ).evaluate([design])[0]

    shutil.copy2(resolve_checkpoint(fine_manifest, family), args.output_dir / "best_controller.pth")
    evaluation_battles = 2 * args.evaluation_episodes * args.evaluation_actors
    write_run_artifacts(
        args.output_dir,
        team_parameters=problem.serialize(design),
        team_design_schema=problem.vector_adapter.schema(),
        observation_schema=problem.encoder.get_schema(),
        metrics=RunMetrics(
            mean_return=float(fine_tuned),
            evaluation_battles=evaluation_battles,
            environment_transitions=args.epochs * args.num_actors * args.horizon,
            wall_time_seconds=time.monotonic() - started,
        ),
        metadata={
            "method": "house_of_dextra",
            "zero_shot_mean_return": float(zero_shot),
            "fine_tuned_mean_return": float(fine_tuned),
            "zero_shot_evaluation_battles": args.evaluation_episodes * args.evaluation_actors,
            "fine_tuned_evaluation_battles": args.evaluation_episodes * args.evaluation_actors,
            "base_controller_manifest": str(args.controller_manifest.resolve()),
        },
    )


if __name__ == "__main__":
    main()
