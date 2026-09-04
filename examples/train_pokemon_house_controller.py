"""Pretrain House of Dextra's recurrent controller across Pokemon teams."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from pokemon_codesign_env.configs import FormatConfig
from pokemon_codesign_env.fasttd3_factory import ENVIRONMENTS
from pokemon_codesign_env.house_problem import make_problem, train_house_family
from pokemon_codesign_env.scaling import ShowdownClusterConfig, managed_showdown_cluster


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--environment", choices=ENVIRONMENTS, default="random")
    parser.add_argument("--designs-per-family", type=int, default=2048)
    parser.add_argument("--num-actors", type=int, default=2048)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--horizon", type=int, default=32)
    parser.add_argument("--minibatch-size", type=int, default=8192)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--feedforward", action="store_true")
    parser.add_argument("--showdown-host", default="127.0.0.1")
    parser.add_argument("--showdown-port", type=int, default=8000)
    parser.add_argument("--showdown-shards", type=int, default=1)
    parser.add_argument("--envs-per-worker", type=int, default=32)
    parser.add_argument("--challenge-timeout", type=float, default=60.0)
    parser.add_argument("--step-timeout", type=float, default=120.0)
    parser.add_argument("--step-timeout-reward", type=float, default=0.0)
    parser.add_argument("--reset-retries", type=int, default=3)
    parser.add_argument("--start-method", default="spawn")
    parser.add_argument("--launch-server", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument(
        "--baseline-root",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "codesign-baselines",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("runs/pokemon_house/controllers"))
    return parser.parse_args()


def problem_config(args: argparse.Namespace) -> dict:
    return {
        "environment": args.environment,
        "showdown_host": args.showdown_host,
        "showdown_port": args.showdown_port,
        "showdown_shards": args.showdown_shards,
        "envs_per_worker": args.envs_per_worker,
        "challenge_timeout": args.challenge_timeout,
        "step_timeout": args.step_timeout,
        "step_timeout_reward": args.step_timeout_reward,
        "reset_retries": args.reset_retries,
        "start_method": args.start_method,
        "save_replays": False,
    }


def main() -> None:
    args = parse_args()
    sys.path.insert(0, str(args.baseline_root.resolve()))
    from house_of_dextra.general_codesign import train_controller
    from house_of_dextra.general_codesign.controller import MANIFEST_VERSION

    config = problem_config(args)
    problem = make_problem(config)
    family = problem.family_name
    format_config = FormatConfig.from_showdown(ENVIRONMENTS[args.environment].BATTLE_FORMAT)
    cluster = ShowdownClusterConfig(args.showdown_host, args.showdown_port, args.showdown_shards)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with managed_showdown_cluster(
        args.launch_server, format_config.showdown_path, cluster, args.verbose
    ):
        entry = train_house_family(
            train_controller,
            problem=problem,
            family=family,
            output_dir=args.output_dir,
            num_designs=args.designs_per_family,
            num_actors=args.num_actors,
            epochs=args.epochs,
            horizon=args.horizon,
            minibatch_size=args.minibatch_size,
            learning_rate=args.learning_rate,
            seed=args.seed,
            device=args.device,
            recurrent=not args.feedforward,
        )
    manifest = {
        "format_version": MANIFEST_VERSION,
        "problem": "pokemon_codesign_env.house_problem:make_problem",
        "problem_config": config,
        "families": {family: entry},
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    (args.output_dir / "problem_schema.json").write_text(
        json.dumps(problem.schema(), indent=2) + "\n", encoding="utf-8"
    )
    print(f"Controller manifest: {(args.output_dir / 'manifest.json').resolve()}")


if __name__ == "__main__":
    main()
