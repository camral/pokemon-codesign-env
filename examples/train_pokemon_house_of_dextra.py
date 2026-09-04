"""Search Pokemon teams with House of Dextra's frozen controller and value GNN."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from pokemon_codesign_env.baseline_artifacts import RunMetrics, write_run_artifacts
from pokemon_codesign_env.configs import FormatConfig
from pokemon_codesign_env.fasttd3_factory import ENVIRONMENTS
from pokemon_codesign_env.house_problem import make_problem
from pokemon_codesign_env.scaling import ShowdownClusterConfig, managed_showdown_cluster


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--controller-manifest", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=40)
    parser.add_argument("--candidates", type=int, default=16)
    parser.add_argument("--pool-multiplier", type=int, default=4)
    parser.add_argument("--epsilon", type=float, default=0.2)
    parser.add_argument("--gumbel-temperature", type=float, default=0.4)
    parser.add_argument("--gumbel-increment", type=float, default=0.0)
    parser.add_argument("--max-expansion-steps", type=int, default=256)
    parser.add_argument("--train-steps", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--eval-episodes", type=int, default=1)
    parser.add_argument("--eval-actors-per-design", type=int, default=1)
    parser.add_argument("--eval-max-steps", type=int, default=100000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--wandb-project")
    parser.add_argument("--launch-server", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument(
        "--baseline-root",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "codesign-baselines",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("runs/pokemon_house/search"))
    return parser.parse_args()


def _upstream_arguments(args: argparse.Namespace) -> list[str]:
    values = [
        "--controller-manifest",
        str(args.controller_manifest),
        "--iterations",
        str(args.iterations),
        "--candidates",
        str(args.candidates),
        "--pool-multiplier",
        str(args.pool_multiplier),
        "--epsilon",
        str(args.epsilon),
        "--gumbel-temperature",
        str(args.gumbel_temperature),
        "--gumbel-increment",
        str(args.gumbel_increment),
        "--max-expansion-steps",
        str(args.max_expansion_steps),
        "--train-steps",
        str(args.train_steps),
        "--learning-rate",
        str(args.learning_rate),
        "--eval-episodes",
        str(args.eval_episodes),
        "--eval-actors-per-design",
        str(args.eval_actors_per_design),
        "--eval-max-steps",
        str(args.eval_max_steps),
        "--seed",
        str(args.seed),
        "--device",
        args.device,
        "--output-dir",
        str(args.output_dir),
    ]
    if args.wandb_project:
        values.extend(("--wandb-project", args.wandb_project))
    return values


def main() -> None:
    args = parse_args()
    sys.path.insert(0, str(args.baseline_root.resolve()))
    from house_of_dextra.general_codesign import search
    from house_of_dextra.general_codesign.controller import load_manifest

    started = time.monotonic()
    manifest = load_manifest(args.controller_manifest)
    config = dict(manifest.get("problem_config", {}))
    problem = make_problem(config)
    environment = config.get("environment", "random")
    format_config = FormatConfig.from_showdown(ENVIRONMENTS[environment].BATTLE_FORMAT)
    cluster = ShowdownClusterConfig(
        config.get("showdown_host", "127.0.0.1"),
        int(config.get("showdown_port", 8000)),
        int(config.get("showdown_shards", 1)),
    )
    original_argv = sys.argv
    try:
        sys.argv = [str(Path(search.__file__).resolve()), *_upstream_arguments(args)]
        with managed_showdown_cluster(
            args.launch_server, format_config.showdown_path, cluster, args.verbose
        ):
            search.main()
    finally:
        sys.argv = original_argv

    result = json.loads((args.output_dir / "best_design.json").read_text(encoding="utf-8"))
    design = problem.deserialize(result["design"])
    per_design_battles = args.eval_episodes * args.eval_actors_per_design
    write_run_artifacts(
        args.output_dir,
        team_parameters=problem.serialize(design),
        team_design_schema=problem.vector_adapter.schema(),
        observation_schema=problem.encoder.get_schema(),
        metrics=RunMetrics(
            mean_return=float(result["score"]),
            evaluation_battles=per_design_battles,
            training_battles=int(result["evaluations"]) * per_design_battles,
            wall_time_seconds=time.monotonic() - started,
        ),
        metadata={
            "method": "house_of_dextra_search",
            "unique_designs_evaluated": int(result["evaluations"]),
            "controller_frozen_during_search": True,
        },
    )


if __name__ == "__main__":
    main()
