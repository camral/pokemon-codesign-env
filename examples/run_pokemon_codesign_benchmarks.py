"""Run configured Pokemon co-design trainers and summarize their common artifacts."""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path

REQUIRED_ARTIFACTS = (
    "best_team.json",
    "team_design_schema.json",
    "observation_schema.json",
    "metrics.json",
)


def collect_results(runs):
    results = []
    for run in runs:
        output_dir = Path(run["output_dir"])
        missing = [name for name in REQUIRED_ARTIFACTS if not (output_dir / name).is_file()]
        if missing:
            raise FileNotFoundError(f"{run['name']} is missing artifacts: {', '.join(missing)}")
        metrics = json.loads((output_dir / "metrics.json").read_text(encoding="utf-8"))
        results.append({"name": run["name"], "output_dir": str(output_dir), **metrics})
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("runs/benchmark_summary.json"))
    parser.add_argument("--collect-only", action="store_true")
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    runs = config.get("runs", [])
    if not runs:
        raise ValueError("Benchmark config must contain a non-empty runs list")
    started = time.monotonic()
    if not args.collect_only:
        for run in runs:
            command = run.get("command")
            if not isinstance(command, list) or not command:
                raise ValueError(f"Run {run.get('name')!r} must provide a command list")
            subprocess.run(command, check=True)
    results = collect_results(runs)
    payload = {
        "wall_time_seconds": time.monotonic() - started,
        "runs": results,
        "comparison_fields": [
            "mean_return",
            "win_rate",
            "mean_episode_length",
            "mean_turn_count",
            "invalid_candidate_rate",
            "environment_transitions",
            "training_battles",
            "evaluation_battles",
            "wall_time_seconds",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
