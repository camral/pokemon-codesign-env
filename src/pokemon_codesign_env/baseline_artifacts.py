"""Consistent local artifacts for Pokemon co-design baseline runs."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class RunMetrics:
    mean_return: float
    evaluation_battles: int
    training_battles: int = 0
    environment_transitions: int = 0
    invalid_candidate_rate: float = 0.0
    repair_rate: float = 0.0
    win_rate: float | None = None
    mean_episode_length: float | None = None
    mean_turn_count: float | None = None
    wall_time_seconds: float | None = None


def write_run_artifacts(
    output_dir: Path,
    *,
    team_parameters: dict,
    team_design_schema: dict,
    observation_schema: dict,
    metrics: RunMetrics,
    metadata: dict | None = None,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    payloads = {
        "best_team.json": team_parameters,
        "team_design_schema.json": team_design_schema,
        "observation_schema.json": observation_schema,
        "metrics.json": {**asdict(metrics), **dict(metadata or {})},
    }
    for filename, payload in payloads.items():
        (output_dir / filename).write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    (output_dir / "replays").mkdir(exist_ok=True)
