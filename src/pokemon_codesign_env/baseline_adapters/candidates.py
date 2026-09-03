"""Shared compilation and validation of optimizer-produced team designs."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from pokemon_codesign_env.configs import TeamSearchConfig
from pokemon_codesign_env.team_designer import BatchedTeamDesigner
from pokemon_codesign_env.team_pool import validate_team_batch

from .team_design import TeamDesignVectorAdapter


@dataclass(frozen=True, slots=True)
class CompiledCandidate:
    """One decoded candidate and the result of Showdown validation."""

    candidate_id: str
    parameters: dict[str, dict]
    packed_team: str
    validation_errors: tuple[str, ...]

    @property
    def valid(self) -> bool:
        return not self.validation_errors


def compile_candidates(
    adapter: TeamDesignVectorAdapter,
    search_config: TeamSearchConfig,
    parameter_values: list[dict[str, dict]],
    seeds: list[int],
    *,
    validator: Callable[[Path, str, list[str]], list[list[str] | None]] = validate_team_batch,
) -> list[CompiledCandidate]:
    """Compile a population through TeamDesigner, then validate it as one batch."""

    if len(parameter_values) != len(seeds):
        raise ValueError("parameter_values and seeds must have the same length")
    if not parameter_values:
        return []

    teams = BatchedTeamDesigner.serialize_parameter_values(search_config, parameter_values, seeds)
    errors = validator(
        search_config.format.showdown_path,
        search_config.format.format_id,
        teams,
    )
    if len(errors) != len(teams):
        raise RuntimeError("The team validator returned the wrong number of results")

    return [
        CompiledCandidate(
            candidate_id=adapter.candidate_id(parameters),
            parameters=parameters,
            packed_team=team,
            validation_errors=tuple(candidate_errors or ()),
        )
        for parameters, team, candidate_errors in zip(parameter_values, teams, errors, strict=True)
    ]
