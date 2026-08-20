import json
import random
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from poke_env.teambuilder import Teambuilder


@dataclass(frozen=True)
class RandomTeamPool:
    """A reusable collection of packed Showdown teams generated in one process."""

    teams: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.teams:
            raise ValueError("RandomTeamPool requires at least one team")

    def builder(self, seed: int | None = None) -> "PooledTeamBuilder":
        return PooledTeamBuilder(self.teams, seed)

    @classmethod
    def from_showdown(cls, showdown_path: Path, format_id: str, size: int) -> "RandomTeamPool":
        if size < 1:
            raise ValueError("Random team pool size must be at least one")
        showdown_path = Path(showdown_path).resolve()
        result = subprocess.run(
            [str(showdown_path / "pokemon-showdown"), "generate-teams", format_id, str(size)], cwd=showdown_path,
            capture_output=True, check=False, text=True,
        )
        if result.returncode:
            message = result.stderr.strip() or result.stdout.strip()
            raise RuntimeError(f"Pokemon Showdown could not generate the random team pool: {message}")
        teams = tuple(team for team in result.stdout.splitlines() if team)
        if len(teams) != size:
            raise RuntimeError(f"Pokemon Showdown generated {len(teams)} teams instead of {size}")
        return cls(teams)

    @classmethod
    def from_generator(
        cls, showdown_path: Path, format_id: str, size: int, generate_team: Callable[[], str], max_batches: int = 10
    ) -> "RandomTeamPool":
        """Generate varied teams in Python and validate each batch in one Node process."""
        if size < 1:
            raise ValueError("Team pool size must be at least one")
        valid_teams = []
        for _ in range(max_batches):
            candidates = [generate_team() for _ in range(size - len(valid_teams))]
            validation_errors = validate_team_batch(Path(showdown_path).resolve(), format_id, candidates)
            valid_teams.extend(team for team, errors in zip(candidates, validation_errors, strict=True) if errors is None)
            if len(valid_teams) == size:
                return cls(tuple(valid_teams))
        raise RuntimeError(f"Could only generate {len(valid_teams)} valid teams for a pool of {size}")


def validate_team_batch(showdown_path: Path, format_id: str, teams: list[str]) -> list[list[str] | None]:
    script = """
        const fs = require('fs');
        const { Teams } = require('./dist/sim/teams');
        const { TeamValidator } = require('./dist/sim/team-validator');
        const validator = TeamValidator.get(process.argv[1]);
        const teams = JSON.parse(fs.readFileSync(0, 'utf8'));
        console.log(JSON.stringify(teams.map(team => validator.validateTeam(Teams.unpack(team)) || null)));
    """
    result = subprocess.run(["node", "-e", script, format_id], cwd=showdown_path, input=json.dumps(teams), capture_output=True, check=False, text=True)
    if result.returncode:
        message = result.stderr.strip() or result.stdout.strip()
        raise RuntimeError(f"Pokemon Showdown could not validate the generated team pool: {message}")
    return json.loads(result.stdout)


class PooledTeamBuilder(Teambuilder):
    """Sample packed teams without launching a subprocess for every battle."""

    def __init__(self, teams: tuple[str, ...], seed: int | None = None):
        super().__init__()
        self.teams = teams
        self.rng = random.Random(seed)

    def yield_team(self) -> str:
        return self.rng.choice(self.teams)
