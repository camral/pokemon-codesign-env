import atexit
import json
import os
import random
import subprocess
from collections.abc import Callable
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from threading import Lock

import numpy as np
from poke_env.teambuilder import Teambuilder

from pokemon_codesign_env.showdown import get_format_rules
from pokemon_codesign_env.team_levels import LEVEL_MEAN, LEVEL_STD, resample_packed_team_levels


@dataclass(frozen=True)
class RandomTeamPool:
    """A reusable collection of packed Showdown teams generated in one process."""

    teams: tuple[str, ...]
    level_mean: float = LEVEL_MEAN
    level_std: float = LEVEL_STD
    min_level: int = 1
    max_level: int = 100

    def __post_init__(self) -> None:
        if not self.teams:
            raise ValueError("RandomTeamPool requires at least one team")

    def builder(self, seed: int | None = None) -> "PooledTeamBuilder":
        return PooledTeamBuilder(
            self.teams,
            seed,
            self.level_mean,
            self.level_std,
            self.min_level,
            self.max_level,
        )

    @classmethod
    def from_showdown(cls, showdown_path: Path, format_id: str, size: int) -> "RandomTeamPool":
        if size < 1:
            raise ValueError("Random team pool size must be at least one")
        showdown_path = Path(showdown_path).resolve()
        result = subprocess.run(
            [str(showdown_path / "pokemon-showdown"), "generate-teams", format_id, str(size)],
            cwd=showdown_path,
            capture_output=True,
            check=False,
            text=True,
        )
        if result.returncode:
            message = result.stderr.strip() or result.stdout.strip()
            raise RuntimeError(
                f"Pokemon Showdown could not generate the random team pool: {message}"
            )
        teams = tuple(team for team in result.stdout.splitlines() if team)
        if len(teams) != size:
            raise RuntimeError(f"Pokemon Showdown generated {len(teams)} teams instead of {size}")
        rules = get_format_rules(format_id, showdown_path)
        return cls(
            teams,
            min_level=int(rules["min_level"]),
            max_level=int(rules["max_level"]),
        )

    @classmethod
    def from_generator(
        cls,
        showdown_path: Path,
        format_id: str,
        size: int,
        generate_team: Callable[[], str],
        max_batches: int = 10,
    ) -> "RandomTeamPool":
        """Generate varied teams in Python and validate each batch in one Node process."""
        if size < 1:
            raise ValueError("Team pool size must be at least one")
        valid_teams = []
        for _ in range(max_batches):
            candidates = [generate_team() for _ in range(size - len(valid_teams))]
            validation_errors = validate_team_batch(
                Path(showdown_path).resolve(), format_id, candidates
            )
            valid_teams.extend(
                team
                for team, errors in zip(candidates, validation_errors, strict=True)
                if errors is None
            )
            if len(valid_teams) == size:
                rules = get_format_rules(format_id, Path(showdown_path).resolve())
                return cls(
                    tuple(valid_teams),
                    min_level=int(rules["min_level"]),
                    max_level=int(rules["max_level"]),
                )
        raise RuntimeError(
            f"Could only generate {len(valid_teams)} valid teams for a pool of {size}"
        )


class TeamBatchValidator:
    """Keep one Showdown validator process alive for repeated team batches."""

    SCRIPT = """
        const readline = require('readline');
        const { Teams } = require('./dist/sim/teams');
        const { TeamValidator } = require('./dist/sim/team-validator');
        const validator = TeamValidator.get(process.argv[1]);
        const input = readline.createInterface({ input: process.stdin, crlfDelay: Infinity });
        input.on('line', line => {
            try {
                const teams = JSON.parse(line);
                const results = teams.map(
                    team => validator.validateTeam(Teams.unpack(team)) || null
                );
                process.stdout.write(JSON.stringify(results) + '\\n');
            } catch (error) {
                const result = { error: error.stack || String(error) };
                process.stdout.write(JSON.stringify(result) + '\\n');
            }
        });
    """

    def __init__(self, showdown_path: Path, format_id: str):
        self.showdown_path = Path(showdown_path).resolve()
        self.format_id = format_id
        self.lock = Lock()
        self.process = None
        self.owner_pid = None

    def _start(self) -> None:
        self.process = subprocess.Popen(
            ["node", "-e", self.SCRIPT, self.format_id],
            cwd=self.showdown_path,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        self.owner_pid = os.getpid()

    def _discard_inherited_process(self) -> None:
        if self.process is not None:
            for stream in (self.process.stdin, self.process.stdout, self.process.stderr):
                if stream is not None:
                    stream.close()
        self.process = None
        self.owner_pid = None

    def _ensure_started(self) -> None:
        if self.owner_pid is not None and self.owner_pid != os.getpid():
            self._discard_inherited_process()
        if self.process is None or self.process.poll() is not None:
            self.close()
            self._start()

    def validate(self, teams: list[str]) -> list[list[str] | None]:
        with self.lock:
            for attempt in range(2):
                self._ensure_started()
                try:
                    self.process.stdin.write(json.dumps(teams) + "\n")
                    self.process.stdin.flush()
                    response = self.process.stdout.readline()
                    if not response:
                        message = self.process.stderr.read().strip()
                        raise RuntimeError(message or "validator process exited without a response")
                    result = json.loads(response)
                    if isinstance(result, dict) and "error" in result:
                        raise RuntimeError(result["error"])
                    if not isinstance(result, list) or len(result) != len(teams):
                        result_count = (
                            len(result) if isinstance(result, list) else type(result).__name__
                        )
                        raise RuntimeError(
                            f"validator returned {result_count} results for {len(teams)} teams"
                        )
                    return result
                except (BrokenPipeError, json.JSONDecodeError, OSError, RuntimeError) as error:
                    self.close()
                    if attempt:
                        raise RuntimeError(
                            f"Pokemon Showdown could not validate the generated team pool: {error}"
                        ) from error
        raise RuntimeError("Pokemon Showdown could not validate the generated team pool")

    def close(self) -> None:
        if self.process is None:
            return
        process = self.process
        self.process = None
        if self.owner_pid != os.getpid():
            for stream in (process.stdin, process.stdout, process.stderr):
                if stream is not None:
                    stream.close()
            self.owner_pid = None
            return
        if process.stdin is not None:
            with suppress(BrokenPipeError):
                process.stdin.close()
        try:
            process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            process.terminate()
            process.wait(timeout=2)
        self.owner_pid = None


_TEAM_BATCH_VALIDATORS = {}
_TEAM_BATCH_VALIDATORS_LOCK = Lock()


def _get_team_batch_validator(showdown_path: Path, format_id: str) -> TeamBatchValidator:
    key = (Path(showdown_path).resolve(), format_id)
    with _TEAM_BATCH_VALIDATORS_LOCK:
        if key not in _TEAM_BATCH_VALIDATORS:
            _TEAM_BATCH_VALIDATORS[key] = TeamBatchValidator(*key)
        return _TEAM_BATCH_VALIDATORS[key]


def _close_team_batch_validators() -> None:
    for validator in _TEAM_BATCH_VALIDATORS.values():
        validator.close()


atexit.register(_close_team_batch_validators)


def validate_team_batch(
    showdown_path: Path, format_id: str, teams: list[str]
) -> list[list[str] | None]:
    return _get_team_batch_validator(showdown_path, format_id).validate(teams)


class PooledTeamBuilder(Teambuilder):
    """Sample packed teams without launching a subprocess for every battle."""

    def __init__(
        self,
        teams: tuple[str, ...],
        seed: int | None = None,
        level_mean: float = LEVEL_MEAN,
        level_std: float = LEVEL_STD,
        min_level: int = 1,
        max_level: int = 100,
    ):
        super().__init__()
        self.teams = teams
        self.rng = random.Random(seed)
        self.level_rng = np.random.default_rng(seed)
        self.level_mean = level_mean
        self.level_std = level_std
        self.min_level = min_level
        self.max_level = max_level

    def yield_team(self) -> str:
        return resample_packed_team_levels(
            self.rng.choice(self.teams),
            self.level_rng,
            minimum=self.min_level,
            maximum=self.max_level,
            mean=self.level_mean,
            std=self.level_std,
        )
