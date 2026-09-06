import atexit
import json
import os
import random
import subprocess
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
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


class _PersistentShowdownProcess:
    """Keep one Showdown worker process alive and talk to it in JSON lines.

    Spawning Node costs a few hundred milliseconds, which is most of what an episode boundary
    spends, so the processes that generate and validate teams are started once and reused. The
    owning pid is tracked because a worker process inherits this object across a fork and must not
    speak to a pipe another process owns.
    """

    SCRIPT: str
    FAILURE: str

    def __init__(self, showdown_path: Path, format_id: str):
        self.showdown_path = Path(showdown_path).resolve()
        self.format_id = format_id
        self.lock = Lock()
        self.process = None
        self.owner_pid = None

    def _start(self) -> None:
        self.process = subprocess.Popen(
            ["node", "-e", self.SCRIPT, self.format_id], cwd=self.showdown_path, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, text=True, bufsize=1,
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

    def _check(self, result, request):
        """Hook for a subclass to reject a structurally wrong reply."""
        return result

    def request(self, payload):
        """Send one request and return its parsed reply, restarting a dead worker once."""
        with self.lock:
            for attempt in range(2):
                self._ensure_started()
                try:
                    self.process.stdin.write(json.dumps(payload) + "\n")
                    self.process.stdin.flush()
                    response = self.process.stdout.readline()
                    if not response:
                        message = self.process.stderr.read().strip()
                        raise RuntimeError(message or "worker process exited without a response")
                    result = json.loads(response)
                    if isinstance(result, dict) and "error" in result:
                        raise RuntimeError(result["error"])
                    return self._check(result, payload)
                except (BrokenPipeError, json.JSONDecodeError, OSError, RuntimeError) as error:
                    self.close()
                    if attempt:
                        raise RuntimeError(f"{self.FAILURE}: {error}") from error
        raise RuntimeError(self.FAILURE)

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
            try:
                process.stdin.close()
            except BrokenPipeError:
                pass
        try:
            process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            process.terminate()
            process.wait(timeout=2)
        self.owner_pid = None


class TeamBatchGenerator(_PersistentShowdownProcess):
    """Generate seeded random teams in batches without a process per team.

    Showdown's own generate-teams command produces a batch but ignores a seed, which would give up
    the reproducible opponents the environment depends on. Passing one seed per team reproduces
    exactly what generate-team does one battle at a time.
    """

    FAILURE = "Pokemon Showdown could not generate a team batch"

    SCRIPT = """
        const readline = require('readline');
        const { Teams } = require('./dist/sim/teams');
        const format = process.argv[1];
        const input = readline.createInterface({ input: process.stdin, crlfDelay: Infinity });
        input.on('line', line => {
            try {
                const seeds = JSON.parse(line);
                process.stdout.write(JSON.stringify(seeds.map(seed => Teams.pack(Teams.generate(format, { seed })))) + '\\n');
            } catch (error) {
                process.stdout.write(JSON.stringify({ error: error.stack || String(error) }) + '\\n');
            }
        });
    """

    def _check(self, result, request):
        if not isinstance(result, list) or len(result) != len(request):
            count = len(result) if isinstance(result, list) else type(result).__name__
            raise RuntimeError(f"generator returned {count} teams for {len(request)} seeds")
        return result

    def generate(self, seeds: list[str]) -> list[str]:
        return self.request(seeds)


class TeamBatchValidator(_PersistentShowdownProcess):
    """Keep one Showdown validator process alive for repeated team batches."""

    FAILURE = "Pokemon Showdown could not validate the generated team pool"

    SCRIPT = """
        const readline = require('readline');
        const { Teams } = require('./dist/sim/teams');
        const { TeamValidator } = require('./dist/sim/team-validator');
        const validator = TeamValidator.get(process.argv[1]);
        const input = readline.createInterface({ input: process.stdin, crlfDelay: Infinity });
        input.on('line', line => {
            try {
                const teams = JSON.parse(line);
                process.stdout.write(JSON.stringify(teams.map(team => validator.validateTeam(Teams.unpack(team)) || null)) + '\\n');
            } catch (error) {
                process.stdout.write(JSON.stringify({ error: error.stack || String(error) }) + '\\n');
            }
        });
    """

    def _check(self, result, request):
        if not isinstance(result, list) or len(result) != len(request):
            count = len(result) if isinstance(result, list) else type(result).__name__
            raise RuntimeError(f"validator returned {count} results for {len(request)} teams")
        return result

    def validate(self, teams: list[str]) -> list[list[str] | None]:
        return self.request(teams)


_SHOWDOWN_WORKERS = {}
_SHOWDOWN_WORKERS_LOCK = Lock()


def _get_showdown_worker(worker_class, showdown_path: Path, format_id: str):
    key = (worker_class, Path(showdown_path).resolve(), format_id)
    with _SHOWDOWN_WORKERS_LOCK:
        if key not in _SHOWDOWN_WORKERS:
            _SHOWDOWN_WORKERS[key] = worker_class(*key[1:])
        return _SHOWDOWN_WORKERS[key]


def _close_showdown_workers() -> None:
    for worker in _SHOWDOWN_WORKERS.values():
        worker.close()


atexit.register(_close_showdown_workers)


def validate_team_batch(showdown_path: Path, format_id: str, teams: list[str]) -> list[list[str] | None]:
    return _get_showdown_worker(TeamBatchValidator, showdown_path, format_id).validate(teams)


def generate_team_batch(showdown_path: Path, format_id: str, seeds: list[str]) -> list[str]:
    """Generate one packed team per seed, reusing a single Showdown process."""
    return _get_showdown_worker(TeamBatchGenerator, showdown_path, format_id).generate(seeds)


class PooledTeamBuilder(Teambuilder):
    """Sample packed teams without launching a subprocess for every battle."""

    def __init__(self, teams: tuple[str, ...], seed: int | None = None):
        super().__init__()
        self.teams = teams
        self.rng = random.Random(seed)

    def yield_team(self) -> str:
        return self.rng.choice(self.teams)
