"""Showdown runtimes left by a killed owner are swept, and servers do not outlive their owner.

Real short-lived processes stand in for owners and servers, so no Showdown is needed.
"""

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

from pokemon_codesign_env import scaling
from pokemon_codesign_env.scaling import OWNER_FILE, RUNTIME_PREFIX, _guarded, _process_start, _write_owner, sweep_stale_runtimes

HAS_PROC = Path("/proc/self/stat").exists()


def _runtime(root: Path, name: str = "8000-abc") -> Path:
    path = root / f"{RUNTIME_PREFIX}{name}"
    path.mkdir()
    return path


def _dead_pid() -> int:
    process = subprocess.Popen([sys.executable, "-c", "pass"])
    process.wait()
    return process.pid


def _record(path: Path, **fields) -> None:
    (path / OWNER_FILE).write_text(json.dumps(fields))


def _group_gone(pgid: int) -> bool:
    try:
        os.killpg(pgid, 0)
    except (ProcessLookupError, PermissionError):  # PermissionError: only reaped/zombie members left (macOS)
        return True
    return False


def test_a_runtime_whose_owner_is_running_is_kept(tmp_path):
    path = _runtime(tmp_path)
    _write_owner(path, port=8000)
    assert sweep_stale_runtimes(str(tmp_path)) == []
    assert path.is_dir()


def test_a_runtime_whose_owner_died_is_deleted_and_its_server_stopped(tmp_path):
    path = _runtime(tmp_path)
    server = subprocess.Popen(["sleep", "60"], start_new_session=True)
    try:
        _record(path, owner_pid=_dead_pid(), owner_start=None, server_pgid=server.pid, server_start=_process_start(server.pid))
        assert sweep_stale_runtimes(str(tmp_path)) == [path]
        assert not path.exists()
        assert server.wait(timeout=10) == -signal.SIGTERM
    finally:
        if server.poll() is None:
            server.kill()


def test_only_showdown_runtimes_are_considered(tmp_path):
    other = tmp_path / "something-else"
    other.mkdir()
    _record(other, owner_pid=_dead_pid(), owner_start=None)
    assert sweep_stale_runtimes(str(tmp_path)) == []
    assert other.is_dir()


@pytest.mark.skipif(not HAS_PROC, reason="PID reuse is detected from /proc start times")
def test_a_reused_owner_pid_counts_as_dead(tmp_path):
    path = _runtime(tmp_path)
    _record(path, owner_pid=os.getpid(), owner_start="0")
    assert sweep_stale_runtimes(str(tmp_path)) == [path]


@pytest.mark.skipif(not HAS_PROC, reason="a legacy runtime is judged by which processes run inside it, from /proc")
def test_a_legacy_runtime_is_deleted_only_when_old_and_unused(tmp_path, monkeypatch):
    young, old, busy = _runtime(tmp_path, "8000-young"), _runtime(tmp_path, "8001-old"), _runtime(tmp_path, "8002-busy")
    past = time.time() - 2 * scaling.LEGACY_GRACE_SECONDS
    for path in (old, busy):
        os.utime(path, (past, past))
    inside = subprocess.Popen(["sleep", "60"], cwd=busy)
    try:
        assert sweep_stale_runtimes(str(tmp_path)) == [old]
        assert young.is_dir() and busy.is_dir()
    finally:
        inside.kill()
        inside.wait()


def test_the_guard_stops_the_server_group_when_the_owner_dies(monkeypatch):
    monkeypatch.setattr(scaling, "GUARD_POLL_SECONDS", 0.2)
    owner = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    server = subprocess.Popen(_guarded(["sleep", "60"], owner.pid), start_new_session=True)
    try:
        time.sleep(0.5)
        assert server.poll() is None
        owner.kill()
        owner.wait()
        server.wait(timeout=10)            # the guard shell ends with its group
        time.sleep(0.2)
        assert _group_gone(server.pid), "the server outlived its owner"
    finally:
        for process in (owner, server):
            if process.poll() is None:
                process.kill()


def test_the_guard_passes_the_server_exit_code_through():
    server = subprocess.Popen(_guarded(["sh", "-c", "exit 3"], os.getpid()), start_new_session=True)
    assert server.wait(timeout=10) == 3
