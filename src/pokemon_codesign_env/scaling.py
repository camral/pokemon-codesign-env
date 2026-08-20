import os
import signal
import shutil
import socket
import subprocess
import tempfile
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

from poke_env.ps_client import ServerConfiguration


@dataclass(frozen=True)
class ShowdownEndpoint:
    """Connection details for one Pokemon Showdown server shard."""

    host: str = "127.0.0.1"
    port: int = 8000

    @property
    def server_configuration(self) -> ServerConfiguration:
        return ServerConfiguration(f"ws://{self.host}:{self.port}/showdown/websocket", "https://play.pokemonshowdown.com/action.php?")


@dataclass(frozen=True)
class ShowdownClusterConfig:
    """A round-robin pool of independent Pokemon Showdown endpoints."""

    host: str = "127.0.0.1"
    base_port: int = 8000
    shards: int = 1

    def __post_init__(self) -> None:
        if self.shards < 1:
            raise ValueError("Showdown cluster must contain at least one shard")
        if not 1 <= self.base_port <= 65_535 or self.base_port + self.shards - 1 > 65_535:
            raise ValueError("Showdown shard ports must be between 1 and 65535")

    @property
    def endpoints(self) -> tuple[ShowdownEndpoint, ...]:
        return tuple(ShowdownEndpoint(self.host, self.base_port + shard) for shard in range(self.shards))

    def endpoint_for(self, environment_index: int) -> ShowdownEndpoint:
        return self.endpoints[environment_index % self.shards]


def is_showdown_running(endpoint: ShowdownEndpoint, timeout: float = 1.0) -> bool:
    """Return whether an endpoint completes a Pokemon Showdown WebSocket handshake."""
    try:
        with socket.create_connection((endpoint.host, endpoint.port), timeout=timeout) as connection:
            request = (
                f"GET /showdown/websocket HTTP/1.1\r\nHost: {endpoint.host}:{endpoint.port}\r\nUpgrade: websocket\r\n"
                "Connection: Upgrade\r\nSec-WebSocket-Key: dGhlIHNhbXBsZSBub25jZQ==\r\nSec-WebSocket-Version: 13\r\n\r\n"
            )
            connection.sendall(request.encode("ascii"))
            response = connection.recv(512)
            return response.startswith(b"HTTP/1.1 101") and b"upgrade: websocket" in response.lower()
    except OSError:
        return False


def _link_or_copy(source: str, destination: str) -> None:
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def _create_showdown_runtime(showdown_path: Path, port: int) -> tuple[tempfile.TemporaryDirectory, Path]:
    temporary_directory = tempfile.TemporaryDirectory(prefix=f"pokemon-showdown-{port}-", dir="/tmp")
    runtime_path = Path(temporary_directory.name)
    for source in showdown_path.iterdir():
        destination = runtime_path / source.name
        if source.name in {"config", "databases"}:
            shutil.copytree(source, destination)
        elif source.name == "dist":
            shutil.copytree(source, destination, copy_function=_link_or_copy)
        elif source.name == "logs":
            (destination / "repl").mkdir(parents=True)
        elif source.name == "pokemon-showdown":
            shutil.copy2(source, destination)
        elif source.name != ".git":
            destination.symlink_to(source.resolve(), target_is_directory=source.is_dir())
    return temporary_directory, runtime_path


@contextmanager
def managed_showdown_cluster(enabled: bool, showdown_path: Path, cluster: ShowdownClusterConfig, verbose: bool = False, startup_timeout: float = 120.0):
    """Launch missing local Showdown shards and stop only those launched here."""
    endpoints = cluster.endpoints
    missing = [endpoint for endpoint in endpoints if not is_showdown_running(endpoint)]
    if not enabled:
        if missing:
            addresses = ", ".join(f"{endpoint.host}:{endpoint.port}" for endpoint in missing)
            raise RuntimeError(f"Pokemon Showdown is not running at {addresses}. Start the shards or pass --launch-server.")
        yield endpoints
        return
    if cluster.host not in ("127.0.0.1", "localhost") and missing:
        raise ValueError("--launch-server can only launch local Showdown shards")
    if not missing:
        if verbose:
            print(f"showdown_cluster=status_existing shards={len(endpoints)} ports={cluster.base_port}-{cluster.base_port + cluster.shards - 1}")
        yield endpoints
        return

    showdown_path = showdown_path.resolve()
    executable = showdown_path / "pokemon-showdown"
    if not executable.is_file():
        raise FileNotFoundError(f"Pokemon Showdown launcher not found: {executable}")
    subprocess.run(
        ["node", "build"], cwd=showdown_path, check=True,
        stdout=None if verbose else subprocess.DEVNULL, stderr=None if verbose else subprocess.DEVNULL,
    )
    output = None if verbose else subprocess.DEVNULL
    processes = []
    runtimes = []
    for endpoint in missing:
        temporary_directory, runtime_path = _create_showdown_runtime(showdown_path, endpoint.port)
        process = subprocess.Popen(
            ["node", "--preserve-symlinks", str(runtime_path / "pokemon-showdown"), "start", "--skip-build", str(endpoint.port), "--no-security"],
            cwd=runtime_path,
            stdout=output, stderr=output, start_new_session=True,
        )
        processes.append((endpoint, process))
        runtimes.append(temporary_directory)
    deadline = time.monotonic() + startup_timeout

    try:
        while True:
            unavailable = [endpoint for endpoint in missing if not is_showdown_running(endpoint)]
            if not unavailable:
                break
            for endpoint, process in processes:
                return_code = process.poll()
                if return_code is not None:
                    raise RuntimeError(f"Pokemon Showdown shard {endpoint.host}:{endpoint.port} exited during startup with code {return_code}.")
            if time.monotonic() >= deadline:
                addresses = ", ".join(f"{endpoint.host}:{endpoint.port}" for endpoint in unavailable)
                raise TimeoutError(f"Pokemon Showdown shards did not start within {startup_timeout:g} seconds: {addresses}")
            time.sleep(0.1)
        if verbose:
            print(
                f"showdown_cluster=status_started launched_shards={len(processes)} total_shards={len(endpoints)} "
                f"ports={cluster.base_port}-{cluster.base_port + cluster.shards - 1}"
            )
        yield endpoints
    finally:
        for _, process in processes:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
        for _, process in processes:
            if process.poll() is None:
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
        for temporary_directory in runtimes:
            temporary_directory.cleanup()
        if verbose:
            print("showdown_cluster=status_stopped")
