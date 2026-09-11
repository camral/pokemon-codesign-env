import os
import signal
import shutil
import socket
import subprocess
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from pathlib import Path

from poke_env.ps_client import ServerConfiguration


RUNTIME_DIR_ENV = "POKEMON_SHOWDOWN_RUNTIME_DIR"


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


RUNTIME_WORKERS = 16


def _same_filesystem(left, right) -> bool:
    """Whether two paths live on one filesystem, which is what decides if os.link can work."""
    return os.stat(left).st_dev == os.stat(right).st_dev


def _runtime_base_dir() -> str:
    """Return the configured scratch directory used to build shard runtimes."""
    configured = os.environ.get(RUNTIME_DIR_ENV)
    if configured:
        runtime_dir = Path(configured).expanduser()
        runtime_dir.mkdir(parents=True, exist_ok=True)
        return str(runtime_dir.resolve())
    return "/tmp" if os.path.isdir("/tmp") else tempfile.gettempdir()


@contextmanager
def _shared_dist(showdown_path: Path, base_dir: str):
    """Yield a dist tree the shard runtimes can hardlink from, copying it once if they cannot.

    A runtime hardlinks the 106MB dist tree rather than copying it, which only works within one
    filesystem. A cluster node usually keeps the checkout on shared storage and the scratch
    directory on a local disk, so os.link fails with EXDEV and _link_or_copy quietly falls back to
    copying: 128 shards then move 13GB off shared storage before the first server starts, slowly
    enough to look like the servers are failing to boot. Copying dist once into the scratch
    directory restores the hardlinks for every shard and leaves the servers reading from the local
    disk rather than over the network.
    """
    if _same_filesystem(showdown_path, base_dir):
        yield showdown_path / "dist"
        return
    with tempfile.TemporaryDirectory(prefix="pokemon-showdown-dist-", dir=base_dir) as seed:
        destination = Path(seed) / "dist"
        shutil.copytree(showdown_path / "dist", destination)
        yield destination


def _default_startup_timeout(shards: int) -> float:
    """Startup budget for a cluster, which grows with the number of servers booting together.

    Each shard loads the full Showdown dex, so a large cluster is contending for the machine and
    a fixed budget that suits one server expires long before 128 of them are serving.
    """
    return max(120.0, 60.0 + 4.0 * shards)


def _create_showdown_runtime(
    showdown_path: Path, port: int, base_dir: str | None = None, dist_source: Path | None = None
) -> tuple[tempfile.TemporaryDirectory, Path]:
    temporary_directory = tempfile.TemporaryDirectory(prefix=f"pokemon-showdown-{port}-", dir=base_dir or _runtime_base_dir())
    runtime_path = Path(temporary_directory.name)
    for source in showdown_path.iterdir():
        destination = runtime_path / source.name
        if source.name in {"config", "databases"}:
            shutil.copytree(source, destination)
        elif source.name == "dist":
            shutil.copytree(dist_source or source, destination, copy_function=_link_or_copy)
        elif source.name == "logs":
            (destination / "repl").mkdir(parents=True)
        elif source.name == "pokemon-showdown":
            shutil.copy2(source, destination)
        elif source.name != ".git":
            destination.symlink_to(source.resolve(), target_is_directory=source.is_dir())
    return temporary_directory, runtime_path


def _describe_stuck_shard(endpoint: ShowdownEndpoint, log_path: Path | None) -> str:
    """Explain a shard that is running but not answering, using its own startup output."""
    detail = ""
    try:
        if log_path is not None:
            lines = [line for line in log_path.read_text().splitlines() if line.strip()]
            detail = f" last output: {lines[-1]}" if lines else " no output"
    except OSError:
        detail = ""
    listening = "reported itself listening but is not answering" if "listening" in detail else "never reported listening"
    return f"  {endpoint.host}:{endpoint.port} {listening}.{detail}"


@contextmanager
def managed_showdown_cluster(enabled: bool, showdown_path: Path, cluster: ShowdownClusterConfig, verbose: bool = False, startup_timeout: float | None = None):
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
    logs: dict[ShowdownEndpoint, Path] = {}
    # Building one runtime is filesystem work rather than computation, so the shards are prepared
    # together. Serially they cost about a third of a second each, which is a minute of dead time
    # on a 128 shard cluster before the first server is even asked to start.
    base_dir = _runtime_base_dir()
    started = time.monotonic()
    with ExitStack() as preparation:
        dist_source = preparation.enter_context(_shared_dist(showdown_path, base_dir))
        shared = dist_source != showdown_path / "dist"
        with ThreadPoolExecutor(max_workers=min(RUNTIME_WORKERS, len(missing))) as pool:
            prepared = list(pool.map(
                lambda endpoint: _create_showdown_runtime(showdown_path, endpoint.port, base_dir, dist_source), missing
            ))
        if verbose:
            print(
                f"showdown_cluster=status_prepared shards={len(missing)} seconds={time.monotonic() - started:.1f} "
                f"runtime_dir={base_dir} shared_dist={shared}"
            )
        for endpoint, (temporary_directory, runtime_path) in zip(missing, prepared):
            # A shard that cannot take its port keeps running rather than exiting, so the only
            # account of what went wrong is its own output. Kept on disk when it is not being
            # printed, so a startup failure can say why instead of just naming the address.
            log_path = runtime_path / "startup.log"
            destination = output if verbose else open(log_path, "w")
            process = subprocess.Popen(
                ["node", "--preserve-symlinks", str(runtime_path / "pokemon-showdown"), "start", "--skip-build", str(endpoint.port), "--no-security"],
                cwd=runtime_path,
                stdout=destination, stderr=subprocess.STDOUT if not verbose else destination, start_new_session=True,
            )
            if not verbose:
                destination.close()
            logs[endpoint] = log_path
            processes.append((endpoint, process))
            runtimes.append(temporary_directory)
    if startup_timeout is None:
        startup_timeout = _default_startup_timeout(len(missing))
    started = time.monotonic()
    deadline = started + startup_timeout
    reported = started

    try:
        # Probing the shards one after another would take as long as a whole sweep to notice the
        # deadline, since a shard that has bound its port but is not serving yet holds the probe
        # open for its full timeout. On a large cluster that is longer than the budget itself.
        with ThreadPoolExecutor(max_workers=min(RUNTIME_WORKERS, len(missing))) as pool:
            while True:
                ready = list(pool.map(is_showdown_running, missing))
                unavailable = [endpoint for endpoint, up in zip(missing, ready) if not up]
                if not unavailable:
                    break
                for endpoint, process in processes:
                    return_code = process.poll()
                    if return_code is not None:
                        raise RuntimeError(f"Pokemon Showdown shard {endpoint.host}:{endpoint.port} exited during startup with code {return_code}.")
                now = time.monotonic()
                if now >= deadline:
                    raise TimeoutError(
                        f"Pokemon Showdown shards did not start within {startup_timeout:g} seconds "
                        f"({len(missing) - len(unavailable)}/{len(missing)} started):\n"
                        + "\n".join(_describe_stuck_shard(endpoint, logs.get(endpoint)) for endpoint in unavailable)
                    )
                if verbose and now - reported >= 10.0:
                    reported = now
                    print(f"showdown_cluster=status_starting started={len(missing) - len(unavailable)}/{len(missing)} seconds={now - started:.0f}")
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
