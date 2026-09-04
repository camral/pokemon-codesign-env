"""Constructors for single and vectorized codesign environments.

These mirror the names Gymnasium uses, but build `PokEnv` directly rather than going through
Gymnasium's registry, so the extended `reset(embodiment=)` and `step(embodiment=)` signatures are
not hidden behind `OrderEnforcing` or `PassiveEnvChecker`, and vectorization stays this project's
own `ShardedVecEnv` rather than `SyncVectorEnv` or `AsyncVectorEnv`.

Both constructors launch the Pokemon Showdown servers their environments need and shut them down
again when the environment is closed. There is no option to attach to a server someone else
started: ports are claimed from the operating system, so concurrent runs never collide.
"""

import socket
from contextlib import ExitStack
from math import ceil
from pathlib import Path

from stable_baselines3.common.vec_env import VecEnvWrapper, VecMonitor

from pokemon_codesign_env.configs import FormatConfig, ObservationConfig, TeamSearchConfig
from pokemon_codesign_env.gym_environment_factory import PokEnv
from pokemon_codesign_env.scaling import ShowdownClusterConfig, is_showdown_running, managed_showdown_cluster
from pokemon_codesign_env.sharded_vec_env import ShardedVecEnv

DEFAULT_ENVS_PER_SHARD = 32
DEFAULT_SHOWDOWN_HOST = "127.0.0.1"


def default_showdown_path() -> Path:
    """The bundled Pokemon Showdown checkout."""
    return Path(__file__).resolve().parents[2] / "pokemon-showdown"


def _claim_ports(count: int, host: str = DEFAULT_SHOWDOWN_HOST) -> list[int]:
    """Ask the operating system for `count` contiguous-free ports to launch shards on."""
    sockets = []
    try:
        for _ in range(count):
            candidate = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            candidate.bind((host, 0))
            sockets.append(candidate)
        return [candidate.getsockname()[1] for candidate in sockets]
    finally:
        # Released immediately so Showdown can bind them; nothing else is racing for them in a
        # normal run, and a fixed base_port is available when that guarantee matters.
        for candidate in sockets:
            candidate.close()


def _port_is_free(host: str, port: int) -> bool:
    """Whether a shard could bind this port right now."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        try:
            probe.bind((host, port))
        except OSError:
            return False
    return True


def _require_free_ports(cluster: ShowdownClusterConfig) -> None:
    """Refuse to start when anything already holds a shard's port.

    These environments own the servers they battle on, so silently attaching to a server this
    process did not start would hand it an unknown configuration and leave it running afterwards.
    """
    occupied = [endpoint for endpoint in cluster.endpoints if not _port_is_free(endpoint.host, endpoint.port)]
    if not occupied:
        return
    details = ", ".join(
        f"{endpoint.host}:{endpoint.port}"
        f"{' (a Pokemon Showdown server is already running there)' if is_showdown_running(endpoint) else ' (port in use)'}"
        for endpoint in occupied
    )
    raise RuntimeError(
        f"cannot launch {cluster.shards} Pokemon Showdown shard(s) from port {cluster.base_port}: {details}. "
        f"Stop whatever holds the port, or omit base_port to claim free ports automatically."
    )


def shard_count(num_envs: int, envs_per_shard: int = DEFAULT_ENVS_PER_SHARD) -> int:
    """How many Showdown servers `num_envs` environments need."""
    if num_envs < 1:
        raise ValueError(f"num_envs must be at least one, got {num_envs}")
    if envs_per_shard < 1:
        raise ValueError(f"envs_per_shard must be at least one, got {envs_per_shard}")
    return ceil(num_envs / envs_per_shard)


def _launch_cluster(
    stack: ExitStack, shards: int, host: str, base_port: int | None, showdown_path: Path | None, verbose: bool
) -> ShowdownClusterConfig:
    """Launch `shards` Showdown servers, registering their shutdown on `stack`."""
    if base_port is None:
        # One contiguous range is enough because ShowdownClusterConfig numbers shards from a base.
        base_port = min(_claim_ports(shards, host))
    cluster = ShowdownClusterConfig(host=host, base_port=base_port, shards=shards)
    # managed_showdown_cluster attaches to a shard that is already up, which these environments
    # must never do, so an occupied port is an error rather than a server to borrow.
    _require_free_ports(cluster)
    stack.enter_context(managed_showdown_cluster(True, showdown_path or default_showdown_path(), cluster, verbose=verbose))
    return cluster


def make(
    opponent: str = "random",
    format_config: FormatConfig | None = None,
    embodiment_config: TeamSearchConfig | None = None,
    observation_config: ObservationConfig | None = None,
    is_eval_env: bool = False,
    max_action_space_size: int | None = None,
    showdown_path: Path | None = None,
    host: str = DEFAULT_SHOWDOWN_HOST,
    base_port: int | None = None,
    verbose: bool = False,
    **env_kwargs,
) -> PokEnv:
    """Create one codesign environment on a Showdown server it owns.

    Closing the environment shuts the server down.

    :param is_eval_env: create an evaluation environment, which does not allow embodiment changes
        during evaluation.
    :param max_action_space_size: bound the action space. A format needing more raises; a smaller
        natural action space is padded up to it, with the padded actions masked as illegal.
    :param base_port: launch on this port instead of one claimed from the operating system.
    """
    if "server_configuration" in env_kwargs:
        raise ValueError("server_configuration cannot be set; make launches and owns its Showdown server")

    stack = ExitStack()
    try:
        cluster = _launch_cluster(stack, 1, host, base_port, showdown_path, verbose)
        env = PokEnv(
            opponent=opponent,
            format_config=format_config,
            embodiment_config=embodiment_config,
            observation_config=observation_config,
            is_eval_env=is_eval_env,
            max_action_space_size=max_action_space_size,
            showdown_path=showdown_path,
            server_configuration=cluster.endpoint_for(0).server_configuration,
            **env_kwargs,
        )
    except BaseException:
        stack.close()
        raise
    env.server_context = stack
    return env


class ServerOwningVecEnv(VecEnvWrapper):
    """A vector environment that shuts down the Showdown servers it was launched with."""

    def __init__(self, venv, server_context: ExitStack, cluster: ShowdownClusterConfig):
        super().__init__(venv)
        self.server_context = server_context
        self.cluster = cluster

    def reset(self):
        return self.venv.reset()

    def step_wait(self):
        return self.venv.step_wait()

    def close(self) -> None:
        try:
            self.venv.close()
        finally:
            if self.server_context is not None:
                server_context, self.server_context = self.server_context, None
                server_context.close()


def _env_factory(env_kwargs: dict):
    """Return a picklable constructor for one environment of a vector."""
    def initialize() -> PokEnv:
        return PokEnv(**env_kwargs)

    return initialize


def make_vec(
    opponent: str = "random",
    num_envs: int = 1,
    shards: int | None = None,
    envs_per_shard: int = DEFAULT_ENVS_PER_SHARD,
    envs_per_worker: int = 32,
    reset_retries: int = 3,
    step_timeout: float = 60.0,
    step_timeout_reward: float = 0.0,
    start_method: str = "spawn",
    monitor: bool = True,
    showdown_path: Path | None = None,
    host: str = DEFAULT_SHOWDOWN_HOST,
    base_port: int | None = None,
    verbose: bool = False,
    **env_kwargs,
):
    """Create `num_envs` codesign environments on a Showdown cluster they own.

    Closing the vector environment shuts every launched server down.

    :param num_envs: how many environments to run, packed `envs_per_worker` to a worker process.
    :param shards: how many Showdown servers to launch. Defaults to one per `envs_per_shard`
        environments; environment `rank` battles on shard `rank % shards`.
    :param base_port: launch the shards from this port instead of a range claimed from the
        operating system.
    """
    if "server_configuration" in env_kwargs:
        raise ValueError("server_configuration cannot be set; make_vec launches and owns its Showdown servers")
    if shards is None:
        shards = shard_count(num_envs, envs_per_shard)
    elif shards < 1:
        raise ValueError(f"shards must be at least one, got {shards}")
    if num_envs < 1:
        raise ValueError(f"num_envs must be at least one, got {num_envs}")

    stack = ExitStack()
    try:
        cluster = _launch_cluster(stack, shards, host, base_port, showdown_path, verbose)
        factories = [
            _env_factory({
                "opponent": opponent,
                "showdown_path": showdown_path,
                "server_configuration": cluster.endpoint_for(rank).server_configuration,
                **env_kwargs,
            })
            for rank in range(num_envs)
        ]
        vector_env = ShardedVecEnv(
            factories, envs_per_worker=envs_per_worker, reset_retries=reset_retries, start_method=start_method,
            step_timeout=step_timeout, step_timeout_reward=step_timeout_reward,
        )
        if monitor:
            vector_env = VecMonitor(vector_env)
    except BaseException:
        stack.close()
        raise
    return ServerOwningVecEnv(vector_env, stack, cluster)
