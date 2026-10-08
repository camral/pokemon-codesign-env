"""Constructors for single and vectorized codesign environments.

These mirror the names Gymnasium uses, but build `PokEnv` directly rather than going through
Gymnasium's registry, so the extended `reset(embodiment=)` and `step(embodiment=)` signatures are
not hidden behind `OrderEnforcing` or `PassiveEnvChecker`, and vectorization stays this project's
own `ShardedVecEnv` rather than `SyncVectorEnv` or `AsyncVectorEnv`.

Both constructors launch the Pokemon Showdown servers their environments need and shut them down
again when the environment is closed. There is no option to attach to a server someone else
started: ports are claimed from the operating system, so concurrent runs never collide.
"""

import random
import socket
import time
from contextlib import ExitStack
from math import ceil
from pathlib import Path

import numpy as np
from gymnasium.vector import VectorWrapper

from pokemon_codesign_env.configs import FormatConfig, ObservationConfig, TeamSearchConfig
from pokemon_codesign_env.gym_environment_factory import PokEnv
from pokemon_codesign_env.scaling import ShowdownClusterConfig, is_showdown_running, managed_showdown_cluster
from pokemon_codesign_env.sharded_vec_env import ShardedVecEnv

DEFAULT_ENVS_PER_SHARD = 32
DEFAULT_SHOWDOWN_HOST = "127.0.0.1"


def default_showdown_path() -> Path:
    """The bundled Pokemon Showdown checkout."""
    return Path(__file__).resolve().parents[2] / "pokemon-showdown"


# Shard ports are searched below the kernel's ephemeral range but above the registered services.
SHARD_PORT_FLOOR = 10_000
PORT_BLOCK_ATTEMPTS = 200


def _ephemeral_port_range() -> tuple[int, int]:
    """The port range the kernel hands out for outgoing connections."""
    try:
        with open("/proc/sys/net/ipv4/ip_local_port_range") as source:
            low, high = (int(value) for value in source.read().split())
            return low, high
    except (OSError, ValueError):
        return 32_768, 60_999


def _claim_ports(count: int, host: str = DEFAULT_SHOWDOWN_HOST) -> list[int]:
    """Ask the operating system for `count` free ports to launch shards on."""
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


def _claim_port_block(count: int, host: str = DEFAULT_SHOWDOWN_HOST) -> int:
    """Return the base of `count` contiguous free ports the kernel will not reassign.

    Binding port 0 always answers from the ephemeral range, which is the same range the kernel
    draws on for every outgoing connection. A cluster whose shards live there is racing every
    other socket on the machine, including the thousands this run is about to open and anything
    another user is doing, and a shard that loses the race binds nothing while its process stays
    up: the cluster then waits out the whole startup budget for a server that will never answer.
    Searching below that range leaves only deliberate services to collide with, which
    _require_free_ports reports immediately. The start is random so concurrent runs rarely meet.
    """
    ephemeral_low, _ephemeral_high = _ephemeral_port_range()
    ceiling = ephemeral_low - count
    if ceiling <= SHARD_PORT_FLOOR:
        return min(_claim_ports(count, host))
    for _ in range(PORT_BLOCK_ATTEMPTS):
        base = random.randrange(SHARD_PORT_FLOOR, ceiling)
        if all(_port_is_free(host, base + offset) for offset in range(count)):
            return base
    raise RuntimeError(
        f"could not find {count} contiguous free ports between {SHARD_PORT_FLOOR} and {ephemeral_low}. "
        f"Pass base_port to choose the range yourself, or run fewer shards."
    )


def _port_is_free(host: str, port: int) -> bool:
    """Whether a shard could bind this port right now.

    Showdown binds the wildcard address rather than one interface, so a port is only usable if
    that bind would succeed. Checking the loopback address alone lets a port through that the
    shard cannot actually take, and the shard then reports itself as listening while every client
    connection reaches whatever already held the address.
    """
    for address in {"", host}:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
            try:
                probe.bind((address, port))
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
    stack: ExitStack, shards: int, host: str, base_port: int | None, showdown_path: Path | None, verbose: bool,
    startup_timeout: float | None = None,
) -> ShowdownClusterConfig:
    """Launch `shards` Showdown servers, registering their shutdown on `stack`."""
    if base_port is None:
        # One contiguous range is enough because ShowdownClusterConfig numbers shards from a base.
        base_port = _claim_port_block(shards, host)
    cluster = ShowdownClusterConfig(host=host, base_port=base_port, shards=shards)
    # managed_showdown_cluster attaches to a shard that is already up, which these environments
    # must never do, so an occupied port is an error rather than a server to borrow.
    _require_free_ports(cluster)
    stack.enter_context(managed_showdown_cluster(
        True, showdown_path or default_showdown_path(), cluster, verbose=verbose, startup_timeout=startup_timeout,
    ))
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
) -> tuple[PokEnv, dict]:
    """Create one codesign environment on a Showdown server it owns, and its base design.

    Closing the environment shuts the server down.

    :returns: the environment and its `base_embodiment`, the design it starts from and the one a
        reset falls back to when given none. Keeping it lets a caller return to a known state.

    :param is_eval_env: create an evaluation environment, which does not allow embodiment changes
        during evaluation.
    :param max_action_space_size: bound the action space. Part of the codesign API for
        environments whose action space depends on the embodiment, which Pokemon's does not: a
        format needing more actions raises, and one needing fewer is padded up to the bound with
        the padded actions masked illegal, so a policy can span formats of different generations.
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
    return env, env.base_embodiment


class SameStepEpisodeStatistics(VectorWrapper):
    """Episode return, length and wall time in ``info["episode"]`` (``"r"``, ``"l"``, ``"t"``) on the step an episode
    ends, with the ``info["_episode"]`` mask, as Gymnasium's ``RecordEpisodeStatistics`` writes them.

    That wrapper starts counting a finished environment's new episode on the step after it ended, which suits
    NEXT_STEP autoreset. These environments reset within the step that ended the episode (SAME_STEP), so the next
    step is already part of the new episode: here the count restarts right after the ending step is recorded.
    """

    def __init__(self, env, stats_key: str = "episode"):
        super().__init__(env)
        self._stats_key = stats_key
        self._zero()

    def _zero(self) -> None:
        self.episode_returns = np.zeros(self.num_envs)
        self.episode_lengths = np.zeros(self.num_envs, dtype=int)
        self.episode_start_times = np.full(self.num_envs, time.perf_counter())

    def reset(self, *, seed=None, options=None):
        result = self.env.reset(seed=seed, options=options)
        self._zero()
        return result

    def step(self, actions):
        observations, rewards, terminations, truncations, infos = self.env.step(actions)
        self.episode_returns += np.asarray(rewards, dtype=float)
        self.episode_lengths += 1
        dones = np.logical_or(terminations, truncations)
        if dones.any():
            if self._stats_key in infos or f"_{self._stats_key}" in infos:
                raise ValueError(f"info already has the key {self._stats_key!r}: {list(infos)}")
            now = time.perf_counter()
            infos[self._stats_key] = {"r": np.where(dones, self.episode_returns, 0.0), "l": np.where(dones, self.episode_lengths, 0),
                                      "t": np.where(dones, np.round(now - self.episode_start_times, 6), 0.0)}
            infos[f"_{self._stats_key}"] = dones
            self.episode_returns[dones] = 0.0
            self.episode_lengths[dones] = 0
            self.episode_start_times[dones] = now
        return observations, rewards, terminations, truncations, infos


class ServerOwningVecEnv(VectorWrapper):
    """A vector environment that shuts down the Showdown servers it was launched with."""

    def __init__(self, env, server_context: ExitStack, cluster: ShowdownClusterConfig):
        super().__init__(env)
        self.server_context = server_context
        self.cluster = cluster
        self.embodiment_rejected = np.zeros(self.num_envs, dtype=bool)

    def _embodiments_for_each_environment(self, embodiment) -> list:
        """Spread one embodiment argument over the environments.

        A `(num_envs, embodiment_dim)` array of `flat_space` points gives each environment its own
        design, and so does a sequence of parameter dictionaries. A single dictionary or a single
        flat point is given to every environment.
        """
        if isinstance(embodiment, dict):
            return [embodiment] * self.num_envs
        if isinstance(embodiment, np.ndarray) and embodiment.ndim == 1:
            return [embodiment] * self.num_envs
        embodiments = list(embodiment)
        if len(embodiments) != self.num_envs:
            raise ValueError(f"Expected one embodiment per environment, got {len(embodiments)} for {self.num_envs}")
        return embodiments

    def set_embodiment(self, embodiment) -> None:
        """Install the embodiments immediately, raising if one cannot be designed with.

        `reset` is the forgiving path: it reports a rejected design instead of raising. This is
        the strict one, for a caller that wants to know at once.
        """
        for index, values in enumerate(self._embodiments_for_each_environment(embodiment)):
            self.env.env_method("set_embodiment", values, indices=index)

    def reset(self, *, seed=None, options=None, embodiment=None):
        """Reset every environment, optionally with a new embodiment for each.

        Each environment applies its own design during its own reset rather than having one
        installed beforehand, which is what lets a design that cannot be battled with come back as
        `embodiment_rejected` in the returned infos instead of raising inside a worker and failing
        the whole batch.

        Omitting the embodiment keeps whatever each environment already has, which is what the
        automatic reset after a finished episode relies on.

        :returns: the observations, the infos, and `embodiment_rejected`, one flag per environment.
        """
        if embodiment is not None:
            if options is not None:
                raise ValueError("pass the design as embodiment or inside options, not both")
            options = [{"embodiment": values} for values in self._embodiments_for_each_environment(embodiment)]
        observations, infos = self.env.reset(seed=seed, options=options)
        rejected = infos.get("embodiment_rejected")
        self.embodiment_rejected = np.zeros(self.num_envs, dtype=bool) if rejected is None else np.asarray(rejected, dtype=bool)
        return observations, infos, self.embodiment_rejected

    def step(self, actions):
        """Step every environment, refusing to battle while any design was rejected.

        An environment whose design was rejected never started a battle, so stepping it would
        simulate nothing. Reporting that here names the environments responsible, rather than
        letting it surface as a failure inside a worker.
        """
        if self.embodiment_rejected.any():
            rejected = np.flatnonzero(self.embodiment_rejected).tolist()
            raise RuntimeError(
                f"environments {rejected} were given an embodiment that was rejected, so they have "
                f"no battle to step; reset them with designs this environment accepts"
            )
        return self.env.step(actions)

    def close(self, **kwargs) -> None:
        try:
            self.env.close(**kwargs)
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
    reset_timeout: float = 300.0,
    worker_startup_timeout: float = 300.0,
    start_method: str = "spawn",
    monitor: bool = True,
    showdown_path: Path | None = None,
    host: str = DEFAULT_SHOWDOWN_HOST,
    base_port: int | None = None,
    startup_timeout: float | None = None,
    verbose: bool = False,
    **env_kwargs,
):
    """Create `num_envs` codesign environments on a Showdown cluster they own, and their base designs.

    Closing the vector environment shuts every launched server down.

    :returns: the vector environment and one `base_embodiment` per environment, in order.

    :param num_envs: how many environments to run, packed `envs_per_worker` to a worker process.
    :param shards: how many Showdown servers to launch. Defaults to one per `envs_per_shard`
        environments; environment `rank` battles on shard `rank % shards`.
    :param base_port: launch the shards from this port instead of a range claimed from the
        operating system.
    :param startup_timeout: how long the shards have to start serving. Defaults to a budget that
        grows with `shards`, since a large cluster boots every server against the same machine.
    :param reset_timeout: maximum time for all workers to complete a reset before failed workers
        are reconstructed.
    :param worker_startup_timeout: maximum time for a new or reconstructed worker to build its
        environments and report ready.
    """
    if "server_configuration" in env_kwargs:
        raise ValueError("server_configuration cannot be set; make_vec launches and owns its Showdown servers")
    if shards is None:
        shards = shard_count(num_envs, envs_per_shard)
    elif shards < 1:
        raise ValueError(f"shards must be at least one, got {shards}")
    if num_envs < 1:
        raise ValueError(f"num_envs must be at least one, got {num_envs}")

    # Resolve and validate the shared format once in the parent. Besides avoiding one Node process
    # per environment, this catches an incomplete or corrupt Showdown build before hundreds of
    # workers and websocket clients have been started.
    if env_kwargs.get("format_config") is None:
        env_kwargs["format_config"] = FormatConfig.from_showdown(
            PokEnv.default_battle_format(opponent, env_kwargs), showdown_path or default_showdown_path()
        )

    stack = ExitStack()
    try:
        cluster = _launch_cluster(stack, shards, host, base_port, showdown_path, verbose, startup_timeout)
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
            step_timeout=step_timeout, step_timeout_reward=step_timeout_reward, reset_timeout=reset_timeout,
            worker_startup_timeout=worker_startup_timeout,
        )
        if monitor:
            vector_env = SameStepEpisodeStatistics(vector_env)
        owning_env = ServerOwningVecEnv(vector_env, stack, cluster)
        base_embodiments = owning_env.env.unwrapped.get_attr("base_embodiment")
    except BaseException:
        stack.close()
        raise
    return owning_env, base_embodiments
