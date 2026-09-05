import multiprocessing as mp
import logging
import time
import traceback
import warnings
from concurrent.futures import ThreadPoolExecutor
from multiprocessing.connection import wait
from collections.abc import Iterable
from typing import Any, Callable, Sequence

import gymnasium as gym
import numpy as np
from gymnasium.vector import AutoresetMode, VectorEnv
from gymnasium.vector.utils import CloudpickleWrapper, batch_space, concatenate, create_empty_array

# One environment, several of them, or all of them when None.
EnvIndices = int | Iterable[int] | None


def _is_wrapped(env: gym.Env, wrapper_class: type[gym.Wrapper]) -> bool:
    """Whether a wrapper of this class appears anywhere in an environment's chain."""
    while isinstance(env, gym.Wrapper):
        if isinstance(env, wrapper_class):
            return True
        env = env.env
    return False


def _stack_observations(observations, single_space: gym.Space):
    """Batch one observation per environment into the vector observation."""
    return concatenate(single_space, observations, create_empty_array(single_space, len(observations)))


def _step_environment(arguments):
    env, action, reset_retries = arguments
    observation, reward, terminated, truncated, info = env.step(action)
    reset_info = {}
    if terminated or truncated:
        # Same-step autoreset: the observation that ended the episode travels in the info under
        # the names Gymnasium uses, and the observation returned is already the next episode's.
        info = {**info, "final_obs": observation, "final_info": info}
        observation, reset_info = _reset_environment((env, None, {}, reset_retries))
    return observation, reward, terminated, truncated, info, reset_info


def _reset_environment(arguments):
    env, seed, options, reset_retries = arguments
    maybe_options = {"options": options} if options else {}
    for attempt in range(reset_retries):
        try:
            return env.reset(seed=seed, **maybe_options)
        except TimeoutError:
            if attempt + 1 == reset_retries:
                raise


def _sharded_worker(remote, parent_remote, env_fn_wrappers, reset_retries: int) -> None:
    parent_remote.close()
    envs = [wrapper() for wrapper in env_fn_wrappers]
    executor = ThreadPoolExecutor(max_workers=len(envs), thread_name_prefix="battle")

    def respond(value) -> None:
        remote.send((True, value))

    try:
        while True:
            command, data = remote.recv()
            try:
                if command == "step":
                    respond(list(executor.map(_step_environment, [(env, action, reset_retries) for env, action in zip(envs, data, strict=True)])))
                elif command == "reset":
                    reset_arguments = [(env, seed, options, reset_retries) for env, (seed, options) in zip(envs, data, strict=True)]
                    respond(list(executor.map(_reset_environment, reset_arguments)))
                elif command == "close":
                    list(executor.map(lambda env: env.close(), envs))
                    respond(None)
                    break
                elif command == "get_spaces":
                    respond((envs[0].observation_space, envs[0].action_space))
                elif command == "get_attr":
                    name, indices = data
                    respond([envs[index].get_wrapper_attr(name) for index in indices])
                elif command == "has_attr":
                    try:
                        for env in envs:
                            env.get_wrapper_attr(data)
                        respond(True)
                    except AttributeError:
                        respond(False)
                elif command == "set_attr":
                    name, value, indices = data
                    respond([setattr(envs[index], name, value) for index in indices])
                elif command == "env_method":
                    name, arguments, keywords, indices = data
                    respond([envs[index].get_wrapper_attr(name)(*arguments, **keywords) for index in indices])
                elif command == "is_wrapped":
                    wrapper_class, indices = data
                    respond([_is_wrapped(envs[index], wrapper_class) for index in indices])
                elif command == "render":
                    respond([env.render() for env in envs])
                else:
                    raise NotImplementedError(f"ShardedVecEnv command {command!r} is not implemented")
            except Exception:
                remote.send((False, traceback.format_exc()))
    except (EOFError, KeyboardInterrupt):
        pass
    finally:
        executor.shutdown(wait=False, cancel_futures=True)
        remote.close()


class ShardedVecEnv(VectorEnv):
    """Run many I/O-bound environments in a small number of batched processes.

    Each worker process owns a batch of environments and steps them on threads, because a Pokemon
    Showdown battle spends its time waiting on a socket rather than on the CPU. That keeps a
    thousand environments to a few dozen processes instead of a thousand.
    """

    def __init__(
        self, env_fns: list[Callable[[], gym.Env]], envs_per_worker: int = 32, reset_retries: int = 3, start_method: str = "spawn",
        step_timeout: float = 60.0, step_timeout_reward: float = 0.0,
    ):
        if not env_fns:
            raise ValueError("ShardedVecEnv requires at least one environment")
        if envs_per_worker < 1:
            raise ValueError("envs_per_worker must be at least one")
        if reset_retries < 1:
            raise ValueError("reset_retries must be at least one")
        if step_timeout <= 0:
            raise ValueError("step_timeout must be greater than zero")
        self.waiting = False
        self.closed = False
        self.step_send_failures = {}
        self.reset_retries = reset_retries
        self.step_timeout = step_timeout
        self.step_timeout_reward = step_timeout_reward
        self.context = mp.get_context(start_method)
        self.worker_sizes = tuple(min(envs_per_worker, len(env_fns) - offset) for offset in range(0, len(env_fns), envs_per_worker))
        self.worker_starts = tuple(sum(self.worker_sizes[:index]) for index in range(len(self.worker_sizes)))
        self.env_fn_batches = [
            [CloudpickleWrapper(env_fn) for env_fn in env_fns[offset:offset + envs_per_worker]]
            for offset in range(0, len(env_fns), envs_per_worker)
        ]
        self.remotes = []
        self.processes = []
        for worker_index in range(len(self.env_fn_batches)):
            self._start_worker(worker_index)
        self.remotes[0].send(("get_spaces", None))
        single_observation_space, single_action_space = self._receive(self.remotes[0])
        self.num_envs = len(env_fns)
        self.single_observation_space = single_observation_space
        self.single_action_space = single_action_space
        self.observation_space = batch_space(single_observation_space, self.num_envs)
        self.action_space = batch_space(single_action_space, self.num_envs)
        # A finished environment is reset inside the same step that ended it, which is what
        # Gymnasium calls same-step autoreset.
        self.metadata = {**VectorEnv.metadata, "autoreset_mode": AutoresetMode.SAME_STEP}
        self.reset_infos = [{} for _ in range(self.num_envs)]

    def _start_worker(self, worker_index: int) -> None:
        remote, work_remote = self.context.Pipe()
        process = self.context.Process(
            target=_sharded_worker, args=(work_remote, remote, self.env_fn_batches[worker_index], self.reset_retries), daemon=True
        )
        process.start()
        work_remote.close()
        if worker_index == len(self.remotes):
            self.remotes.append(remote)
            self.processes.append(process)
        else:
            self.remotes[worker_index] = remote
            self.processes[worker_index] = process

    def _restart_worker(self, worker_index: int, reason: str) -> list:
        worker_start = self.worker_starts[worker_index]
        environment_indices = list(range(worker_start, worker_start + self.worker_sizes[worker_index]))
        process = self.processes[worker_index]
        logging.getLogger(__name__).warning(
            "Restarting stalled environment worker %d (pid=%s, environments=%s, reason=%s)",
            worker_index, process.pid, environment_indices, reason,
        )
        reset_results = None
        for attempt in range(self.reset_retries):
            try:
                self.remotes[worker_index].close()
            except OSError:
                pass
            if process.is_alive():
                process.terminate()
            process.join(timeout=2)
            self._start_worker(worker_index)
            process = self.processes[worker_index]
            try:
                self.remotes[worker_index].send(("reset", [(None, {}) for _ in range(self.worker_sizes[worker_index])]))
                reset_results = self._receive(self.remotes[worker_index], timeout=max(60.0, self.step_timeout))
                break
            except (BrokenPipeError, EOFError, OSError, RuntimeError, TimeoutError):
                if attempt + 1 == self.reset_retries:
                    raise
        if reset_results is None:
            raise RuntimeError(f"Could not restart sharded environment worker {worker_index}")
        recovered = []
        for environment_index, (observation, reset_info) in zip(environment_indices, reset_results, strict=True):
            # Losing a worker ends its episodes without them finishing, which is a truncation.
            info = {
                "final_obs": observation, "final_info": {}, "sharded_worker_restarted": True,
                "sharded_worker_index": worker_index, "environment_index": environment_index, "restart_reason": reason,
            }
            recovered.append((observation, self.step_timeout_reward, False, True, info, reset_info))
        return recovered

    def _receive_step_workers(self) -> list[list]:
        results = {
            worker_index: self._restart_worker(worker_index, reason)
            for worker_index, reason in self.step_send_failures.items()
        }
        pending = set(range(len(self.remotes))) - set(results)
        self.step_send_failures = {}
        deadline = time.monotonic() + self.step_timeout
        while pending:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            remote_indices = {self.remotes[index]: index for index in pending}
            ready = wait(list(remote_indices), timeout=remaining)
            if not ready:
                break
            for remote in ready:
                worker_index = remote_indices[remote]
                pending.remove(worker_index)
                try:
                    results[worker_index] = self._receive(remote)
                except (EOFError, OSError, RuntimeError) as error:
                    results[worker_index] = self._restart_worker(worker_index, f"worker failure: {error}")
        for worker_index in sorted(pending):
            results[worker_index] = self._restart_worker(worker_index, f"step timeout after {self.step_timeout:g}s")
        return [results[index] for index in range(len(self.remotes))]

    @staticmethod
    def _receive(remote, timeout: float | None = None):
        if timeout is not None and not remote.poll(timeout):
            raise TimeoutError("Timed out waiting for a sharded environment worker")
        success, value = remote.recv()
        if not success:
            raise RuntimeError(f"Sharded environment worker failed:\n{value}")
        return value

    def _split(self, values) -> list:
        batches = []
        offset = 0
        for size in self.worker_sizes:
            batches.append(values[offset:offset + size])
            offset += size
        return batches

    def step_async(self, actions: np.ndarray) -> None:
        self.step_send_failures = {}
        for worker_index, (remote, batch) in enumerate(zip(self.remotes, self._split(actions), strict=True)):
            try:
                remote.send(("step", batch))
            except (BrokenPipeError, EOFError, OSError) as error:
                self.step_send_failures[worker_index] = f"could not submit step: {error}"
        self.waiting = True

    def step_wait(self):
        try:
            results = [result for worker_results in self._receive_step_workers() for result in worker_results]
        finally:
            self.waiting = False
        observations, rewards, terminations, truncations, infos, self.reset_infos = zip(*results, strict=True)
        return (
            _stack_observations(observations, self.single_observation_space),
            np.asarray(rewards, dtype=np.float64),
            np.asarray(terminations, dtype=bool),
            np.asarray(truncations, dtype=bool),
            self._batch_infos(infos),
        )

    def step(self, actions):
        self.step_async(actions)
        return self.step_wait()

    def _batch_infos(self, infos) -> dict[str, Any]:
        """Gather one info per environment into Gymnasium's vectorized info dictionary."""
        vector_infos: dict[str, Any] = {}
        for environment_index, info in enumerate(infos):
            vector_infos = self._add_info(vector_infos, info, environment_index)
        return vector_infos

    def reset(self, *, seed=None, options=None):
        """Reset every environment.

        `options` may be one dictionary for all of them, or one per environment, which is how a
        caller gives each environment something different to reset with.
        """
        if seed is None or isinstance(seed, int):
            seeds = [seed] * self.num_envs if seed is None else [seed + index for index in range(self.num_envs)]
        else:
            seeds = list(seed)
        if options is None or isinstance(options, dict):
            per_environment_options = [options] * self.num_envs
        else:
            per_environment_options = list(options)
        if len(seeds) != self.num_envs or len(per_environment_options) != self.num_envs:
            raise ValueError(f"Expected one seed and one option dictionary per environment, got {len(seeds)} and {len(per_environment_options)}")

        arguments = self._split(list(zip(seeds, per_environment_options, strict=True)))
        for remote, batch in zip(self.remotes, arguments, strict=True):
            remote.send(("reset", batch))

        # Every worker has been sent a command, so every reply must be read even once one of them
        # has failed. A reply left in the pipe desynchronises that worker for good: the next
        # command reads this reply instead of its own, and a transient failure that the retries
        # exist to absorb would take the whole vector environment down with it.
        results, failures = [], []
        for remote in self.remotes:
            try:
                results.extend(self._receive(remote))
            except Exception as error:
                failures.append(error)
        if failures:
            if len(failures) > 1:
                raise RuntimeError(f"{len(failures)} sharded environment workers failed on reset") from failures[0]
            raise failures[0]

        observations, self.reset_infos = zip(*results, strict=True)
        return _stack_observations(observations, self.single_observation_space), self._batch_infos(self.reset_infos)

    def close(self, **kwargs) -> None:
        if self.closed:
            return
        if self.waiting:
            for remote in self.remotes:
                try:
                    self._receive(remote, timeout=10)
                except (EOFError, OSError, TimeoutError):
                    pass
        for remote in self.remotes:
            try:
                remote.send(("close", None))
            except (BrokenPipeError, EOFError, OSError):
                pass
        for remote in self.remotes:
            try:
                self._receive(remote, timeout=10)
            except (EOFError, OSError, TimeoutError):
                pass
        for process in self.processes:
            process.join(timeout=2)
            if process.is_alive():
                process.terminate()
                process.join()
        self.closed = True

    def _get_indices(self, indices: EnvIndices) -> Iterable[int]:
        if indices is None:
            return range(self.num_envs)
        if isinstance(indices, int):
            return [indices]
        return indices

    def _locations(self, indices: EnvIndices = None) -> list[tuple[int, int, int]]:
        selected = list(self._get_indices(indices))
        locations = []
        for output_index, environment_index in enumerate(selected):
            worker_index = next(index for index in range(len(self.worker_sizes)) if environment_index < sum(self.worker_sizes[:index + 1]))
            worker_start = sum(self.worker_sizes[:worker_index])
            locations.append((output_index, worker_index, environment_index - worker_start))
        return locations

    def _group_locations(self, indices: EnvIndices = None) -> tuple[list[tuple[int, int, int]], dict[int, list[int]]]:
        locations = self._locations(indices)
        grouped = {}
        for _, worker_index, local_index in locations:
            grouped.setdefault(worker_index, []).append(local_index)
        return locations, grouped

    def get_attr(self, attr_name: str, indices: EnvIndices = None) -> list[Any]:
        locations, grouped = self._group_locations(indices)
        for worker_index, local_indices in grouped.items():
            self.remotes[worker_index].send(("get_attr", (attr_name, local_indices)))
        responses = {worker_index: iter(self._receive(self.remotes[worker_index])) for worker_index in grouped}
        return [next(responses[worker_index]) for _, worker_index, _ in locations]

    def has_attr(self, attr_name: str) -> bool:
        for remote in self.remotes:
            remote.send(("has_attr", attr_name))
        return all(self._receive(remote) for remote in self.remotes)

    def set_attr(self, attr_name: str, value: Any, indices: EnvIndices = None) -> None:
        _, grouped = self._group_locations(indices)
        for worker_index, local_indices in grouped.items():
            self.remotes[worker_index].send(("set_attr", (attr_name, value, local_indices)))
        for worker_index in grouped:
            self._receive(self.remotes[worker_index])

    def env_method(self, method_name: str, *method_args, indices: EnvIndices = None, **method_kwargs) -> list[Any]:
        locations, grouped = self._group_locations(indices)
        for worker_index, local_indices in grouped.items():
            self.remotes[worker_index].send(("env_method", (method_name, method_args, method_kwargs, local_indices)))
        responses = {worker_index: iter(self._receive(self.remotes[worker_index])) for worker_index in grouped}
        return [next(responses[worker_index]) for _, worker_index, _ in locations]

    def env_is_wrapped(self, wrapper_class: type[gym.Wrapper], indices: EnvIndices = None) -> list[bool]:
        locations, grouped = self._group_locations(indices)
        for worker_index, local_indices in grouped.items():
            self.remotes[worker_index].send(("is_wrapped", (wrapper_class, local_indices)))
        responses = {worker_index: iter(self._receive(self.remotes[worker_index])) for worker_index in grouped}
        return [next(responses[worker_index]) for _, worker_index, _ in locations]

    def get_images(self) -> Sequence[np.ndarray | None]:
        if self.render_mode != "rgb_array":
            warnings.warn(f"The render mode is {self.render_mode}, but get_images requires rgb_array")
            return [None for _ in range(self.num_envs)]
        for remote in self.remotes:
            remote.send(("render", None))
        return [image for remote in self.remotes for image in self._receive(remote)]
