import multiprocessing as mp
import time
import traceback
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import cloudpickle


class _CloudpickleWrapper:
    def __init__(self, value):
        self.value = value

    def __getstate__(self):
        return cloudpickle.dumps(self.value)

    def __setstate__(self, value):
        self.value = cloudpickle.loads(value)


@dataclass(frozen=True)
class IsolatedEvaluationResult:
    mean_return: float
    episode_returns: tuple[float, ...]
    timed_out_episodes: int


def _evaluation_episode_worker(remote, parent_remote, env_factory: _CloudpickleWrapper) -> None:
    """Own one disposable environment and exchange observations for actions."""
    parent_remote.close()
    env = None
    try:
        env = env_factory.value()
        observation, _ = env.reset()
        remote.send(("observation", observation))
        terminated = truncated = False
        while not terminated and not truncated:
            action = remote.recv()
            observation, reward, terminated, truncated, _ = env.step(action)
            remote.send(("transition", observation, float(reward), bool(terminated), bool(truncated)))
    except (BrokenPipeError, EOFError):
        pass
    except Exception:
        try:
            remote.send(("error", traceback.format_exc()))
        except (BrokenPipeError, EOFError, OSError):
            pass
    finally:
        if env is not None:
            env.close()
        remote.close()


def evaluate_isolated(
    env_factory: Callable[[int], Any], action_selector: Callable[[Any], Any], episodes: int, timeout: float,
    timeout_return: float = -10.0, start_method: str = "spawn",
) -> IsolatedEvaluationResult:
    """Evaluate each episode in a disposable environment process with a hard deadline."""
    if episodes < 1:
        raise ValueError("episodes must be at least one")
    if timeout <= 0:
        raise ValueError("timeout must be greater than zero")

    episode_returns = []
    timed_out_episodes = 0
    context = mp.get_context(start_method)

    for episode in range(episodes):
        remote, work_remote = context.Pipe()
        factory = _CloudpickleWrapper(lambda episode=episode: env_factory(episode))
        process = context.Process(target=_evaluation_episode_worker, args=(work_remote, remote, factory), daemon=True)
        process.start()
        work_remote.close()
        deadline = time.monotonic() + timeout
        episode_return = 0.0
        completed = False
        try:
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0 or not remote.poll(remaining):
                    timed_out_episodes += 1
                    episode_returns.append(float(timeout_return))
                    break
                try:
                    message = remote.recv()
                except EOFError:
                    raise RuntimeError(f"Evaluation episode {episode + 1} exited without returning a result") from None
                if message[0] == "error":
                    raise RuntimeError(f"Evaluation episode {episode + 1} failed:\n{message[1]}")
                if message[0] == "observation":
                    observation = message[1]
                else:
                    _, observation, reward, terminated, truncated = message
                    episode_return += reward
                    if terminated or truncated:
                        episode_returns.append(episode_return)
                        completed = True
                        break
                remote.send(action_selector(observation))
        finally:
            remote.close()
            process.join(timeout=5 if completed else 0)
            if process.is_alive():
                process.terminate()
                process.join(timeout=5)

    returns = tuple(episode_returns)
    return IsolatedEvaluationResult(float(sum(returns) / len(returns)), returns, timed_out_episodes)
