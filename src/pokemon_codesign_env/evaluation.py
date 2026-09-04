import multiprocessing as mp
import time
import traceback
from collections.abc import Callable
from contextlib import suppress
from dataclasses import dataclass
from typing import Any

import cloudpickle
import gymnasium as gym


class SeededResetEnv(gym.Wrapper):
    """Apply a fixed seed when an external evaluator calls reset without one."""

    def __init__(self, env, seed: int):
        super().__init__(env)
        self.initial_seed = seed

    def reset(self, *, seed=None, options=None):
        return self.env.reset(
            seed=self.initial_seed if seed is None else seed,
            options=options,
        )


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
    episode_lengths: tuple[int, ...]
    wins: tuple[bool | None, ...]
    turn_counts: tuple[int | None, ...]

    @property
    def win_rate(self) -> float | None:
        known = [won for won in self.wins if won is not None]
        return sum(known) / len(known) if known else None

    @property
    def mean_episode_length(self) -> float:
        return sum(self.episode_lengths) / len(self.episode_lengths)

    @property
    def mean_turn_count(self) -> float | None:
        known = [turns for turns in self.turn_counts if turns is not None]
        return sum(known) / len(known) if known else None


def _episode_summary(env, info: dict, episode_length: int):
    won = info.get("won", info.get("win"))
    turns = info.get("turn", info.get("turn_count"))
    unwrapped = getattr(env, "unwrapped", env)
    battle_env = getattr(unwrapped, "env", unwrapped)
    battle = getattr(battle_env, "battle1", None)
    if battle is not None:
        won = getattr(battle, "won", won)
        turns = getattr(battle, "turn", turns)
    return episode_length, None if won is None else bool(won), None if turns is None else int(turns)


def _evaluation_episode_worker(remote, parent_remote, env_factory: _CloudpickleWrapper) -> None:
    """Own one disposable environment and exchange observations for actions."""
    parent_remote.close()
    env = None
    try:
        env = env_factory.value()
        observation, _ = env.reset()
        remote.send(("observation", observation))
        terminated = truncated = False
        episode_length = 0
        info = {}
        while not terminated and not truncated:
            action = remote.recv()
            observation, reward, terminated, truncated, info = env.step(action)
            episode_length += 1
            summary = _episode_summary(env, dict(info or {}), episode_length)
            remote.send(
                (
                    "transition",
                    observation,
                    float(reward),
                    bool(terminated),
                    bool(truncated),
                    summary,
                )
            )
    except (BrokenPipeError, EOFError):
        pass
    except Exception:
        with suppress(BrokenPipeError, EOFError, OSError):
            remote.send(("error", traceback.format_exc()))
    finally:
        if env is not None:
            env.close()
        remote.close()


def evaluate_isolated(
    env_factory: Callable[[int], Any],
    action_selector: Callable[[Any], Any],
    episodes: int,
    timeout: float,
    timeout_return: float = -10.0,
    start_method: str = "spawn",
) -> IsolatedEvaluationResult:
    """Evaluate each episode in a disposable environment process with a hard deadline."""
    if episodes < 1:
        raise ValueError("episodes must be at least one")
    if timeout <= 0:
        raise ValueError("timeout must be greater than zero")

    episode_returns = []
    episode_lengths = []
    wins = []
    turn_counts = []
    timed_out_episodes = 0
    context = mp.get_context(start_method)

    for episode in range(episodes):
        remote, work_remote = context.Pipe()
        factory = _CloudpickleWrapper(lambda episode=episode: env_factory(episode))
        process = context.Process(
            target=_evaluation_episode_worker, args=(work_remote, remote, factory), daemon=True
        )
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
                    episode_lengths.append(0)
                    wins.append(None)
                    turn_counts.append(None)
                    break
                try:
                    message = remote.recv()
                except EOFError:
                    raise RuntimeError(
                        f"Evaluation episode {episode + 1} exited without returning a result"
                    ) from None
                if message[0] == "error":
                    raise RuntimeError(f"Evaluation episode {episode + 1} failed:\n{message[1]}")
                if message[0] == "observation":
                    observation = message[1]
                else:
                    _, observation, reward, terminated, truncated, summary = message
                    episode_return += reward
                    if terminated or truncated:
                        episode_returns.append(episode_return)
                        length, won, turns = summary
                        episode_lengths.append(length)
                        wins.append(won)
                        turn_counts.append(turns)
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
    return IsolatedEvaluationResult(
        float(sum(returns) / len(returns)),
        returns,
        timed_out_episodes,
        tuple(episode_lengths),
        tuple(wins),
        tuple(turn_counts),
    )
