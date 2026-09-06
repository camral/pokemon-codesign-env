"""Measure how many environment steps per second the codesign environment sustains.

A Pokemon Showdown battle is I/O bound: almost all of a step is spent waiting on the simulator
rather than computing, so throughput comes from running many environments at once rather than from
making any one of them faster. This measures that, with a policy cheap enough not to matter:
uniformly random legal actions, chosen from the action mask each observation carries.

    python examples/benchmark_fps.py --num-envs 512

Setup and stepping are timed separately, because launching the Showdown servers and opening a
battle for every environment is a fixed cost paid once, while the step rate is what a training run
actually lives with.

The headline number is the effective rate: environment steps per second including the episode
boundaries. Autoreset starts the next battle inside the step that ended the previous one, so a run
short enough to finish no episodes measures only the stepping of battles already under way and
flatters the result. That case is called out rather than reported as the effective rate.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np

import pokemon_codesign_env as pce
from pokemon_codesign_env.configs import AttributeMode, FormatConfig, SelectionMode, TeamSearchConfig
from pokemon_codesign_env.envs import RandomPlayerEnv
from pokemon_codesign_env.gym_utils import shard_count
from pokemon_codesign_env.spaces import TeamDesignSpace

TEAM_SIZE = 6


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--num-envs", type=int, default=512)
    parser.add_argument("--steps", type=int, default=50, help="Measured vector steps, after the warmup.")
    parser.add_argument("--warmup-steps", type=int, default=10, help="Unmeasured steps, so the first battles settle.")
    parser.add_argument("--envs-per-worker", type=int, default=32)
    parser.add_argument("--shards", type=int, default=0, help="Showdown servers, or 0 for one per 32 environments.")
    parser.add_argument("--step-timeout", type=float, default=120.0)
    parser.add_argument(
        "--startup-timeout", type=float,
        help="Seconds for the Showdown servers to start. Defaults to a budget that grows with --shards.",
    )
    parser.add_argument("--verbose", action="store_true", help="Report cluster startup progress.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, help="Write the measurements here as JSON.")
    return parser.parse_args()


def random_legal_actions(observation, rng: np.random.Generator) -> np.ndarray:
    """One uniformly random legal action per environment, read off the action mask."""
    mask = np.asarray(observation["action_mask"])
    # Ranking random draws inside the mask picks a legal action without a Python loop. A row with
    # nothing legal falls back to action 0, which the environment masks to a single choice anyway.
    return np.where(mask.any(axis=1), (rng.random(mask.shape) * mask).argmax(axis=1), 0).astype(np.int64)


def summarize(
    *, num_envs: int, shards: int, envs_per_worker: int, vector_steps: int, episodes: int,
    boundary_vector_steps: int, boundary_seconds: float, construction_seconds: float,
    reset_seconds: float, step_seconds: float, teardown_seconds: float,
) -> dict:
    """Turn the phase timings into the reported measurements."""
    environment_steps = vector_steps * num_envs
    ongoing_vector_steps = vector_steps - boundary_vector_steps
    ongoing_seconds = step_seconds - boundary_seconds
    return {
        "num_envs": num_envs,
        "shards": shards,
        "envs_per_worker": envs_per_worker,
        "vector_steps": vector_steps,
        "environment_steps": environment_steps,
        "episodes_finished": episodes,
        "construction_seconds": round(construction_seconds, 2),
        "reset_seconds": round(reset_seconds, 2),
        "step_seconds": round(step_seconds, 2),
        "teardown_seconds": round(teardown_seconds, 2),
        # What a training run sustains: every environment step, including the episode boundaries.
        "effective_steps_per_second": round(environment_steps / step_seconds, 1),
        "vector_steps_per_second": round(vector_steps / step_seconds, 2),
        "seconds_per_vector_step": round(step_seconds / vector_steps, 3),
        "steps_per_episode": round(environment_steps / episodes, 1) if episodes else None,
        "episodes_per_second": round(episodes / step_seconds, 2),
        "vector_steps_with_an_episode_boundary": boundary_vector_steps,
        "episode_boundary_seconds": round(boundary_seconds, 2),
        "episode_boundary_share_of_step_time": round(boundary_seconds / step_seconds, 3),
        # Battles already under way, with no episode starting: the ceiling the effective rate is
        # pulled down from. Undefined when every vector step happened to finish something.
        "ongoing_battle_steps_per_second": (
            round(ongoing_vector_steps * num_envs / ongoing_seconds, 1)
            if ongoing_vector_steps and ongoing_seconds > 0 else None
        ),
    }


def benchmark(args: argparse.Namespace) -> dict:
    format_config = FormatConfig.from_showdown(RandomPlayerEnv.BATTLE_FORMAT)
    search_config = TeamSearchConfig(
        format=format_config, team_size=TEAM_SIZE,
        species=SelectionMode.OPTIMIZED, natures=AttributeMode.OPTIMIZED,
    )
    embodiment_space = TeamDesignSpace(search_config, seed=args.seed)
    shards = args.shards or shard_count(args.num_envs)
    rng = np.random.default_rng(args.seed)

    print(
        f"environments={args.num_envs} shards={shards} envs_per_worker={args.envs_per_worker} "
        f"workers={-(-args.num_envs // args.envs_per_worker)} format={format_config.format_id}",
        flush=True,
    )

    started = time.perf_counter()
    vector_env, _base_embodiments = pce.make_vec(
        "random", num_envs=args.num_envs, format_config=format_config, embodiment_config=search_config,
        shards=shards, envs_per_worker=args.envs_per_worker, step_timeout=args.step_timeout,
        startup_timeout=args.startup_timeout, verbose=args.verbose, monitor=False,
    )
    construction_seconds = time.perf_counter() - started
    print(f"construction_seconds={construction_seconds:.1f}", flush=True)

    try:
        designs = [embodiment_space.sample() for _ in range(args.num_envs)]
        started = time.perf_counter()
        observation, _infos, rejected = vector_env.reset(seed=args.seed, embodiment=designs)
        reset_seconds = time.perf_counter() - started
        print(f"reset_seconds={reset_seconds:.1f} rejected={int(np.sum(rejected))}", flush=True)
        if rejected.any():
            raise RuntimeError(f"{int(np.sum(rejected))} designs were rejected, so those environments cannot step")

        for step in range(args.warmup_steps):
            observation, *_ = vector_env.step(random_legal_actions(observation, rng))
        print(f"warmup_steps={args.warmup_steps} done", flush=True)

        # A finished episode is reset inside the step that ended it, so the cost of starting the
        # next battle lands on that step rather than anywhere a caller could measure separately.
        # Timing each vector step and splitting on whether an episode ended in it separates what a
        # training run actually sustains from the rate of stepping battles that are already going.
        episodes = 0
        boundary_seconds = 0.0
        boundary_vector_steps = 0
        started = time.perf_counter()
        for step in range(args.steps):
            step_started = time.perf_counter()
            observation, _rewards, terminations, truncations, _infos = vector_env.step(
                random_legal_actions(observation, rng)
            )
            elapsed = time.perf_counter() - step_started
            finished = int(np.sum(terminations | truncations))
            episodes += finished
            if finished:
                boundary_seconds += elapsed
                boundary_vector_steps += 1
        step_seconds = time.perf_counter() - started
    finally:
        started = time.perf_counter()
        vector_env.close()
        teardown_seconds = time.perf_counter() - started

    return summarize(
        num_envs=args.num_envs, shards=shards, envs_per_worker=args.envs_per_worker, vector_steps=args.steps,
        episodes=episodes, boundary_vector_steps=boundary_vector_steps, boundary_seconds=boundary_seconds,
        construction_seconds=construction_seconds, reset_seconds=reset_seconds, step_seconds=step_seconds,
        teardown_seconds=teardown_seconds,
    )


def main() -> None:
    args = parse_args()
    measurements = benchmark(args)
    print()
    for name, value in measurements.items():
        print(f"{name}={value}")
    print(
        f"\nEffective FPS={measurements['effective_steps_per_second']:.0f} environment steps/second "
        f"across {measurements['num_envs']} environments"
    )
    if not measurements["episodes_finished"]:
        print(
            "WARNING: no episode finished, so this is the rate of stepping battles already under way "
            "and not the effective rate. Starting the next battle is charged to the step that ends an "
            "episode, so raise --steps past one episode length to measure what a training run sustains."
        )
    elif measurements["ongoing_battle_steps_per_second"]:
        print(
            f"  {measurements['ongoing_battle_steps_per_second']:.0f} steps/second between episodes, "
            f"{measurements['episode_boundary_share_of_step_time'] * 100:.0f}% of step time spent on the "
            f"{measurements['episodes_finished']} episode boundaries "
            f"({measurements['steps_per_episode']:.0f} steps/episode)"
        )
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(measurements, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
