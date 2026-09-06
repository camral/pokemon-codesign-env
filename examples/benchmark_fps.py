"""Measure how many environment steps per second the codesign environment sustains.

A Pokemon Showdown battle is I/O bound: almost all of a step is spent waiting on the simulator
rather than computing, so throughput comes from running many environments at once rather than from
making any one of them faster. This measures that, with a policy cheap enough not to matter:
uniformly random legal actions, chosen from the action mask each observation carries.

    python examples/benchmark_fps.py --num-envs 512

Setup and stepping are timed separately, because launching the Showdown servers and opening a
battle for every environment is a fixed cost paid once, while the step rate is what a training run
actually lives with.
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

        episodes = 0
        started = time.perf_counter()
        for step in range(args.steps):
            observation, _rewards, terminations, truncations, _infos = vector_env.step(
                random_legal_actions(observation, rng)
            )
            episodes += int(np.sum(terminations | truncations))
        step_seconds = time.perf_counter() - started
    finally:
        started = time.perf_counter()
        vector_env.close()
        teardown_seconds = time.perf_counter() - started

    environment_steps = args.steps * args.num_envs
    return {
        "num_envs": args.num_envs,
        "shards": shards,
        "envs_per_worker": args.envs_per_worker,
        "vector_steps": args.steps,
        "environment_steps": environment_steps,
        "episodes_finished": episodes,
        "construction_seconds": round(construction_seconds, 2),
        "reset_seconds": round(reset_seconds, 2),
        "step_seconds": round(step_seconds, 2),
        "teardown_seconds": round(teardown_seconds, 2),
        "steps_per_second": round(environment_steps / step_seconds, 1),
        "vector_steps_per_second": round(args.steps / step_seconds, 2),
        "seconds_per_vector_step": round(step_seconds / args.steps, 3),
    }


def main() -> None:
    args = parse_args()
    measurements = benchmark(args)
    print()
    for name, value in measurements.items():
        print(f"{name}={value}")
    print(
        f"\nFPS={measurements['steps_per_second']:.0f} environment steps/second "
        f"across {measurements['num_envs']} environments"
    )
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(measurements, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
