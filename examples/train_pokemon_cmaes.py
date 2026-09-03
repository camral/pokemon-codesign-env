"""Jointly evolve an all-attribute Pokemon team and compact battle controller."""

from __future__ import annotations

import argparse
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from gymnasium import spaces
from poke_env.environment import SinglesEnv

from pokemon_codesign_env.baseline_adapters import (
    TeamDesignVectorAdapter,
    make_all_attribute_search_config,
)
from pokemon_codesign_env.baseline_artifacts import RunMetrics, write_run_artifacts
from pokemon_codesign_env.cmaes_evaluator import PokemonCMAEvaluator
from pokemon_codesign_env.compact_controller import CompactBattleController
from pokemon_codesign_env.configs import FormatConfig, ObservationConfig
from pokemon_codesign_env.fasttd3_factory import ENVIRONMENTS
from pokemon_codesign_env.observations import HumanVisibleBattleEncoder


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--environment", choices=ENVIRONMENTS, default="random")
    parser.add_argument("--budget", type=int, default=100)
    parser.add_argument("--evaluation-repeats", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output-dir", type=Path, default=Path("runs/pokemon_cmaes"))
    return parser.parse_args()


def main():
    args = parse_args()
    baseline_root = Path(__file__).resolve().parents[2] / "codesign-baselines"
    sys.path.insert(0, str(baseline_root))
    from cmaes_evolutionary.main import CMAESConfig, run_cmaes

    environment_class = ENVIRONMENTS[args.environment]
    format_config = FormatConfig.from_showdown(environment_class.BATTLE_FORMAT)
    search_config = make_all_attribute_search_config(format_config)
    adapter = TeamDesignVectorAdapter(search_config)
    encoder = HumanVisibleBattleEncoder.from_showdown(
        format_config, observation_config=ObservationConfig(team_size=6)
    )
    action_count = SinglesEnv.get_action_space_size(format_config.generation)
    observation_space = spaces.Dict(
        {**encoder.observation_space.spaces, "action_mask": spaces.MultiBinary(action_count)}
    )
    controller = CompactBattleController(observation_space, action_count)
    started = time.monotonic()
    config = CMAESConfig(
        design_dimension=adapter.dimension,
        policy_dimension=controller.parameter_count,
        budget=args.budget,
        num_workers=args.num_workers,
        evaluation_repeats=args.evaluation_repeats,
        design_lower_bound=0,
        design_upper_bound=1,
        policy_lower_bound=-1,
        policy_upper_bound=1,
        seed=args.seed,
    )
    evaluator = PokemonCMAEvaluator(args.environment, args.seed)
    if args.num_workers == 1:
        result = run_cmaes(evaluator, config, device=args.device)
    else:
        with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
            result = run_cmaes(evaluator, config, executor=executor, device=args.device)
    parameters = adapter.decode(result.design)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.save(args.output_dir / "best_controller.npy", result.policy)
    write_run_artifacts(
        args.output_dir,
        team_parameters=parameters,
        team_design_schema=adapter.schema(),
        observation_schema=encoder.get_schema(),
        metrics=RunMetrics(
            mean_return=result.score,
            evaluation_battles=result.rollouts_evaluated,
            wall_time_seconds=time.monotonic() - started,
        ),
        metadata={"method": "cmaes", "candidates_evaluated": result.candidates_evaluated},
    )


if __name__ == "__main__":
    main()
