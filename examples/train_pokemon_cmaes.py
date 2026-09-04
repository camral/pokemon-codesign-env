"""Jointly evolve an all-attribute Pokemon team and compact battle controller."""

from __future__ import annotations

import argparse
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from poke_env.environment import SinglesEnv
from poke_env.ps_client import ServerConfiguration

from pokemon_codesign_env.baseline_adapters import (
    TeamDesignVectorAdapter,
    make_all_attribute_search_config,
)
from pokemon_codesign_env.baseline_artifacts import RunMetrics, write_run_artifacts
from pokemon_codesign_env.cmaes_evaluator import PokemonCMAEvaluator
from pokemon_codesign_env.compact_controller import CompactBattleController
from pokemon_codesign_env.configs import FormatConfig, ObservationConfig
from pokemon_codesign_env.evaluation import evaluate_isolated
from pokemon_codesign_env.fasttd3_factory import ENVIRONMENTS
from pokemon_codesign_env.observations import HumanVisibleBattleEncoder
from pokemon_codesign_env.team_designer import TeamDesigner


class _SeededResetEnv(gym.Wrapper):
    def __init__(self, env, seed):
        super().__init__(env)
        self.initial_seed = seed

    def reset(self, *, seed=None, options=None):
        return self.env.reset(
            seed=self.initial_seed if seed is None else seed,
            options=options,
        )


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--environment", choices=ENVIRONMENTS, default="random")
    parser.add_argument("--budget", type=int, default=100)
    parser.add_argument("--evaluation-repeats", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--showdown-host", default="127.0.0.1")
    parser.add_argument("--showdown-port", type=int, default=8000)
    parser.add_argument("--challenge-timeout", type=float, default=60.0)
    parser.add_argument("--held-out-evaluation-battles", type=int, default=16)
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
    evaluator = PokemonCMAEvaluator(
        args.environment,
        args.seed,
        args.evaluation_repeats,
        showdown_host=args.showdown_host,
        showdown_port=args.showdown_port,
        challenge_timeout=args.challenge_timeout,
    )
    if args.num_workers == 1:
        result = run_cmaes(evaluator, config, device=args.device)
    else:
        with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
            result = run_cmaes(evaluator, config, executor=executor, device=args.device)
    parameters = adapter.decode(result.design)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.save(args.output_dir / "best_controller.npy", result.policy)
    controller.set_parameters(np.load(args.output_dir / "best_controller.npy"))
    designer = TeamDesigner(search_config, validate_teams=True)
    designer.rng = np.random.default_rng(args.seed + 100_000)
    designer.set_optimizable_parameter_values(parameters)
    packed_team = designer.yield_team()
    server = ServerConfiguration(
        f"ws://{args.showdown_host}:{args.showdown_port}/showdown/websocket",
        "https://play.pokemonshowdown.com/action.php?",
    )
    (args.output_dir / "replays").mkdir(exist_ok=True)

    def evaluation_env(episode):
        return _SeededResetEnv(
            environment_class(
                team=packed_team,
                observation_encoder=encoder,
                server_configuration=server,
                challenge_timeout=args.challenge_timeout,
                save_replays=str(args.output_dir / "replays"),
            ),
            args.seed + 100_000 + episode,
        )

    held_out = evaluate_isolated(
        evaluation_env,
        controller.predict,
        args.held_out_evaluation_battles,
        args.challenge_timeout,
        start_method="spawn",
    )
    write_run_artifacts(
        args.output_dir,
        team_parameters=parameters,
        team_design_schema=adapter.schema(),
        observation_schema=encoder.get_schema(),
        metrics=RunMetrics(
            mean_return=held_out.mean_return,
            evaluation_battles=args.held_out_evaluation_battles,
            training_battles=result.rollouts_evaluated,
            win_rate=held_out.win_rate,
            mean_episode_length=held_out.mean_episode_length,
            mean_turn_count=held_out.mean_turn_count,
            wall_time_seconds=time.monotonic() - started,
        ),
        metadata={
            "method": "cmaes",
            "candidates_evaluated": result.candidates_evaluated,
            "optimization_score": result.score,
            "timed_out_evaluation_battles": held_out.timed_out_episodes,
        },
    )


if __name__ == "__main__":
    main()
