"""Iterative frozen-policy self-play with SB3 PPO and Nevergrad NGOpt.

Round zero bootstraps against RandomPlayer. Every later round trains rollout
populations against the previous round's frozen champion controller and
embodiment, then promotes the final NGOpt recommendation.
"""

import argparse
import json
import time
from pathlib import Path

import nevergrad as ng
import numpy as np
from poke_env.environment import SinglesEnv
from poke_env.player import Player
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor

from pokemon_codesign_env.configs import FormatConfig, ObservationConfig
from pokemon_codesign_env.envs import RandomPlayerEnv
from pokemon_codesign_env.envs.base import BaselineOpponentEnv
from pokemon_codesign_env.observations import HumanVisibleBattleEncoder
from pokemon_codesign_env.team_pool import RandomTeamPool

if __package__:
    from examples.train_sb3_ngopt import (
        MaskedActorCriticPolicy,
        RetryingTeamDesigner,
        RolloutTeamOptimizerCallback,
        WandbEvaluationLogger,
        add_compute_arguments,
        collect_evaluation_replays,
        evaluation_account_configurations,
        evaluation_replay_dir,
        make_ngopt_parameterization,
        make_search_config,
        make_team,
        make_training_callback,
        make_vector_env,
        managed_showdown_cluster,
        rollout_population_evaluations,
        sample_team_population,
        server_configuration_for,
        showdown_cluster,
        validate_compute_arguments,
        validate_device,
        wandb_config,
    )
else:
    from train_sb3_ngopt import (
        MaskedActorCriticPolicy,
        RetryingTeamDesigner,
        RolloutTeamOptimizerCallback,
        WandbEvaluationLogger,
        add_compute_arguments,
        collect_evaluation_replays,
        evaluation_account_configurations,
        evaluation_replay_dir,
        make_ngopt_parameterization,
        make_search_config,
        make_team,
        make_training_callback,
        make_vector_env,
        managed_showdown_cluster,
        rollout_population_evaluations,
        sample_team_population,
        server_configuration_for,
        showdown_cluster,
        validate_compute_arguments,
        validate_device,
        wandb_config,
    )

TEAM_SIZE = 6
OBSERVATION_CONFIG = ObservationConfig(team_size=TEAM_SIZE)
FROZEN_MODEL_CACHE = {}


class FrozenPolicyPlayer(Player):
    """Use a frozen masked PPO policy as a poke-env opponent."""

    def __init__(self, model_path: Path, device: str, observation_encoder: HumanVisibleBattleEncoder, **kwargs):
        cache_key = (str(model_path.resolve()), device)
        if cache_key not in FROZEN_MODEL_CACHE:
            FROZEN_MODEL_CACHE[cache_key] = PPO.load(model_path, device=device)
        self.model = FROZEN_MODEL_CACHE[cache_key]
        self.observation_encoder = observation_encoder
        super().__init__(**kwargs)

    def choose_move(self, battle):
        observation = {**self.observation_encoder.encode(battle), "action_mask": np.asarray(SinglesEnv.get_action_mask(battle), dtype=np.int8)}
        action, _ = self.model.predict(observation, deterministic=True)
        return SinglesEnv.action_to_order(np.int64(action), battle, strict=False)


class FrozenPolicyOpponentEnv(BaselineOpponentEnv):
    """Gen 4 environment whose opponent is a frozen PPO snapshot."""

    BATTLE_FORMAT = RandomPlayerEnv.BATTLE_FORMAT
    OPPONENT_TEAM_FORMAT = RandomPlayerEnv.OPPONENT_TEAM_FORMAT
    OPPONENT_CLASS = FrozenPolicyPlayer


def make_env(
    learner_team: RetryingTeamDesigner,
    observation_encoder: HumanVisibleBattleEncoder,
    champion_controller: Path | None,
    champion_team: RetryingTeamDesigner | None,
    opponent_device: str,
    random_team=None,
    monitor: bool = True,
    save_replays: bool | str = False,
    **env_kwargs,
):
    """Create a bootstrap or frozen-champion training environment."""
    if champion_controller is None:
        env = RandomPlayerEnv(team=learner_team, opponent_team=random_team, observation_encoder=observation_encoder, save_replays=save_replays, **env_kwargs)
    else:
        env = FrozenPolicyOpponentEnv(
            team=learner_team,
            opponent_team=champion_team,
            observation_encoder=observation_encoder,
            opponent_kwargs={"model_path": champion_controller, "device": opponent_device, "observation_encoder": observation_encoder},
            save_replays=save_replays,
            **env_kwargs,
        )
    return Monitor(env) if monitor else env


def make_training_env(
    parameters: dict,
    search_config,
    observation_encoder: HumanVisibleBattleEncoder,
    champion_controller: Path | None,
    champion_parameters: dict | None,
    args: argparse.Namespace,
    seed: int,
    learner_team_pool: RandomTeamPool | None = None,
    random_team_pool: RandomTeamPool | None = None,
    champion_team_pool: RandomTeamPool | None = None,
    num_envs: int | None = None,
    vec_env: str | None = None,
    save_replays: bool | str = False,
    account_configuration1=None,
    account_configuration2=None,
    learner_teams: list[str] | None = None,
):
    """Create vectorized bootstrap or frozen-champion workers."""
    num_envs = num_envs or args.num_envs
    vec_env = vec_env or args.vec_env

    def initialize(rank: int):
        def make_worker():
            champion_team = None
            if champion_parameters is not None:
                champion_team = (
                    make_team(search_config, champion_parameters, seed + args.num_envs + rank)
                    if champion_team_pool is None else champion_team_pool.builder(seed + args.num_envs + rank)
                )
            learner_team = (
                learner_teams[rank] if learner_teams is not None else
                make_team(search_config, parameters, seed + rank) if learner_team_pool is None else learner_team_pool.builder(seed + rank)
            )
            return make_env(
                learner_team,
                observation_encoder,
                champion_controller,
                champion_team,
                args.opponent_device,
                random_team=None if random_team_pool is None else random_team_pool.builder(seed + rank),
                monitor=False,
                save_replays=save_replays,
                server_configuration=server_configuration_for(args, rank),
                challenge_timeout=args.challenge_timeout,
                account_configuration1=account_configuration1,
                account_configuration2=account_configuration2,
            )

        return make_worker

    return make_vector_env([initialize(rank) for rank in range(num_envs)], vec_env, args.envs_per_worker, args.reset_retries)


def evaluate_controller(model: PPO, env, episodes: int) -> float:
    """Return mean deterministic return against the frozen round opponent."""
    episode_returns = []
    for _ in range(episodes):
        observation, _ = env.reset()
        terminated = truncated = False
        episode_return = 0.0
        while not terminated and not truncated:
            action, _ = model.predict(observation, deterministic=True)
            observation, reward, terminated, truncated, _ = env.step(action)
            episode_return += float(reward)
        episode_returns.append(episode_return)
    return float(np.mean(episode_returns))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-play-rounds", type=int, default=3)
    parser.add_argument("--ppo-timesteps", type=int, default=2048)
    parser.add_argument("--evaluation-episodes", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    add_compute_arguments(parser)
    parser.add_argument("--opponent-device", default="cpu")
    parser.add_argument("--output-dir", type=Path, default=Path("runs/self_play"))
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--launch-server", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    validate_compute_arguments(args)
    validate_device(args.device)
    validate_device(args.opponent_device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    format_config = FormatConfig.from_showdown(RandomPlayerEnv.BATTLE_FORMAT)
    search_config = make_search_config(format_config)
    observation_encoder = HumanVisibleBattleEncoder.from_showdown(format_config, observation_config=OBSERVATION_CONFIG)
    (args.output_dir / "observation_schema.json").write_text(json.dumps(observation_encoder.get_schema(), indent=2) + "\n", encoding="utf-8")

    champion_controller = None
    champion_parameters = None
    with (
        WandbEvaluationLogger(args, wandb_config(args)) as wandb_logger,
        managed_showdown_cluster(args.launch_server, format_config.showdown_path, showdown_cluster(args), args.verbose),
    ):
        pool_size = args.random_team_pool_size or (max(args.num_envs, 1_024) if args.vec_env == "sharded" else 0)
        random_team_pool = None
        if pool_size:
            wandb_logger.log_system("team_pool_generating", team_pool_size=pool_size)
            pool_started_at = time.monotonic()
            random_team_pool = RandomTeamPool.from_showdown(format_config.showdown_path, RandomPlayerEnv.OPPONENT_TEAM_FORMAT, pool_size)
            wandb_logger.log_system("team_pool_ready", team_pool_size=pool_size, team_pool_startup_s=time.monotonic() - pool_started_at)
        rollout_size = args.num_envs * args.rollout_steps
        rollout_count = (args.ppo_timesteps + rollout_size - 1) // rollout_size
        population_evaluations = rollout_population_evaluations(args.ppo_timesteps, args.num_envs, args.rollout_steps)
        for round_index in range(args.self_play_rounds):
            champion_team_pool = None
            champion_pool_size = args.learner_team_pool_size or (max(args.num_envs, 1_024) if args.vec_env == "sharded" else 0)
            if champion_parameters is not None and champion_pool_size:
                wandb_logger.log_system("champion_team_pool_generating", round_index=round_index, team_pool_size=champion_pool_size)
                pool_started_at = time.monotonic()
                champion_designer = make_team(search_config, champion_parameters, args.seed + round_index * population_evaluations)
                champion_designer.yield_team()
                champion_designer.validate_teams = False
                champion_team_pool = RandomTeamPool.from_generator(
                    search_config.format.showdown_path, search_config.format.format_id, champion_pool_size, champion_designer.yield_team
                )
                wandb_logger.log_system(
                    "champion_team_pool_ready", round_index=round_index, team_pool_size=champion_pool_size,
                    team_pool_startup_s=time.monotonic() - pool_started_at,
                )
            parametrization = make_ngopt_parameterization(search_config)
            parametrization.random_state.seed(args.seed + round_index)
            optimizer = ng.optimizers.NGOpt(parametrization=parametrization, budget=population_evaluations, num_workers=args.num_envs)
            population_seed = args.seed + round_index * population_evaluations
            candidates, parameters, teams = sample_team_population(optimizer, search_config, args.num_envs, population_seed)
            training_env = make_training_env(
                parameters[0], search_config, observation_encoder, champion_controller, champion_parameters, args, population_seed,
                random_team_pool=random_team_pool, champion_team_pool=champion_team_pool, learner_teams=teams,
            )
            label = f"round-{round_index + 1}-codesign"
            team_optimizer_callback = RolloutTeamOptimizerCallback(
                optimizer, search_config, candidates, parameters, population_seed, args.ppo_timesteps, wandb_logger, label, round_index
            )
            round_model = PPO(
                MaskedActorCriticPolicy, training_env, n_steps=args.rollout_steps, batch_size=args.batch_size, learning_rate=3e-4,
                gamma=0.99, ent_coef=0.01, seed=population_seed, device=args.device, verbose=int(args.verbose),
            )
            try:
                round_model.learn(
                    total_timesteps=args.ppo_timesteps,
                    callback=make_training_callback(args, wandb_logger, label, 0, round_index, team_optimizer_callback),
                )
            finally:
                training_env.close()

            previous_champion_parameters = champion_parameters
            champion_parameters = optimizer.provide_recommendation().value
            learner_team = make_team(search_config, champion_parameters, population_seed + population_evaluations)
            champion_team = None if previous_champion_parameters is None else make_team(
                search_config, previous_champion_parameters, population_seed + population_evaluations + 1
            )
            replay_dir = evaluation_replay_dir(args.output_dir, f"round_{round_index + 1}_final")
            previous_replays = set(replay_dir.glob("*.html"))
            opponent_name = "RandomPlayer" if champion_controller is None else "SelfPlayOpponent"
            evaluation_accounts = evaluation_account_configurations(opponent_name, round_index)
            evaluation_env = make_env(
                learner_team, observation_encoder, champion_controller, champion_team, args.opponent_device,
                random_team=None if random_team_pool is None else random_team_pool.builder(population_seed + population_evaluations),
                save_replays=str(replay_dir), server_configuration=server_configuration_for(args, args.num_envs), challenge_timeout=args.challenge_timeout,
                account_configuration1=evaluation_accounts[0], account_configuration2=evaluation_accounts[1],
            )
            try:
                round_return = evaluate_controller(round_model, evaluation_env, args.evaluation_episodes)
            finally:
                evaluation_env.close()
            replay_paths = collect_evaluation_replays(replay_dir, previous_replays)
            wandb_logger.log(label, round_index, round_return, champion_parameters, replay_paths, **{"evaluation/round": round_index + 1})
            round_dir = args.output_dir / f"round_{round_index + 1}"
            round_dir.mkdir(parents=True, exist_ok=True)
            round_model.save(round_dir / "controller")
            champion_controller = round_dir / "controller"
            (round_dir / "team.json").write_text(json.dumps(champion_parameters, indent=2) + "\n", encoding="utf-8")
            round_model.save(args.output_dir / "best_controller")
            (args.output_dir / "best_team.json").write_text(json.dumps(champion_parameters, indent=2) + "\n", encoding="utf-8")
            wandb_logger.log_checkpoint(
                f"round-{round_index + 1}-champion",
                (round_dir / "controller.zip", round_dir / "team.json", args.output_dir / "observation_schema.json"),
            )
            wandb_logger.log_final(round_return, champion_parameters)
            opponent = "random" if round_index == 0 else f"round_{round_index}"
            print(f"round={round_index + 1} promoted_return={round_return:.3f} opponent={opponent} team={champion_parameters}")


if __name__ == "__main__":
    main()
