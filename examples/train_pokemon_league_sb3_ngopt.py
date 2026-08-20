"""Train SB3 PPO control and NGOpt embodiment on PokemonLeagueEnv.

NGOpt selects the species and nature of a six-Pokemon team. A fresh masked PPO
controller is trained for every candidate. Use --random-league to sample one of
the nine Leagues per episode, or --generation for a fixed League.
"""

import argparse
import json
import time
from pathlib import Path

import nevergrad as ng
import numpy as np
from poke_env.ps_client import AccountConfiguration
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor

from pokemon_codesign_env.configs import FormatConfig, ObservationConfig
from pokemon_codesign_env.envs import PokemonLeagueEnv, get_pokemon_league
from pokemon_codesign_env.observations import HumanVisibleBattleEncoder
from pokemon_codesign_env.team_pool import RandomTeamPool

if __package__:
    from examples.train_sb3_ngopt import (
        INVALID_CANDIDATE_LOSS,
        MaskedActorCriticPolicy,
        RetryingTeamDesigner,
        WandbEvaluationLogger,
        add_compute_arguments,
        collect_evaluation_replays,
        evaluation_account_configurations,
        evaluation_replay_dir,
        evaluate_vector_controller,
        make_ngopt_parameterization,
        make_search_config,
        make_team,
        make_training_callback,
        make_vector_env,
        managed_showdown_cluster,
        run_ngopt,
        server_configuration_for,
        showdown_cluster,
        validate_compute_arguments,
        validate_device,
        wandb_config,
    )
else:
    from train_sb3_ngopt import (
        INVALID_CANDIDATE_LOSS,
        MaskedActorCriticPolicy,
        RetryingTeamDesigner,
        WandbEvaluationLogger,
        add_compute_arguments,
        collect_evaluation_replays,
        evaluation_account_configurations,
        evaluation_replay_dir,
        evaluate_vector_controller,
        make_ngopt_parameterization,
        make_search_config,
        make_team,
        make_training_callback,
        make_vector_env,
        managed_showdown_cluster,
        run_ngopt,
        server_configuration_for,
        showdown_cluster,
        validate_compute_arguments,
        validate_device,
        wandb_config,
    )

TEAM_SIZE = 6
OBSERVATION_CONFIG = ObservationConfig(team_size=TEAM_SIZE)


def make_league_env(
    team: RetryingTeamDesigner,
    observation_encoder: HumanVisibleBattleEncoder,
    generation: int | None,
    random_league: bool,
    difficulty: int,
    **env_kwargs,
):
    """Create one monitored fixed- or random-League environment."""
    return Monitor(
        PokemonLeagueEnv(
            team=team,
            generation=generation,
            random_league=random_league,
            difficulty=difficulty,
            observation_encoder=observation_encoder,
            **env_kwargs,
        )
    )


def make_league_training_env(
    search_config,
    parameter_values: dict,
    observation_encoder: HumanVisibleBattleEncoder,
    args: argparse.Namespace,
    seed: int,
    learner_team_pool: RandomTeamPool | None = None,
    num_envs: int | None = None,
    vec_env: str | None = None,
    save_replays: bool | str = False,
    account_configuration1: AccountConfiguration | None = None,
    account_configuration2: AccountConfiguration | None = None,
):
    """Create independently seeded League workers for PPO data collection."""
    num_envs = num_envs or args.num_envs
    vec_env = vec_env or args.vec_env

    def make_env(rank: int):
        def initialize():
            return PokemonLeagueEnv(
                team=make_team(search_config, parameter_values, seed + rank) if learner_team_pool is None else learner_team_pool.builder(seed + rank),
                generation=None if args.random_league else args.generation,
                random_league=args.random_league,
                difficulty=args.difficulty,
                observation_encoder=observation_encoder,
                server_configuration=server_configuration_for(args, rank),
                challenge_timeout=args.challenge_timeout,
                save_replays=save_replays,
                account_configuration1=account_configuration1,
                account_configuration2=account_configuration2,
            )

        return initialize

    return make_vector_env([make_env(rank) for rank in range(num_envs)], vec_env, args.envs_per_worker, args.reset_retries)


def evaluate_controller(model: PPO, env, episodes: int) -> float:
    """Return the mean full-League episode return."""
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


def evaluate_candidate(
    parameter_values: dict,
    search_config,
    observation_encoder: HumanVisibleBattleEncoder,
    args: argparse.Namespace,
    evaluation: int,
    seed: int,
    wandb_logger: WandbEvaluationLogger,
) -> tuple[float, PPO | None, tuple[Path, ...]]:
    """Train one candidate controller and return NGOpt's minimization loss."""
    team = make_team(search_config, parameter_values, seed)
    try:
        team.yield_team()
    except ValueError as error:
        print(f"Rejected embodiment {parameter_values}: {error}")
        return INVALID_CANDIDATE_LOSS, None, ()

    learner_pool_size = args.learner_team_pool_size or (max(args.num_envs, 1_024) if args.vec_env == "sharded" else 0)
    learner_team_pool = None
    if learner_pool_size:
        wandb_logger.log_system("learner_team_pool_generating", label=f"candidate-{evaluation + 1}", team_pool_size=learner_pool_size)
        pool_started_at = time.monotonic()
        team.validate_teams = False
        learner_team_pool = RandomTeamPool.from_generator(
            search_config.format.showdown_path, search_config.format.format_id, learner_pool_size, team.yield_team
        )
        wandb_logger.log_system(
            "learner_team_pool_ready", label=f"candidate-{evaluation + 1}", team_pool_size=learner_pool_size,
            team_pool_startup_s=time.monotonic() - pool_started_at,
        )

    wandb_logger.log_system("environments_starting", label=f"candidate-{evaluation + 1}", candidate=evaluation, ready_environments=0)
    environments_started_at = time.monotonic()
    training_env = make_league_training_env(search_config, parameter_values, observation_encoder, args, seed, learner_team_pool)
    wandb_logger.log_system(
        "environments_ready", label=f"candidate-{evaluation + 1}", candidate=evaluation, ready_environments=args.num_envs,
        environment_startup_s=time.monotonic() - environments_started_at,
    )
    replay_dir = evaluation_replay_dir(args.output_dir, f"candidate_{evaluation + 1}")
    previous_replays = set(replay_dir.glob("*.html"))
    evaluation_accounts = evaluation_account_configurations("LeagueOpponent", evaluation, args.ng_workers > 1)
    if args.ng_workers > 1:
        evaluation_env = make_league_training_env(
            search_config, parameter_values, observation_encoder, args, seed + args.num_envs, num_envs=1, vec_env="subproc", save_replays=str(replay_dir),
            account_configuration1=evaluation_accounts[0], account_configuration2=evaluation_accounts[1],
        )
    else:
        evaluation_env = make_league_env(
            make_team(search_config, parameter_values, seed + args.num_envs), observation_encoder,
            None if args.random_league else args.generation, args.random_league, args.difficulty,
            server_configuration=server_configuration_for(args, args.num_envs), challenge_timeout=args.challenge_timeout, save_replays=str(replay_dir),
            account_configuration1=evaluation_accounts[0], account_configuration2=evaluation_accounts[1],
        )
    try:
        model = PPO(
            MaskedActorCriticPolicy,
            training_env,
            n_steps=args.rollout_steps,
            batch_size=args.batch_size,
            learning_rate=3e-4,
            gamma=0.99,
            ent_coef=0.01,
            seed=seed,
            device=args.device,
            verbose=int(args.verbose),
        )
        model.learn(total_timesteps=args.ppo_timesteps, callback=make_training_callback(args, wandb_logger, f"candidate-{evaluation + 1}", evaluation))
        if args.ng_workers > 1:
            mean_return = evaluate_vector_controller(model, evaluation_env, args.evaluation_episodes)
        else:
            mean_return = evaluate_controller(model, evaluation_env, args.evaluation_episodes)
    finally:
        training_env.close()
        evaluation_env.close()
    return -mean_return, model, collect_evaluation_replays(replay_dir, previous_replays)


def watch_saved_controller(
    output_dir: Path,
    games: int,
    turn_delay: float,
    start_delay: float,
    team_config,
    observation_encoder: HumanVisibleBattleEncoder,
    generation: int | None,
    random_league: bool,
    difficulty: int,
    device: str,
) -> None:
    """Play slowed League runs with the saved best team and controller."""
    controller_path = output_dir / "best_controller"
    team_path = output_dir / "best_team.json"
    schema_path = output_dir / "observation_schema.json"
    if (
        not controller_path.with_suffix(".zip").exists()
        or not team_path.exists()
        or not schema_path.exists()
    ):
        raise FileNotFoundError(f"No compatible saved controller, team, and observation schema found in {output_dir}. Run training first.")
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    if schema != observation_encoder.get_schema():
        raise ValueError("The saved controller uses a different observation schema. Retrain it.")

    team = RetryingTeamDesigner(team_config)
    team.set_optimizable_parameter_values(json.loads(team_path.read_text(encoding="utf-8")))
    replay_dir = output_dir / "replays"
    env = make_league_env(
        team,
        observation_encoder,
        generation,
        random_league,
        difficulty,
        save_replays=str(replay_dir),
        account_configuration1=AccountConfiguration("CodesignRL", None),
        account_configuration2=AccountConfiguration("LeagueOpponent", None),
    )
    model = PPO.load(controller_path, env=env, device=device)

    def print_battle(game: int) -> None:
        league_env = env.unwrapped
        battle_tag = league_env._battle_env.env.battle1.battle_tag
        print(f"watch_game={game}/{games} trainer={league_env.current_trainer.name} url=http://localhost:8000/{battle_tag}")
        time.sleep(start_delay)

    try:
        for game in range(1, games + 1):
            observation, _ = env.reset()
            print_battle(game)
            trainer_index = env.unwrapped.current_trainer_index
            terminated = truncated = False
            episode_return = 0.0

            while not terminated and not truncated:
                action, _ = model.predict(observation, deterministic=True)
                observation, reward, terminated, truncated, _ = env.step(action)
                episode_return += float(reward)
                trainer_changed = env.unwrapped.current_trainer_index != trainer_index
                if not terminated and not truncated and trainer_changed:
                    trainer_index = env.unwrapped.current_trainer_index
                    print_battle(game)
                time.sleep(turn_delay)

            print(f"watch_game={game}/{games} return={episode_return:.3f}")
    finally:
        env.close()

    print(f"replays={replay_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    league = parser.add_mutually_exclusive_group()
    league.add_argument("--generation", type=int, choices=range(1, 10))
    league.add_argument("--random-league", action="store_true")
    parser.add_argument("--difficulty", type=int, choices=(0, 1), default=0)
    parser.add_argument("--ng-budget", type=int, default=4)
    parser.add_argument("--ppo-timesteps", type=int, default=2048)
    parser.add_argument("--evaluation-episodes", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    add_compute_arguments(parser)
    parser.add_argument("--output-dir", type=Path, default=Path("runs/pokemon_league"))
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--launch-server", action="store_true")
    parser.add_argument("--watch", action="store_true")
    parser.add_argument("--watch-games", type=int, default=1)
    parser.add_argument("--turn-delay", type=float, default=1.0)
    parser.add_argument("--watch-start-delay", type=float, default=5.0)
    args = parser.parse_args()
    if not args.random_league and args.generation is None:
        args.generation = 4
    return args


def main() -> None:
    args = parse_args()
    validate_compute_arguments(args)
    validate_device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    format_id = (
        PokemonLeagueEnv.RANDOM_BATTLE_FORMAT
        if args.random_league
        else get_pokemon_league(args.generation).battle_format
    )
    format_config = FormatConfig.from_showdown(format_id)
    search_config = make_search_config(format_config)
    observation_encoder = HumanVisibleBattleEncoder.from_showdown(format_config, observation_config=OBSERVATION_CONFIG)

    with (
        WandbEvaluationLogger(args, wandb_config(args), enabled=not args.watch) as wandb_logger,
        managed_showdown_cluster(args.launch_server, format_config.showdown_path, showdown_cluster(args), args.verbose),
    ):
        if args.watch:
            watch_saved_controller(
                args.output_dir,
                args.watch_games,
                args.turn_delay,
                args.watch_start_delay,
                search_config,
                observation_encoder,
                None if args.random_league else args.generation,
                args.random_league,
                args.difficulty,
                args.device,
            )
            return

        schema_path = args.output_dir / "observation_schema.json"
        schema_path.write_text(json.dumps(observation_encoder.get_schema(), indent=2) + "\n", encoding="utf-8")
        parametrization = make_ngopt_parameterization(search_config)
        parametrization.random_state.seed(args.seed)
        optimizer = ng.optimizers.NGOpt(parametrization=parametrization, budget=args.ng_budget, num_workers=args.ng_workers)

        best_loss = float("inf")
        best_parameters = None

        def evaluate(evaluation, parameters):
            return evaluate_candidate(parameters, search_config, observation_encoder, args, evaluation, args.seed + evaluation, wandb_logger)

        for evaluation, parameters, loss, model, replay_paths, duration in run_ngopt(optimizer, args.ng_budget, args.ng_workers, evaluate):
            print(f"candidate={evaluation + 1}/{args.ng_budget} mean_return={-loss:.3f} parameters={parameters}")
            wandb_logger.log(f"candidate-{evaluation + 1}", evaluation, -loss, parameters, replay_paths)
            if model is not None and loss < best_loss:
                best_loss = loss
                best_parameters = parameters
                model.save(args.output_dir / "best_controller")
                (args.output_dir / "best_team.json").write_text(json.dumps(parameters, indent=2) + "\n", encoding="utf-8")
                wandb_logger.log_checkpoint(
                    f"candidate-{evaluation + 1}",
                    (args.output_dir / "best_controller.zip", args.output_dir / "best_team.json", args.output_dir / "observation_schema.json"),
                )
            recommendation = optimizer.provide_recommendation().value
            wandb_logger.log_optimization(f"candidate-{evaluation + 1}", evaluation, -loss, -best_loss, parameters, recommendation, duration)

        recommendation = optimizer.provide_recommendation().value
        wandb_logger.log_final(-best_loss, recommendation)
        print(f"best_evaluated_return={-best_loss:.3f} best_team={best_parameters}")
        print(f"ngopt_recommendation={recommendation}")


if __name__ == "__main__":
    main()
