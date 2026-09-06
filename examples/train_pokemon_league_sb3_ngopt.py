"""Train SB3 PPO control and NGOpt embodiment on PokemonLeagueEnv.

NGOpt selects the species and nature of a six-Pokemon team. Each PPO rollout
uses one independently sampled team per environment, then replaces the
population while one controller keeps learning. Use --random-league to sample
one of the nine Leagues per episode, or --generation for a fixed League.
"""

import argparse
import json
import time
from pathlib import Path

import nevergrad as ng
from poke_env.ps_client import AccountConfiguration
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor

import pokemon_codesign_env as pce
from pokemon_codesign_env.spaces import TeamDesignSpace
from stable_baselines3.common.vec_env import VecMonitor
from pokemon_codesign_env.configs import FormatConfig
from pokemon_codesign_env.envs import PokemonLeagueEnv, get_pokemon_league
from pokemon_codesign_env.observations import HumanVisibleBattleEncoder
from pokemon_codesign_env.team_pool import RandomTeamPool

if __package__:
    from examples.utils import (
        MaskedActorCriticPolicy,
        PeriodicEvaluationCallback,
        OBSERVATION_CONFIG,
        RetryingTeamDesigner,
        RolloutTeamOptimizerCallback,
        WandbEvaluationLogger,
        TEAM_SIZE,
        add_compute_arguments,
        begin_startup_stage,
        collect_evaluation_replays,
        evaluation_account_configurations,
        evaluation_replay_dir,
        make_ngopt_parameterization,
        make_search_config,
        make_training_callback,
        rollout_population_evaluations,
        SB3VecEnvAdapter,
        evaluate_recommendation,
        evaluate_vector_controller,
        sample_team_population,
        startup_stage,
        validate_compute_arguments,
        validate_device,
        wandb_config,
    )
else:
    from utils import (
        MaskedActorCriticPolicy,
        PeriodicEvaluationCallback,
        OBSERVATION_CONFIG,
        RetryingTeamDesigner,
        RolloutTeamOptimizerCallback,
        WandbEvaluationLogger,
        TEAM_SIZE,
        add_compute_arguments,
        begin_startup_stage,
        collect_evaluation_replays,
        evaluation_account_configurations,
        evaluation_replay_dir,
        make_ngopt_parameterization,
        make_search_config,
        make_training_callback,
        rollout_population_evaluations,
        SB3VecEnvAdapter,
        evaluate_recommendation,
        evaluate_vector_controller,
        sample_team_population,
        startup_stage,
        validate_compute_arguments,
        validate_device,
        wandb_config,
    )

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
    args: argparse.Namespace,
    num_envs: int | None = None,
    save_replays: bool | str = False,
    account_configuration1: AccountConfiguration | None = None,
    account_configuration2: AccountConfiguration | None = None,
):
    """Create the League workers PPO collects from, and one base design per environment.

    The environments launch and own the Showdown servers they battle on.
    """
    num_envs = num_envs or args.num_envs
    vector_env, base_embodiments = pce.make_vec(
        opponent="league",
        num_envs=num_envs,
        format_config=search_config.format,
        embodiment_config=search_config,
        generation=None if args.random_league else args.generation,
        random_league=args.random_league,
        difficulty=args.difficulty,
        shards=args.showdown_shards or None,
        envs_per_worker=args.envs_per_worker,
        reset_retries=args.reset_retries,
        step_timeout=args.step_timeout,
        step_timeout_reward=args.step_timeout_reward,
        challenge_timeout=args.challenge_timeout,
        save_replays=save_replays,
        account_configuration1=account_configuration1,
        account_configuration2=account_configuration2,
    )
    return VecMonitor(SB3VecEnvAdapter(vector_env)), base_embodiments


def evaluate_league_recommendation(model, search_config, args, embodiment, replay_dir, accounts, episodes: int):
    """Score one design over several League episodes in an environment reused between them."""
    evaluation_env, _base = make_league_training_env(
        search_config, args, num_envs=1, save_replays=str(replay_dir),
        account_configuration1=accounts[0], account_configuration2=accounts[1],
    )
    try:
        evaluation_env.set_embodiment(embodiment)
        return evaluate_vector_controller(model, evaluation_env, episodes), 0
    finally:
        evaluation_env.close()


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
    parser.add_argument("--ppo-timesteps", type=int, default=2048)
    parser.add_argument("--evaluation-episodes", type=int, default=4)
    parser.add_argument(
        "--evaluation-interval-rollouts", type=int, default=0,
        help="Evaluate after every N completed PPO rollouts; zero disables periodic evaluation.",
    )
    parser.add_argument("--seed", type=int, default=0)
    add_compute_arguments(parser)
    parser.add_argument("--output-dir", type=Path, default=Path("runs/pokemon_league"))
    parser.add_argument("--verbose", action="store_true")
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
        rollout_size = args.num_envs * args.rollout_steps
        rollout_count = (args.ppo_timesteps + rollout_size - 1) // rollout_size
        population_evaluations = rollout_population_evaluations(args.ppo_timesteps, args.num_envs, args.rollout_steps)
        embodiment_space = TeamDesignSpace(search_config, seed=args.seed)
        with startup_stage(args, wandb_logger, "learner_population", teams=args.num_envs):
            parametrization = make_ngopt_parameterization(embodiment_space)
            parametrization.random_state.seed(args.seed)
            optimizer = ng.optimizers.NGOpt(parametrization=parametrization, budget=population_evaluations, num_workers=args.num_envs)
            candidates, parameters = sample_team_population(optimizer, embodiment_space, args.num_envs)
        with startup_stage(args, wandb_logger, "environment_workers", environments=args.num_envs):
            training_env, _base_embodiments = make_league_training_env(search_config, args)
            training_env.set_embodiment(parameters)
        team_optimizer_callback = RolloutTeamOptimizerCallback(
            optimizer, search_config, candidates, parameters, args.seed, args.ppo_timesteps, wandb_logger, "league-codesign"
        )
        model = PPO(
            MaskedActorCriticPolicy, training_env, n_steps=args.rollout_steps, batch_size=args.batch_size, learning_rate=3e-4,
            gamma=0.99, ent_coef=0.01, seed=args.seed, device=args.device, verbose=int(args.verbose),
        )

        def evaluate_periodically(current_model: PPO, completed_rollouts: int, timesteps: int) -> None:
            recommendation = embodiment_space.from_index(optimizer.provide_recommendation().value)
            label = f"rollout-{completed_rollouts}"
            replay_dir = evaluation_replay_dir(args.output_dir, label)
            previous_replays = set(replay_dir.glob("*.html"))
            evaluation_accounts = evaluation_account_configurations("LeagueOpponent", completed_rollouts)
            evaluation_seed = args.seed + population_evaluations + completed_rollouts
            if args.verbose:
                print(f"status=evaluating rollout={completed_rollouts} timesteps={timesteps} episodes={args.evaluation_episodes}", flush=True)
            mean_return, timed_out_episodes = evaluate_league_recommendation(
                current_model, search_config, args, recommendation, replay_dir, evaluation_accounts, args.evaluation_episodes
            )
            replay_paths = collect_evaluation_replays(replay_dir, previous_replays)
            evaluation_metrics = {
                "evaluation/timesteps": timesteps, "evaluation/rollout": completed_rollouts,
                "evaluation/timed_out_episodes": timed_out_episodes,
            }
            wandb_logger.log(label, completed_rollouts, mean_return, recommendation, replay_paths, **evaluation_metrics)
            print(f"evaluation_rollout={completed_rollouts} timesteps={timesteps} mean_return={mean_return:.3f}", flush=True)

        periodic_evaluation_callback = None
        if args.evaluation_interval_rollouts:
            periodic_evaluation_callback = PeriodicEvaluationCallback(args.evaluation_interval_rollouts, rollout_count, evaluate_periodically)
        try:
            initial_reset_started_at = begin_startup_stage(args, wandb_logger, "initial_reset", environments=args.num_envs)
            callback = make_training_callback(
                args, wandb_logger, "league-codesign", 0, team_optimizer_callback=team_optimizer_callback,
                periodic_evaluation_callback=periodic_evaluation_callback, initial_reset_started_at=initial_reset_started_at,
            )
            model.learn(total_timesteps=args.ppo_timesteps, callback=callback)
        finally:
            training_env.close()

        recommendation = embodiment_space.from_index(optimizer.provide_recommendation().value)
        replay_dir = evaluation_replay_dir(args.output_dir, "final")
        previous_replays = set(replay_dir.glob("*.html"))
        evaluation_accounts = evaluation_account_configurations("LeagueOpponent", 0)
        mean_return, timed_out_episodes = evaluate_league_recommendation(
            model, search_config, args, recommendation, replay_dir, evaluation_accounts, args.evaluation_episodes
        )
        replay_paths = collect_evaluation_replays(replay_dir, previous_replays)
        model.save(args.output_dir / "best_controller")
        (args.output_dir / "best_team.json").write_text(json.dumps(recommendation, indent=2) + "\n", encoding="utf-8")
        wandb_logger.log("final", rollout_count - 1, mean_return, recommendation, replay_paths, **{"evaluation/timed_out_episodes": timed_out_episodes})
        wandb_logger.log_final(mean_return, recommendation)
        wandb_logger.log_checkpoint(
            "final", (args.output_dir / "best_controller.zip", args.output_dir / "best_team.json", args.output_dir / "observation_schema.json")
        )
        print(f"evaluated_return={mean_return:.3f} best_team={recommendation}")


if __name__ == "__main__":
    main()
