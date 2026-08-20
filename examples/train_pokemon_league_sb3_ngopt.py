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

if __package__:
    from examples.train_sb3_ngopt import (
        INVALID_CANDIDATE_LOSS,
        MaskedActorCriticPolicy,
        RetryingTeamDesigner,
        make_ngopt_parameterization,
        make_search_config,
        managed_showdown_server,
    )
else:
    from train_sb3_ngopt import (
        INVALID_CANDIDATE_LOSS,
        MaskedActorCriticPolicy,
        RetryingTeamDesigner,
        make_ngopt_parameterization,
        make_search_config,
        managed_showdown_server,
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
    seed: int,
) -> tuple[float, PPO | None]:
    """Train one candidate controller and return NGOpt's minimization loss."""
    team = RetryingTeamDesigner(search_config)
    team.set_optimizable_parameter_values(parameter_values)
    team.rng = np.random.default_rng(seed)
    try:
        team.yield_team()
    except ValueError as error:
        print(f"Rejected embodiment {parameter_values}: {error}")
        return INVALID_CANDIDATE_LOSS, None

    env = make_league_env(
        team,
        observation_encoder,
        None if args.random_league else args.generation,
        args.random_league,
        args.difficulty,
    )
    rollout_steps = min(256, max(2, args.ppo_timesteps))
    batch_size = 64 if rollout_steps % 64 == 0 else rollout_steps
    try:
        model = PPO(
            MaskedActorCriticPolicy,
            env,
            n_steps=rollout_steps,
            batch_size=batch_size,
            learning_rate=3e-4,
            gamma=0.99,
            ent_coef=0.01,
            seed=seed,
            device="cpu",
            verbose=int(args.verbose),
        )
        model.learn(total_timesteps=args.ppo_timesteps)
        mean_return = evaluate_controller(model, env, args.evaluation_episodes)
    finally:
        env.close()
    return -mean_return, model


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
        raise FileNotFoundError(
            f"No compatible saved controller, team, and observation schema found in "
            f"{output_dir}. Run training first."
        )
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
    model = PPO.load(controller_path, env=env, device="cpu")

    def print_battle(game: int) -> None:
        league_env = env.unwrapped
        battle_tag = league_env._battle_env.env.battle1.battle_tag
        print(
            f"watch_game={game}/{games} trainer={league_env.current_trainer.name} "
            f"url=http://localhost:8000/{battle_tag}"
        )
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
    args.output_dir.mkdir(parents=True, exist_ok=True)
    format_id = (
        PokemonLeagueEnv.RANDOM_BATTLE_FORMAT
        if args.random_league
        else get_pokemon_league(args.generation).battle_format
    )
    format_config = FormatConfig.from_showdown(format_id)
    search_config = make_search_config(format_config)
    observation_encoder = HumanVisibleBattleEncoder.from_showdown(
        format_config,
        observation_config=OBSERVATION_CONFIG,
    )

    with managed_showdown_server(
        args.launch_server,
        format_config.showdown_path,
        args.verbose,
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
            )
            return

        schema_path = args.output_dir / "observation_schema.json"
        schema_path.write_text(
            json.dumps(observation_encoder.get_schema(), indent=2) + "\n",
            encoding="utf-8",
        )
        parametrization = make_ngopt_parameterization(search_config)
        parametrization.random_state.seed(args.seed)
        optimizer = ng.optimizers.NGOpt(
            parametrization=parametrization,
            budget=args.ng_budget,
            num_workers=1,
        )

        best_loss = float("inf")
        best_parameters = None
        for evaluation in range(args.ng_budget):
            candidate = optimizer.ask()
            parameters = candidate.value
            loss, model = evaluate_candidate(
                parameters,
                search_config,
                observation_encoder,
                args,
                args.seed + evaluation,
            )
            optimizer.tell(candidate, loss)
            print(
                f"candidate={evaluation + 1}/{args.ng_budget} "
                f"mean_return={-loss:.3f} parameters={parameters}"
            )
            if model is not None and loss < best_loss:
                best_loss = loss
                best_parameters = parameters
                model.save(args.output_dir / "best_controller")
                (args.output_dir / "best_team.json").write_text(
                    json.dumps(parameters, indent=2) + "\n",
                    encoding="utf-8",
                )

        print(f"best_evaluated_return={-best_loss:.3f} best_team={best_parameters}")
        print(f"ngopt_recommendation={optimizer.provide_recommendation().value}")


if __name__ == "__main__":
    main()
