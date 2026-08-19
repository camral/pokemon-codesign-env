"""Iterative frozen-policy self-play with SB3 PPO and Nevergrad NGOpt.

Round zero bootstraps against RandomPlayer. Every later round trains candidates
against the previous round's frozen champion controller and embodiment, then
promotes the best evaluated challenger.
"""

import argparse
import json
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


class FrozenPolicyPlayer(Player):
    """Use a frozen masked PPO policy as a poke-env opponent."""

    def __init__(self, model: PPO, observation_encoder: HumanVisibleBattleEncoder, **kwargs):
        self.model = model
        self.observation_encoder = observation_encoder
        super().__init__(**kwargs)

    def choose_move(self, battle):
        observation = {
            **self.observation_encoder.encode(battle),
            "action_mask": np.asarray(SinglesEnv.get_action_mask(battle), dtype=np.int8),
        }
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
    champion_model: PPO | None,
    champion_team: RetryingTeamDesigner | None,
):
    """Create a bootstrap or frozen-champion training environment."""
    if champion_model is None:
        return Monitor(
            RandomPlayerEnv(
                team=learner_team,
                observation_encoder=observation_encoder,
            )
        )
    return Monitor(
        FrozenPolicyOpponentEnv(
            team=learner_team,
            opponent_team=champion_team,
            observation_encoder=observation_encoder,
            opponent_kwargs={
                "model": champion_model,
                "observation_encoder": observation_encoder,
            },
        )
    )


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


def evaluate_candidate(
    parameters: dict,
    search_config,
    observation_encoder: HumanVisibleBattleEncoder,
    champion_model: PPO | None,
    champion_parameters: dict | None,
    args: argparse.Namespace,
    seed: int,
) -> tuple[float, PPO | None]:
    """Train a challenger against one frozen opponent snapshot."""
    learner_team = RetryingTeamDesigner(search_config)
    learner_team.set_optimizable_parameter_values(parameters)
    learner_team.rng = np.random.default_rng(seed)
    try:
        learner_team.yield_team()
    except ValueError as error:
        print(f"Rejected embodiment {parameters}: {error}")
        return INVALID_CANDIDATE_LOSS, None

    champion_team = None
    if champion_parameters is not None:
        champion_team = RetryingTeamDesigner(search_config)
        champion_team.set_optimizable_parameter_values(champion_parameters)
        champion_team.rng = np.random.default_rng(seed + 1_000_000)

    env = make_env(
        learner_team,
        observation_encoder,
        champion_model,
        champion_team,
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-play-rounds", type=int, default=3)
    parser.add_argument("--ng-budget", type=int, default=4)
    parser.add_argument("--ppo-timesteps", type=int, default=2048)
    parser.add_argument("--evaluation-episodes", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, default=Path("runs/self_play"))
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--launch-server", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    format_config = FormatConfig.from_showdown(RandomPlayerEnv.BATTLE_FORMAT)
    search_config = make_search_config(format_config)
    observation_encoder = HumanVisibleBattleEncoder.from_showdown(
        format_config,
        observation_config=OBSERVATION_CONFIG,
    )
    (args.output_dir / "observation_schema.json").write_text(
        json.dumps(observation_encoder.get_schema(), indent=2) + "\n",
        encoding="utf-8",
    )

    champion_model = None
    champion_parameters = None
    with managed_showdown_server(
        args.launch_server,
        format_config.showdown_path,
        args.verbose,
    ):
        for round_index in range(args.self_play_rounds):
            parametrization = make_ngopt_parameterization(search_config)
            parametrization.random_state.seed(args.seed + round_index)
            optimizer = ng.optimizers.NGOpt(
                parametrization=parametrization,
                budget=args.ng_budget,
                num_workers=1,
            )
            round_loss = float("inf")
            round_model = None
            round_parameters = None

            for evaluation in range(args.ng_budget):
                candidate = optimizer.ask()
                parameters = candidate.value
                seed = args.seed + round_index * args.ng_budget + evaluation
                loss, model = evaluate_candidate(
                    parameters,
                    search_config,
                    observation_encoder,
                    champion_model,
                    champion_parameters,
                    args,
                    seed,
                )
                optimizer.tell(candidate, loss)
                print(
                    f"round={round_index + 1}/{args.self_play_rounds} "
                    f"candidate={evaluation + 1}/{args.ng_budget} "
                    f"mean_return={-loss:.3f} parameters={parameters}"
                )
                if model is not None and loss < round_loss:
                    round_loss = loss
                    round_model = model
                    round_parameters = parameters

            if round_model is None:
                raise RuntimeError(f"Self-play round {round_index + 1} produced no valid team.")

            champion_model = round_model
            champion_parameters = round_parameters
            round_dir = args.output_dir / f"round_{round_index + 1}"
            round_dir.mkdir(parents=True, exist_ok=True)
            champion_model.save(round_dir / "controller")
            (round_dir / "team.json").write_text(
                json.dumps(champion_parameters, indent=2) + "\n",
                encoding="utf-8",
            )
            champion_model.save(args.output_dir / "best_controller")
            (args.output_dir / "best_team.json").write_text(
                json.dumps(champion_parameters, indent=2) + "\n",
                encoding="utf-8",
            )
            opponent = "random" if round_index == 0 else f"round_{round_index}"
            print(
                f"round={round_index + 1} promoted_return={-round_loss:.3f} "
                f"opponent={opponent} team={champion_parameters}"
            )


if __name__ == "__main__":
    main()
