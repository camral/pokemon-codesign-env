"""Minimal nested co-design experiment using SB3 PPO and Nevergrad NGOpt.

Use --launch-server to manage a local Pokemon Showdown server automatically.
NGOpt chooses the species and nature of all six Pokemon. A fresh PPO controller
is trained for each candidate against a selectable baseline environment whose
opponent team is regenerated every battle.
"""

import argparse
import json
import os
import signal
import socket
import subprocess
import time
from contextlib import contextmanager
from pathlib import Path

import nevergrad as ng
import numpy as np
import torch
from poke_env.ps_client import AccountConfiguration
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from pokemon_codesign_env.configs import (
    AttributeMode,
    FormatConfig,
    ObservationConfig,
    SelectionMode,
    TeamSearchConfig,
)
from pokemon_codesign_env.envs import MaxBasePowerPlayerEnv, RandomPlayerEnv, SimpleHeuristicsPlayerEnv
from pokemon_codesign_env.envs.base import BaselineOpponentEnv
from pokemon_codesign_env.observations import HumanVisibleBattleEncoder
from pokemon_codesign_env.team_designer import TeamDesigner

TEAM_SIZE = 6
OBSERVATION_CONFIG = ObservationConfig(team_size=TEAM_SIZE)
INVALID_CANDIDATE_LOSS = 1_000_000.0
SHOWDOWN_HOST = "127.0.0.1"
SHOWDOWN_PORT = 8000
SERVER_STARTUP_TIMEOUT = 120.0
ENVIRONMENTS = {
    "random": RandomPlayerEnv,
    "max-base-power": MaxBasePowerPlayerEnv,
    "simple-heuristics": SimpleHeuristicsPlayerEnv,
}


def is_server_running(host: str = SHOWDOWN_HOST, port: int = SHOWDOWN_PORT) -> bool:
    """Return whether something is accepting connections at the server address."""
    try:
        with socket.create_connection((host, port), timeout=0.2):
            return True
    except OSError:
        return False


@contextmanager
def managed_showdown_server(enabled: bool, showdown_path: Path, verbose: bool = False):
    """Start and stop a local Showdown server when requested and necessary."""
    if not enabled:
        yield
        return

    if is_server_running():
        if verbose:
            print(f"showdown_server=status_existing port={SHOWDOWN_PORT}")
        yield
        return

    executable = showdown_path / "pokemon-showdown"
    if not executable.is_file():
        raise FileNotFoundError(f"Pokemon Showdown launcher not found: {executable}")

    output = None if verbose else subprocess.DEVNULL
    process = subprocess.Popen(
        ["node", str(executable), "start", "--no-security"],
        cwd=showdown_path,
        stdout=output,
        stderr=output,
        start_new_session=True,
    )
    deadline = time.monotonic() + SERVER_STARTUP_TIMEOUT

    try:
        while not is_server_running():
            return_code = process.poll()
            if return_code is not None:
                raise RuntimeError(f"Pokemon Showdown exited during startup with code {return_code}.")
            if time.monotonic() >= deadline:
                raise TimeoutError(f"Pokemon Showdown did not open port {SHOWDOWN_PORT} within {SERVER_STARTUP_TIMEOUT:g} seconds.")
            time.sleep(0.1)

        if verbose:
            print(f"showdown_server=status_started pid={process.pid} port={SHOWDOWN_PORT}")
        yield
    finally:
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
        if verbose:
            print("showdown_server=status_stopped")


class RetryingTeamDesigner(TeamDesigner):
    """Retry random team details that Showdown rejects."""

    def __init__(self, search_config: TeamSearchConfig, max_attempts: int = 10):
        super().__init__(search_config)
        self.max_attempts = max_attempts

    def yield_team(self) -> str:
        last_error = None
        for _ in range(self.max_attempts):
            try:
                return super().yield_team()
            except ValueError as error:
                last_error = error

        raise ValueError(f"Could not generate a valid team: {last_error}") from last_error


class ObservationFeaturesExtractor(BaseFeaturesExtractor):
    """Concatenate raw state values while keeping the action mask separate."""

    def __init__(self, observation_space):
        feature_keys = tuple(key for key in ("categorical", "continuous", "binary") if key in observation_space.spaces)
        features_dim = sum(int(np.prod(observation_space[key].shape)) for key in feature_keys)
        super().__init__(observation_space, features_dim=features_dim)
        self.feature_keys = feature_keys

    def forward(self, observations):
        return torch.cat([observations[key].float() for key in self.feature_keys], dim=1)


class MaskedActorCriticPolicy(ActorCriticPolicy):
    """Apply poke-env's action mask to PPO's categorical distribution."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args,**kwargs,net_arch=[64, 64],features_extractor_class=ObservationFeaturesExtractor,)
        self._action_mask = None

    def forward(self, observations, deterministic=False):
        self._action_mask = observations["action_mask"]
        return super().forward(observations, deterministic)

    def evaluate_actions(self, observations, actions):
        self._action_mask = observations["action_mask"]
        return super().evaluate_actions(observations, actions)

    def _predict(self, observations, deterministic=False):
        self._action_mask = observations["action_mask"]
        return super()._predict(observations, deterministic)

    def extract_features(self, observations, features_extractor=None):
        """Bypass SB3's automatic MultiDiscrete one-hot conversion."""
        if self.share_features_extractor:
            extractor = features_extractor or self.features_extractor
            return extractor(observations)
        return (self.pi_features_extractor(observations), self.vf_features_extractor(observations))

    def get_distribution(self, observations):
        features = self.pi_features_extractor(observations)
        latent_policy = self.mlp_extractor.forward_actor(features)
        return self._get_action_dist_from_latent(latent_policy)

    def predict_values(self, observations):
        features = self.vf_features_extractor(observations)
        latent_value = self.mlp_extractor.forward_critic(features)
        return self.value_net(latent_value)

    def _get_action_dist_from_latent(self, latent_policy):
        action_logits = self.action_net(latent_policy)
        mask = torch.where(self._action_mask == 1, 0.0, float("-inf"))
        return self.action_dist.proba_distribution(action_logits=action_logits + mask)


def make_search_config(format_config: FormatConfig) -> TeamSearchConfig:
    """Create the optimized learner configuration for the selected format."""
    return TeamSearchConfig(
        format=format_config,
        team_size=TEAM_SIZE,
        species=SelectionMode.OPTIMIZED,
        natures=AttributeMode.OPTIMIZED,
        generations_allowed=tuple(range(1, format_config.generation + 1)),
        types_allowed=("water",) if "monotype" in format_config.format_id else None,
    )


def make_ngopt_parameterization(search_config: TeamSearchConfig):
    """Translate this example's categorical embodiment space to Nevergrad."""
    designer = TeamDesigner(search_config)
    parameter_spaces = designer.get_optimizable_parameter_space()

    return ng.p.Dict(**{
        pokemon: ng.p.Dict(
            species=ng.p.Choice(parameter_space["species"]["values"]),
            nature=ng.p.Choice(parameter_space["nature"]["values"]),
        ) for pokemon, parameter_space in parameter_spaces.items()})


def make_control_env(
    environment_class: type[BaselineOpponentEnv],
    learner_team: TeamDesigner,
    observation_encoder: HumanVisibleBattleEncoder,
    save_replays: bool | str = False,
    account_configuration1: AccountConfiguration | None = None,
    account_configuration2: AccountConfiguration | None = None,
):
    """Create the selected baseline environment for one learner team."""
    return Monitor(environment_class(
        team=learner_team,
        observation_encoder=observation_encoder,
        account_configuration1=account_configuration1,
        account_configuration2=account_configuration2,
        save_replays=save_replays,
    ))


def evaluate_controller(model: PPO, env, episodes: int) -> float:
    """Return the mean shaped episode return using deterministic PPO actions."""
    episode_returns = []

    for _ in range(episodes):
        observation, _ = env.reset()
        terminated = False
        truncated = False
        episode_return = 0.0

        while not terminated and not truncated:
            action, _ = model.predict(observation, deterministic=True)
            observation, reward, terminated, truncated, _ = env.step(action)
            episode_return += float(reward)

        episode_returns.append(episode_return)

    return float(np.mean(episode_returns))


def evaluate_candidate(
    parameter_values: dict,
    environment_class: type[BaselineOpponentEnv],
    learner_config: TeamSearchConfig,
    ppo_timesteps: int,
    evaluation_episodes: int,
    seed: int,
    observation_encoder: HumanVisibleBattleEncoder,
    verbose: bool = False,
) -> tuple[float, PPO | None]:
    """Train a fresh controller and return NGOpt's minimization loss."""
    learner_team = RetryingTeamDesigner(learner_config)
    learner_team.set_optimizable_parameter_values(parameter_values)
    learner_team.rng = np.random.default_rng(seed)

    try:
        learner_team.yield_team()
    except ValueError as error:
        print(f"Rejected embodiment {parameter_values}: {error}")
        return INVALID_CANDIDATE_LOSS, None

    env = make_control_env(
        environment_class,
        learner_team,
        observation_encoder,
    )
    rollout_steps = min(256, max(2, ppo_timesteps))
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
            verbose=int(verbose),
        )
        if verbose:
            print(f"status=training timesteps={ppo_timesteps}")
        model.learn(total_timesteps=ppo_timesteps)
        if verbose:
            print(f"status=evaluating episodes={evaluation_episodes}")
        mean_return = evaluate_controller(model, env, evaluation_episodes)
    finally:
        env.close()

    return -mean_return, model


def watch_saved_controller(
    output_dir: Path,
    games: int,
    turn_delay: float,
    start_delay: float,
    environment_class: type[BaselineOpponentEnv],
    learner_config: TeamSearchConfig,
    observation_encoder: HumanVisibleBattleEncoder,
) -> None:
    """Play slowed exhibition games with the saved best team and controller."""
    controller_path = output_dir / "best_controller"
    team_path = output_dir / "best_team.json"
    schema_path = output_dir / "observation_schema.json"
    if (not controller_path.with_suffix(".zip").exists() or not team_path.exists() or not schema_path.exists()):
        raise FileNotFoundError(f"No compatible saved controller, team, and observation schema found in {output_dir}. Run training first.")
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    if schema != observation_encoder.get_schema():
        raise ValueError("The saved controller uses a different observation schema. Retrain it.")

    learner_team = RetryingTeamDesigner(learner_config)
    learner_team.set_optimizable_parameter_values(json.loads(team_path.read_text(encoding="utf-8")))
    replay_dir = output_dir / "replays"
    env = make_control_env(
        environment_class,
        learner_team,
        observation_encoder,
        save_replays=str(replay_dir),
        account_configuration1=AccountConfiguration("CodesignRL", None),
        account_configuration2=AccountConfiguration("BaselineOpponent", None),
    )
    model = PPO.load(controller_path, env=env, device="cpu")

    try:
        for game in range(1, games + 1):
            observation, _ = env.reset()
            battle_env = env.unwrapped.env
            battle_tag = battle_env.battle1.battle_tag
            print(f"watch_game={game}/{games} url=http://localhost:8000/{battle_tag}")
            time.sleep(start_delay)

            terminated = False
            truncated = False
            episode_return = 0.0
            while not terminated and not truncated:
                action, _ = model.predict(observation, deterministic=True)
                observation, reward, terminated, truncated, _ = env.step(action)
                episode_return += float(reward)
                time.sleep(turn_delay)

            print(f"watch_game={game}/{games} return={episode_return:.3f}")
    finally:
        env.close()

    print(f"replays={replay_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ng-budget", type=int, default=4)
    parser.add_argument("--ppo-timesteps", type=int, default=2048)
    parser.add_argument("--evaluation-episodes", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, default=Path("runs/minimal_codesign"))
    parser.add_argument("--environment", choices=ENVIRONMENTS, default="random", help="Baseline environment and opponent policy.")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--launch-server", action="store_true", help="Start and stop the bundled local Pokemon Showdown server.")
    parser.add_argument("--watch", action="store_true")
    parser.add_argument("--watch-games", type=int, default=3)
    parser.add_argument("--turn-delay", type=float, default=1.0)
    parser.add_argument("--watch-start-delay", type=float, default=5.0)
    return parser.parse_args()


def run_experiment(args: argparse.Namespace, environment_class: type[BaselineOpponentEnv], format_config: FormatConfig) -> None:
    """Run training or watch mode against an available Showdown server."""
    learner_config = make_search_config(format_config)
    observation_encoder = HumanVisibleBattleEncoder.from_showdown(
        format_config,
        observation_config=OBSERVATION_CONFIG,
    )
    if args.watch:
        watch_saved_controller(
            args.output_dir,
            args.watch_games,
            args.turn_delay,
            args.watch_start_delay,
            environment_class,
            learner_config,
            observation_encoder,
        )
        return

    (args.output_dir / "observation_schema.json").write_text(json.dumps(observation_encoder.get_schema(), indent=2) + "\n", encoding="utf-8")

    parametrization = make_ngopt_parameterization(learner_config)
    parametrization.random_state.seed(args.seed)
    optimizer = ng.optimizers.NGOpt(
        parametrization=parametrization,
        budget=args.ng_budget,
        num_workers=1,
    )

    if args.verbose:
        print(
            f"environment={args.environment} format={format_config.format_id} team_size={TEAM_SIZE} "
            f"ng_budget={args.ng_budget} ppo_timesteps={args.ppo_timesteps} "
            f"evaluation_episodes={args.evaluation_episodes} seed={args.seed} "
            f"observation_features={len(observation_encoder.feature_names)}"
        )

    best_loss = float("inf")
    best_parameters = None

    for evaluation in range(args.ng_budget):
        candidate = optimizer.ask()
        parameter_values = candidate.value
        if args.verbose:
            print(f"candidate={evaluation + 1}/{args.ng_budget} status=starting parameters={parameter_values}")
        loss, model = evaluate_candidate(
            parameter_values,
            environment_class,
            learner_config,
            args.ppo_timesteps,
            args.evaluation_episodes,
            args.seed + evaluation,
            observation_encoder,
            args.verbose,
        )
        optimizer.tell(candidate, loss)
        print(f"candidate={evaluation + 1}/{args.ng_budget} mean_return={-loss:.3f} parameters={parameter_values}")

        if model is not None and loss < best_loss:
            best_loss = loss
            best_parameters = parameter_values
            model.save(args.output_dir / "best_controller")
            (args.output_dir / "best_team.json").write_text(json.dumps(best_parameters, indent=2) + "\n")

    recommendation = optimizer.provide_recommendation().value
    print(f"best_evaluated_return={-best_loss:.3f} best_team={best_parameters}")
    print(f"ngopt_recommendation={recommendation}")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    environment_class = ENVIRONMENTS[args.environment]
    format_config = FormatConfig.from_showdown(environment_class.BATTLE_FORMAT)

    with managed_showdown_server(args.launch_server, format_config.showdown_path, args.verbose):
        run_experiment(args, environment_class, format_config)


if __name__ == "__main__":
    main()
