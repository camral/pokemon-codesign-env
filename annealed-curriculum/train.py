"""Train the annealed-reservoir Pokémon co-design baseline."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

EXAMPLES_DIR = Path(__file__).resolve().parents[1] / "examples"
if str(EXAMPLES_DIR) not in sys.path:
    sys.path.insert(0, str(EXAMPLES_DIR))

from utils import (
    WandbEvaluationLogger,
    add_compute_arguments,
    collect_evaluation_replays,
    evaluate_controller,
    evaluation_account_configurations,
    evaluation_replay_dir,
    make_control_env,
    make_control_training_env,
    managed_showdown_cluster,
    server_configuration_for,
    showdown_cluster,
    validate_device,
    wandb_config,
)

from pokemon_codesign_env.configs import AttributeMode, FormatConfig, ItemMode, ObservationConfig, SelectionMode, TeamSearchConfig
from pokemon_codesign_env.envs import RandomPlayerEnv
from pokemon_codesign_env.observations import HumanVisibleBattleEncoder

from curriculum import (
    EmbeddedMaskedDQNPolicy,
    MaskedDQN,
    TeamCandidate,
    TeamEmbeddingExtractor,
    TeamPopulation,
    annealed_value,
    embedding_positions,
    minimum_fitness,
)


def make_search_config(format_config: FormatConfig) -> TeamSearchConfig:
    """Optimize every supported team attribute except the deferred move choice."""
    return TeamSearchConfig(
        format=format_config,
        team_size=6,
        species=SelectionMode.OPTIMIZED,
        evs=SelectionMode.OPTIMIZED,
        ivs=AttributeMode.OPTIMIZED,
        moves=SelectionMode.RANDOM,
        items=ItemMode.OPTIMIZED,
        natures=AttributeMode.OPTIMIZED,
        abilities=AttributeMode.OPTIMIZED,
        happiness=AttributeMode.OPTIMIZED,
        gender=SelectionMode.OPTIMIZED,
        tera_types=AttributeMode.OPTIMIZED if format_config.supports_tera_type else AttributeMode.DEFAULT,
        generations_allowed=tuple(range(1, format_config.generation + 1)),
    )


def assign_candidate_batch(environment, population: TeamPopulation, candidates: list[TeamCandidate], seed: int) -> None:
    """Assign one candidate and trajectory-level move sampler to every vector slot."""
    for index, candidate in enumerate(candidates):
        environment.env_method("set_learner_team", population.trajectory_designer([candidate], seed + index), indices=index)


def assign_reservoir(environment, population: TeamPopulation, reservoir: list[TeamCandidate], seed: int, environments: int) -> None:
    """Let every environment sample teams uniformly from the reservoir per episode."""
    for index in range(environments):
        environment.env_method("set_learner_team", population.trajectory_designer(reservoir, seed + index), indices=index)


def estimate_fitness(model: MaskedDQN, environment, population: TeamPopulation, candidates: list[TeamCandidate], state_samples: int, seed: int) -> np.ndarray:
    """Estimate conservative initial-state Q fitness with current network weights."""
    assign_candidate_batch(environment, population, candidates, seed)
    estimates = np.empty((len(candidates), state_samples), dtype=np.float32)
    for sample_index in range(state_samples):
        observation = environment.reset()
        observation_tensor, _ = model.policy.obs_to_tensor(observation)
        with torch.no_grad():
            q_values = model.q_net(observation_tensor).max(dim=1).values.detach().cpu().numpy()
            estimates[:, sample_index] = q_values[:len(candidates)]
    return minimum_fitness(estimates)


def select_reservoir(candidates: list[TeamCandidate], fitness: np.ndarray, size: int) -> list[TeamCandidate]:
    """Retain the highest-valued complete team designs."""
    indices = np.argsort(fitness)[-size:][::-1]
    return [candidates[int(index)] for index in indices]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--total-steps", type=int, default=1_000_000, help="T: total controller transitions.")
    parser.add_argument("--codesign-fraction", type=float, default=0.5, help="T1 / T; the remainder is fixed-team practice T2.")
    parser.add_argument("--evaluation-batch-size", type=int, default=64, help="X: teams evaluated per curriculum cycle; must equal --num-envs.")
    parser.add_argument("--initial-reservoir-size", type=int, default=16, help="Initial Z, annealed to one during T1.")
    parser.add_argument("--state-samples", type=int, default=8, help="Y: random initial states used for each fitness estimate.")
    parser.add_argument("--embedding-dim", type=int, default=16)
    parser.add_argument("--learning-starts", type=int, default=10_000)
    parser.add_argument("--buffer-size", type=int, default=1_000_000)
    parser.add_argument("--gradient-steps", type=int, default=1)
    parser.add_argument("--evaluation-episodes", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, default=Path("runs/annealed_curriculum"))
    parser.add_argument("--launch-server", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    add_compute_arguments(parser)
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.num_envs != args.evaluation_batch_size:
        raise ValueError("--num-envs must equal --evaluation-batch-size so all X candidates are evaluated in one vector batch")
    if not 0.0 <= args.codesign_fraction <= 1.0:
        raise ValueError("--codesign-fraction must be in [0, 1]")
    if not 1 <= args.initial_reservoir_size <= args.evaluation_batch_size:
        raise ValueError("--initial-reservoir-size must be between 1 and X")
    for name in ("total_steps", "state_samples", "rollout_steps", "gradient_steps", "evaluation_episodes"):
        if getattr(args, name) < 1:
            raise ValueError(f"--{name.replace('_', '-')} must be at least one")
    cycle_steps = args.num_envs * args.rollout_steps
    phase_one_steps = round(args.total_steps * args.codesign_fraction)
    if args.total_steps % cycle_steps:
        raise ValueError(f"--total-steps must be divisible by one curriculum cycle ({cycle_steps} = num-envs x rollout-steps)")
    if phase_one_steps % cycle_steps:
        raise ValueError(f"T1 must be divisible by one curriculum cycle ({cycle_steps}); adjust --total-steps or --codesign-fraction")


def train(args: argparse.Namespace, wandb_logger: WandbEvaluationLogger) -> None:
    format_config = FormatConfig.from_showdown(RandomPlayerEnv.BATTLE_FORMAT)
    search_config = make_search_config(format_config)
    observation_config = ObservationConfig(team_size=6)
    observation_encoder = HumanVisibleBattleEncoder.from_showdown(format_config, observation_config=observation_config)
    population = TeamPopulation(search_config, args.seed)
    candidates = population.sample(args.evaluation_batch_size)
    initial_designers = [population.trajectory_designer([candidate], args.seed + index) for index, candidate in enumerate(candidates)]
    environment = make_control_training_env(
        RandomPlayerEnv, search_config, candidates[0].parameters, observation_encoder, args, args.seed,
        learner_teams=initial_designers,
    )
    species_positions, move_positions = embedding_positions(observation_encoder)
    policy_kwargs = {
        "features_extractor_class": TeamEmbeddingExtractor,
        "features_extractor_kwargs": {
            "species_positions": species_positions,
            "move_positions": move_positions,
            "species_count": len(observation_encoder.species),
            "move_count": len(observation_encoder.moves),
            "embedding_dim": args.embedding_dim,
        },
        "net_arch": [256, 256],
    }
    model = MaskedDQN(
        EmbeddedMaskedDQNPolicy,
        environment,
        policy_kwargs=policy_kwargs,
        learning_rate=3e-4,
        buffer_size=args.buffer_size,
        learning_starts=args.learning_starts,
        train_freq=(args.rollout_steps, "step"),
        gradient_steps=args.gradient_steps,
        gamma=0.99,
        seed=args.seed,
        device=args.device,
        verbose=int(args.verbose),
    )
    phase_one_steps = round(args.total_steps * args.codesign_fraction)
    cycle_steps = args.num_envs * args.rollout_steps
    completed_steps = 0
    fitness = estimate_fitness(model, environment, population, candidates, args.state_samples, args.seed)
    reservoir = select_reservoir(candidates, fitness, args.initial_reservoir_size)
    initial_rollout = True
    practice_initialized = False
    cycle = 0
    wandb_logger.log_system("curriculum_initialized", evaluation_batch_size=len(candidates), reservoir_size=len(reservoir))

    try:
        while completed_steps < args.total_steps:
            progress = completed_steps / max(phase_one_steps, 1)
            phase = "codesign" if completed_steps < phase_one_steps else "practice"
            if phase == "codesign":
                if initial_rollout:
                    mutation_probability = 1.0
                else:
                    reservoir_size = max(1, round(annealed_value(args.initial_reservoir_size, 1, progress)))
                    mutation_probability = annealed_value(1.0, 0.0, progress)
                    elites = reservoir[:min(reservoir_size, len(reservoir))]
                    candidates = elites + population.mutate(reservoir, args.evaluation_batch_size - len(elites), mutation_probability)
                    fitness = estimate_fitness(model, environment, population, candidates, args.state_samples, args.seed + cycle * args.num_envs)
                    reservoir = select_reservoir(candidates, fitness, reservoir_size)
            else:
                if not practice_initialized:
                    fitness = estimate_fitness(model, environment, population, reservoir, args.state_samples, args.seed + cycle * args.num_envs)
                    reservoir = select_reservoir(reservoir, fitness, 1)
                    practice_initialized = True
                    wandb_logger.log_system("practice_initialized", mutation_probability=0.0, reservoir_size=1)
                mutation_probability = 0.0

            assign_reservoir(environment, population, reservoir, args.seed + (cycle + 1) * args.num_envs, args.num_envs)
            model._last_obs = environment.reset()
            steps = min(cycle_steps, args.total_steps - completed_steps)
            model.learn(total_timesteps=steps, reset_num_timesteps=False, progress_bar=args.progress)
            completed_steps += steps
            initial_rollout = False
            training_metrics = {
                "curriculum/cycle": cycle,
                "curriculum/phase": phase,
                "curriculum/progress": min(progress, 1.0),
                "curriculum/mutation_probability": mutation_probability,
                "curriculum/reservoir_size": len(reservoir),
                "curriculum/fitness_min": float(np.min(fitness)),
                "curriculum/fitness_mean": float(np.mean(fitness)),
                "curriculum/fitness_max": float(np.max(fitness)),
            }
            for name, value in model.logger.name_to_value.items():
                if isinstance(value, (int, float, np.integer, np.floating)):
                    training_metrics[f"rl/{name}"] = float(value)
            wandb_logger.log_training("annealed-curriculum", cycle, completed_steps, training_metrics)
            if args.verbose:
                print(f"steps={completed_steps}/{args.total_steps} phase={phase} reservoir={len(reservoir)} mutation={mutation_probability:.4f}", flush=True)
            cycle += 1
        if not practice_initialized:
            fitness = estimate_fitness(model, environment, population, reservoir, args.state_samples, args.seed + cycle * args.num_envs)
            reservoir = select_reservoir(reservoir, fitness, 1)
            practice_initialized = True
            wandb_logger.log_system("champion_selected", mutation_probability=0.0, reservoir_size=1)
    finally:
        environment.close()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    model.save(args.output_dir / "controller")
    (args.output_dir / "champion.json").write_text(json.dumps(reservoir[0].parameters, indent=2) + "\n", encoding="utf-8")
    champion_team = population.trajectory_designer(reservoir, args.seed + cycle * args.num_envs).yield_team()
    (args.output_dir / "champion.team").write_text(champion_team + "\n", encoding="utf-8")
    wandb_logger.log_final(float(np.max(fitness)), reservoir[0].parameters)
    wandb_logger.log_checkpoint(
        "final", (args.output_dir / "controller.zip", args.output_dir / "champion.json", args.output_dir / "champion.team")
    )
    replay_dir = evaluation_replay_dir(args.output_dir, "final")
    previous_replays = set(replay_dir.glob("*.html"))
    evaluation_accounts = evaluation_account_configurations("RandomPlayer", cycle)
    evaluation_env = make_control_env(
        RandomPlayerEnv,
        population.trajectory_designer(reservoir, args.seed + cycle * args.num_envs),
        observation_encoder,
        save_replays=str(replay_dir),
        server_configuration=server_configuration_for(args, args.num_envs),
        challenge_timeout=args.challenge_timeout,
        account_configuration1=evaluation_accounts[0],
        account_configuration2=evaluation_accounts[1],
    )
    wandb_logger.log_system("evaluation_started", evaluation_episodes=args.evaluation_episodes)
    try:
        mean_return = evaluate_controller(model, evaluation_env, args.evaluation_episodes)
    finally:
        evaluation_env.close()
    replay_paths = collect_evaluation_replays(replay_dir, previous_replays)
    wandb_logger.log("final", cycle, mean_return, reservoir[0].parameters, replay_paths, **{"evaluation/timesteps": completed_steps})
    wandb_logger.log_final(mean_return, reservoir[0].parameters)
    wandb_logger.log_system("training_complete", completed_steps=completed_steps, curriculum_cycles=cycle)
    print(f"evaluation_mean_return={mean_return:.3f} champion={reservoir[0].parameters}", flush=True)


def main() -> None:
    args = parse_args()
    validate_args(args)
    validate_device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    format_config = FormatConfig.from_showdown(RandomPlayerEnv.BATTLE_FORMAT)
    wandb_logger = WandbEvaluationLogger(args, wandb_config(args))
    try:
        with managed_showdown_cluster(args.launch_server, format_config.showdown_path, showdown_cluster(args), args.verbose):
            train(args, wandb_logger)
    finally:
        wandb_logger.finish()


if __name__ == "__main__":
    main()
