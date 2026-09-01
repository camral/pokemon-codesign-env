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

from utils import add_compute_arguments, make_control_training_env, managed_showdown_cluster, server_configuration_for, showdown_cluster, validate_device

from pokemon_codesign_env.configs import AttributeMode, FormatConfig, ItemMode, ObservationConfig, SelectionMode, TeamSearchConfig
from pokemon_codesign_env.envs import RandomPlayerEnv
from pokemon_codesign_env.observations import HumanVisibleBattleEncoder

from curriculum import (
    EmbeddedMaskedDQNPolicy,
    MaskedDQN,
    TeamCandidate,
    TeamEmbeddingExtractor,
    TeamPopulation,
    aggregate_fitness,
    annealed_value,
    embedding_positions,
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


def assign_teams(environment, teams: list[TeamCandidate]) -> None:
    """Assign one candidate to every vector slot for its next trajectory."""
    for index, candidate in enumerate(teams):
        environment.env_method("set_learner_team", candidate.packed_team, indices=index)


def estimate_fitness(model: MaskedDQN, environment, candidates: list[TeamCandidate], state_samples: int, aggregation: str) -> np.ndarray:
    """Estimate conservative initial-state Q fitness with current network weights."""
    assign_teams(environment, candidates)
    estimates = np.empty((len(candidates), state_samples), dtype=np.float32)
    for sample_index in range(state_samples):
        observation = environment.reset()
        observation_tensor, _ = model.policy.obs_to_tensor(observation)
        with torch.no_grad():
            estimates[:, sample_index] = model.q_net(observation_tensor).max(dim=1).values.detach().cpu().numpy()
    return aggregate_fitness(estimates, aggregation)


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
    parser.add_argument("--fitness", choices=("minimum", "percentile10", "bottom-quartile"), default="minimum")
    parser.add_argument("--embedding-dim", type=int, default=16)
    parser.add_argument("--learning-starts", type=int, default=10_000)
    parser.add_argument("--buffer-size", type=int, default=1_000_000)
    parser.add_argument("--gradient-steps", type=int, default=1)
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
    for name in ("total_steps", "state_samples", "rollout_steps", "gradient_steps"):
        if getattr(args, name) < 1:
            raise ValueError(f"--{name.replace('_', '-')} must be at least one")


def train(args: argparse.Namespace) -> None:
    format_config = FormatConfig.from_showdown(RandomPlayerEnv.BATTLE_FORMAT)
    search_config = make_search_config(format_config)
    observation_config = ObservationConfig(team_size=6)
    observation_encoder = HumanVisibleBattleEncoder.from_showdown(format_config, observation_config=observation_config)
    population = TeamPopulation(search_config, args.seed)
    candidates = population.sample(args.evaluation_batch_size)
    environment = make_control_training_env(
        RandomPlayerEnv, search_config, candidates[0].parameters, observation_encoder, args, args.seed,
        learner_teams=[candidate.packed_team for candidate in candidates],
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
    phase_one_steps = int(args.total_steps * args.codesign_fraction)
    cycle_steps = args.num_envs * args.rollout_steps
    completed_steps = 0
    fitness = estimate_fitness(model, environment, candidates, args.state_samples, args.fitness)
    reservoir = select_reservoir(candidates, fitness, args.initial_reservoir_size)
    initial_rollout = True

    try:
        while completed_steps < args.total_steps:
            progress = completed_steps / max(phase_one_steps, 1)
            if completed_steps < phase_one_steps and not initial_rollout:
                reservoir_size = max(1, round(annealed_value(args.initial_reservoir_size, 1, progress)))
                mutation_probability = annealed_value(1.0, 0.0, progress)
                elites = reservoir[:min(reservoir_size, len(reservoir))]
                candidates = elites + population.mutate(reservoir, args.evaluation_batch_size - len(elites), mutation_probability)
                fitness = estimate_fitness(model, environment, candidates, args.state_samples, args.fitness)
                reservoir = select_reservoir(candidates, fitness, reservoir_size)
            elif initial_rollout:
                mutation_probability = 1.0
            else:
                reservoir = reservoir[:1]
                mutation_probability = 0.0

            rollout_teams = [reservoir[int(population.rng.integers(len(reservoir)))] for _ in range(args.num_envs)]
            assign_teams(environment, rollout_teams)
            model._last_obs = environment.reset()
            steps = min(cycle_steps, args.total_steps - completed_steps)
            model.learn(total_timesteps=steps, reset_num_timesteps=False, progress_bar=args.progress)
            completed_steps += steps
            initial_rollout = False
            if args.verbose:
                phase = "codesign" if completed_steps < phase_one_steps else "practice"
                print(f"steps={completed_steps}/{args.total_steps} phase={phase} reservoir={len(reservoir)} mutation={mutation_probability:.4f}", flush=True)
    finally:
        environment.close()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    model.save(args.output_dir / "controller")
    (args.output_dir / "champion.json").write_text(json.dumps(reservoir[0].parameters, indent=2) + "\n", encoding="utf-8")
    (args.output_dir / "champion.team").write_text(reservoir[0].packed_team + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    validate_args(args)
    validate_device(args.device)
    format_config = FormatConfig.from_showdown(RandomPlayerEnv.BATTLE_FORMAT)
    with managed_showdown_cluster(args.launch_server, format_config.showdown_path, showdown_cluster(args), args.verbose):
        train(args)


if __name__ == "__main__":
    main()
