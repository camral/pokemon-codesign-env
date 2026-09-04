"""Generate native Pokemon Zarr trajectories for Transformer Transformer."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO

from pokemon_codesign_env.baseline_adapters import (
    TeamDesignVectorAdapter,
    make_all_attribute_search_config,
)
from pokemon_codesign_env.configs import FormatConfig, ObservationConfig
from pokemon_codesign_env.fasttd3_factory import ENVIRONMENTS
from pokemon_codesign_env.observations import HumanVisibleBattleEncoder
from pokemon_codesign_env.team_designer import TeamDesigner
from pokemon_codesign_env.transformer_data import (
    PokemonTrajectoryWriter,
    flatten_battle_observation,
    held_out_episode_indices,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--controller", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=1_000)
    parser.add_argument(
        "--environments",
        nargs="+",
        choices=ENVIRONMENTS,
        default=["random", "max-base-power", "simple-heuristics"],
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--held-out-fraction", type=float, default=0.2)
    parser.add_argument("--output", type=Path, default=Path("runs/pokemon_transformer/data.zarr"))
    args = parser.parse_args()
    held_out_episodes = held_out_episode_indices(
        args.episodes, args.held_out_fraction, args.seed + 1
    )
    environment_classes = [ENVIRONMENTS[name] for name in args.environments]
    environment_class = environment_classes[0]
    format_config = FormatConfig.from_showdown(environment_class.BATTLE_FORMAT)
    if any(format_config.format_id != cls.BATTLE_FORMAT for cls in environment_classes):
        raise ValueError("Every task in one dataset must share a battle format/action ABI")
    adapter = TeamDesignVectorAdapter(make_all_attribute_search_config(format_config))
    encoder = HumanVisibleBattleEncoder.from_showdown(
        format_config, observation_config=ObservationConfig(team_size=6)
    )
    keys = tuple(encoder.observation_space.spaces)
    observation_dimension = sum(int(np.prod(encoder.observation_space[key].shape)) for key in keys)
    model = PPO.load(args.controller)
    writer = None
    rng = np.random.default_rng(args.seed)
    for episode in range(args.episodes):
        task = args.environments[episode % len(args.environments)]
        environment_class = ENVIRONMENTS[task]
        while True:
            design = rng.uniform(size=adapter.dimension)
            parameters = adapter.decode(design)
            designer = TeamDesigner(adapter.search_config, validate_teams=True)
            designer.rng = np.random.default_rng(args.seed + episode)
            designer.set_optimizable_parameter_values(parameters)
            try:
                packed_team = designer.yield_team()
                break
            except ValueError:
                continue
        env = environment_class(team=packed_team, observation_encoder=encoder)
        transitions = []
        try:
            observation, _ = env.reset(seed=args.seed + episode)
            if writer is None:
                action_count = len(observation["action_mask"])
                writer = PokemonTrajectoryWriter(
                    adapter.dimension, observation_dimension, action_count
                )
            terminated = truncated = False
            while not terminated and not truncated:
                action, _ = model.predict(observation, deterministic=False)
                state = flatten_battle_observation(observation, keys)
                mask = np.asarray(observation["action_mask"], dtype=bool)
                observation, reward, terminated, truncated, _ = env.step(action)
                transitions.append((state, mask, int(action), float(reward)))
        finally:
            env.close()
        writer.add_episode(
            design,
            parameters,
            task,
            transitions,
            held_out=episode in held_out_episodes,
        )
    schema = {
        "version": 1,
        "team_design": adapter.schema(),
        "observation": encoder.get_schema(),
        "level_generated": False,
        "level_mean": 70,
        "level_std": 10,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    writer.write(args.output, schema=schema)
    (args.output.parent / "dataset_schema.json").write_text(
        json.dumps(schema, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
