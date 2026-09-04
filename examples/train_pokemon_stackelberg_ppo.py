"""Train Stackelberg PPO with a six-node Pokemon team leader."""

from __future__ import annotations

import argparse
import pickle
import shutil
import sys
import time
from pathlib import Path

import numpy as np

from pokemon_codesign_env.baseline_artifacts import RunMetrics, write_run_artifacts
from pokemon_codesign_env.configs import FormatConfig, ObservationConfig
from pokemon_codesign_env.fasttd3_factory import ENVIRONMENTS
from pokemon_codesign_env.observations import HumanVisibleBattleEncoder


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("runs/pokemon_stackelberg"))
    parser.add_argument("--epochs", type=int, default=1_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-threads", type=int, default=1)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--environment", choices=ENVIRONMENTS, default="random")
    parser.add_argument("--showdown-host", default="127.0.0.1")
    parser.add_argument("--showdown-port", type=int, default=8000)
    parser.add_argument("--showdown-shards", type=int, default=1)
    parser.add_argument("--challenge-timeout", type=float, default=60)
    parser.add_argument("--min-batch-size", type=int, default=50_000)
    parser.add_argument("--mini-batch-size", type=int, default=2_048)
    parser.add_argument("--eval-batch-size", type=int, default=10_000)
    parser.add_argument("--final-evaluation-steps", type=int, default=1_000)
    return parser.parse_args()


def main():
    args = parse_args()
    if not 0 < args.mini_batch_size <= args.min_batch_size:
        raise ValueError("mini-batch-size must be positive and no larger than min-batch-size")
    started = time.monotonic()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "replays").mkdir(exist_ok=True)
    baseline = Path(__file__).resolve().parents[2] / "codesign-baselines" / "stackelberg_ppo"
    sys.path[:0] = [str(baseline.parent), str(baseline)]
    import torch
    from design_opt.agents.genesis_agent import BodyGenAgent
    from design_opt.envs import env_dict
    from design_opt.utils.config import Config
    from design_opt.utils.tools import set_global_seed
    from omegaconf import OmegaConf

    from pokemon_codesign_env.envs.stackelberg_codesign import LazyStackelbergPokemonEnv

    env_dict["codesign"] = LazyStackelbergPokemonEnv
    flags = OmegaConf.load(baseline / "design_opt" / "conf" / "config.yaml")
    flags.cfg = "general"
    flags.skel_transform_nsteps = 0
    flags.enable_wandb = False
    flags.max_epoch_num = args.epochs
    flags.seed = args.seed
    flags.num_threads = args.num_threads
    flags.min_batch_size = args.min_batch_size
    flags.mini_batch_size = args.mini_batch_size
    flags.eval_batch_size = args.eval_batch_size
    requested = args.device
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(requested)
    flags.gpu_index = device.index or 0
    config = Config(flags, str(baseline), str(args.output_dir.resolve()))
    config.env_specs.update(
        factory="pokemon_codesign_env.stackelberg_factory:make_problem",
        problem={
            "environment": args.environment,
            "showdown_host": args.showdown_host,
            "showdown_port": args.showdown_port,
            "showdown_shards": args.showdown_shards,
            "challenge_timeout": args.challenge_timeout,
            "save_replays": False,
        },
        design_step_scale=1.0,
    )
    dtype = torch.float32 if device.type == "mps" else torch.float64
    torch.set_default_dtype(dtype)
    set_global_seed(config.seed)

    class PokemonBodyGenAgent(BodyGenAgent):
        """Create fresh poke-env clients inside every sampling worker process."""

        training_battles = 0
        training_transitions = 0

        def optimize_policy(self, epoch):
            info = super().optimize_policy(epoch)
            self.training_battles += int(info["log"].num_episodes)
            self.training_transitions += int(info["log"].num_steps)
            return info

        def sample_worker(self, pid, queue, min_batch_size, mean_action, render):
            if pid == 0:
                return super().sample_worker(pid, queue, min_batch_size, mean_action, render)
            parent_env = self.env
            self.cfg.env_specs["problem"]["worker_index"] = pid
            self.env = env_dict[self.cfg.env_name](self.cfg, self)
            self.env.seed(self.cfg.seed + pid)
            try:
                return super().sample_worker(pid, queue, min_batch_size, mean_action, render)
            finally:
                self.env.problem.close()
                self.env = parent_env

    agent = PokemonBodyGenAgent(config, dtype, device, config.seed, args.num_threads, training=True)
    for epoch in range(config.max_epoch_num):
        agent.optimize(epoch)
        agent.save_checkpoint(epoch)
    checkpoint = Path(config.model_dir) / "best.p"
    if not checkpoint.is_file():
        agent.save_best_flag = True
        agent.save_checkpoint(max(config.max_epoch_num - 1, 0))
    with checkpoint.open("rb") as checkpoint_file:
        best_checkpoint = pickle.load(checkpoint_file)
    agent.policy_net.load_state_dict(best_checkpoint["policy_dict"])
    agent.value_net.load_state_dict(best_checkpoint["value_dict"])
    if agent.obs_norm is not None and best_checkpoint["obs_norm"] is not None:
        agent.obs_norm.load_state_dict(best_checkpoint["obs_norm"])
    agent.env.problem.runtime_options["save_replays"] = str(args.output_dir / "replays")
    agent.env.reset_tracking()
    agent.env.seed(args.seed + 100_000)
    _, final_log = agent.sample(
        args.final_evaluation_steps,
        mean_action=True,
        nthreads=1,
    )
    if agent.env.best_design is None:
        raise RuntimeError("Stackelberg final evaluation completed without a valid team")
    problem = agent.env.problem
    parameters = problem.adapter.decode(agent.env.best_design.reshape(-1))
    environment_class = ENVIRONMENTS[args.environment]
    format_config = FormatConfig.from_showdown(environment_class.BATTLE_FORMAT)
    encoder = HumanVisibleBattleEncoder.from_showdown(
        format_config, observation_config=ObservationConfig(team_size=6)
    )
    shutil.copy2(checkpoint, args.output_dir / "best_controller.p")
    known_wins = [won for won in agent.env.wins if won is not None]
    known_turns = [turns for turns in agent.env.turn_counts if turns is not None]
    write_run_artifacts(
        args.output_dir,
        team_parameters=parameters,
        team_design_schema=problem.adapter.schema(),
        observation_schema=encoder.get_schema(),
        metrics=RunMetrics(
            mean_return=float(final_log.avg_exec_episode_reward),
            evaluation_battles=int(final_log.num_episodes),
            training_battles=agent.training_battles,
            environment_transitions=agent.training_transitions,
            win_rate=sum(known_wins) / len(known_wins) if known_wins else None,
            mean_episode_length=float(np.mean(agent.env.episode_lengths)),
            mean_turn_count=float(np.mean(known_turns)) if known_turns else None,
            wall_time_seconds=time.monotonic() - started,
        ),
        metadata={
            "method": "stackelberg_ppo",
            "training_target_transitions": config.max_epoch_num
            * (config.min_batch_size + config.eval_batch_size),
            "best_final_episode_return": agent.env.best_return,
        },
    )


if __name__ == "__main__":
    main()
