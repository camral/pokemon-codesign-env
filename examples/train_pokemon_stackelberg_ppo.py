"""Train Stackelberg PPO with a six-node Pokemon team leader."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("runs/pokemon_stackelberg"))
    parser.add_argument("--epochs", type=int, default=1_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-threads", type=int, default=1)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--environment", default="random")
    parser.add_argument("--showdown-host", default="127.0.0.1")
    parser.add_argument("--showdown-port", type=int, default=8000)
    parser.add_argument("--challenge-timeout", type=float, default=60)
    parser.add_argument("--min-batch-size", type=int, default=50_000)
    parser.add_argument("--eval-batch-size", type=int, default=10_000)
    return parser.parse_args()


def main():
    args = parse_args()
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
            "challenge_timeout": args.challenge_timeout,
        },
        design_step_scale=1.0,
    )
    dtype = torch.float32 if device.type == "mps" else torch.float64
    torch.set_default_dtype(dtype)
    set_global_seed(config.seed)
    agent = BodyGenAgent(config, dtype, device, config.seed, args.num_threads, training=True)
    for epoch in range(config.max_epoch_num):
        agent.optimize(epoch)
        agent.save_checkpoint(epoch)


if __name__ == "__main__":
    main()
