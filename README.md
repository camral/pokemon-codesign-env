# Pokémon Showdown Codesign Environment

<p align="center">
  <img src="docs/figures/codesign.png" alt="Pokémon Showdown as a co-design benchmark: an embodiment designer feeding the team design space and a control policy feeding the battle MDP, both composing your side of a single Pokémon Showdown battle that returns one score to both" width="100%">
</p>

**Pokémon Showdown** - A turn-based Pokémon battle simulator. This is based off [Pokémon Showdown](https://github.com/smogon/pokemon-showdown) and [Poke-Env](https://github.com/hsahovic/poke-env).
In this problem setup, the agent is tasked with designing a Pokémon team (embodiment) and learning a battle strategy (control policy) to maximize its win rate against various opponents.

Two searches run against each other. An **embodiment designer** searches the design space &Theta; for a
team; a **control policy** acts in the battle MDP. Your side of a battle is assembled from both &mdash;
the team you designed, played by the policy you trained &mdash; and the battle returns a single scalar
`R` that scores the pair.

The benchmark fixes &Theta;, the MDP and the opponent task suite, all resolved from a bundled,
unpatched Pokémon Showdown, so legality and turn resolution are the real game's rather than ours. It
does not fix the *method*: the designer and the policy are yours to bring, and the pair shipped in
`examples/` (Nevergrad NGOpt and SB3 MaskablePPO) are reference baselines you are meant to replace.

What makes it a co-design problem rather than two separate ones is that `R` cannot be attributed to
either half. The strongest team depends on how it is played and the strongest policy depends on what
it was handed, so neither can be optimised in isolation. Swapping a task-suite preset changes the
problem without changing anything above it.

The figure's source is [`docs/figures/codesign.drawio`](docs/figures/codesign.drawio); regenerate it
with `python docs/figures/build_figure.py` and export via the draw.io CLI. Sprite credits and licence
status for every asset used in the figures are listed in
[`docs/figures/assets/CREDITS.md`](docs/figures/assets/CREDITS.md).

The difficulties of the learning problems are yet to be fully explored, but the environment is designed to be flexible and allow for a wide range of co-design challenges.
Potential environment setups:
 - Randomized opponent teams with fixed battle rules.
 - Randomized battle rules with fixed opponent teams.
 - Randomized opponent teams and battle rules.
 - Curated opponent teams and battle rules, designed to test specific co-design strategies.
 - Multi-agent co-design, where multiple agents design their own teams and strategies, competing against each other in a shared environment.

## Installation

Pokémon Showdown is included as a Git submodule. After cloning this repository,
initialize the submodule and install its Node.js dependencies:

```bash
git submodule update --init --recursive
npm --prefix pokemon-showdown ci
```

The bundled Pokémon Showdown revision requires Node.js 22 or newer. Install this Python project and
its reinforcement-learning dependencies from the repository root:

```bash
pip install -e '.[rl]'
```

Finally, install and build the packaged Pokémon League formats in the bundled
Showdown server:

```bash
pokemon-codesign-install-showdown pokemon-showdown
```

This uses Showdown's supported `config/custom-formats.ts` extension point and
builds the server. It does not patch the Showdown simulator. If the server
already has custom formats, the command preserves them and explains the import
that must be added manually.

## The codesign Gymnasium API

A codesign environment is a Gymnasium environment with a third space alongside the
observation and action spaces: the embodiment design space &Theta;. Choose a point in it
when you reset, and the environment launches and owns the Showdown server it needs.

```python
import Pokemon_codesign_env as pce

env, base_embodiment = pce.make("random")
observation, info, embodiment_rejected = env.reset(seed=0, embodiment=env.embodiment_space.sample())
env.close()
```

[`docs/codesign_gym_api.md`](docs/codesign_gym_api.md) documents the whole extension: the
design space and how designs are rejected rather than repaired, the flat `MultiDiscrete`
representation and the batched `validate_embodiments` screen, passing one design per
environment to a vector environment, and how to convert &Theta; between coordinate systems
if your designer wants different ones.

## Environment Throughput

The environment is an entirely cpu-based simulation, and as such scales with what the target device allows. We have included a simple scaling test on a machine with 2 AMD EPYC 9555 64-Core Processor (64-core, 128-thread processor) CPUs on different numbers of parallel environments with different shard numbers.

<p align="center">
  <img src="docs/figures/scaling_curve.png" alt="Effective and Environment FPS on 2 random-actors in Pokemon Showdown across different numbers of parallel environments and shard numbers. Effective FPS includes the time taken to reconstruct the environments with new teams/embodiments." width="100%">
</p>


## Minimal SB3 + NGOpt experiment

The example in `examples/train_sb3_ngopt.py` uses NGOpt to choose the species
and nature of all six Pokémon. Every vector environment receives an independent
NGOpt team, and the complete population is scored and replaced after every PPO
rollout. One masked PPO controller learns continuously across these embodiment
populations using `RandomPlayerEnv` by default. Each baseline environment gives
the opponent a newly generated random team for every battle.

Run the small experiment:

```bash
python examples/train_sb3_ngopt.py \
  --ppo-timesteps 2048 \
  --evaluation-episodes 8 \
  --verbose
```

There is no server to start. The environments launch the Pokémon Showdown servers
they need, on ports claimed from the operating system, and stop them when they are
closed. Concurrent runs therefore never collide, and nothing is left running
afterwards.

All training examples use the CPU and one in-process environment by default.
Select any PyTorch device with `--device`, including a specific GPU:

```bash
python examples/train_sb3_ngopt.py --device cuda
python examples/train_pokemon_league_sb3_ngopt.py --device cuda:6
```

Use `--device auto` to let SB3 select CUDA when available and otherwise use the
CPU. The same option also controls where saved controllers are loaded in watch
mode.

For a GPU training run, use process-parallel Showdown environments and a larger
rollout/minibatch:

```bash
python examples/train_sb3_ngopt.py \
  --device cuda:6 \
  --num-envs 8 \
  --vec-env subproc \
  --rollout-steps 512 \
  --batch-size 1024 \
  --profile-gpu \
  --gpu-log-interval 4096 \
  --verbose
```

For hundreds or thousands of logical environments, use the sharded backend
instead of creating one Python process per environment. This example creates
1,024 environments using 32 rollout processes and distributes their battles
across eight Showdown servers on ports 8000 through 8007:

```bash
python examples/train_sb3_ngopt.py \
  --device cuda:0 \
  --num-envs 1024 \
  --vec-env sharded \
  --envs-per-worker 32 \
  --reset-retries 3 \
  --challenge-timeout 20 \
  --showdown-shards 8 \
  --rollout-steps 32 \
  --batch-size 8192 \
  --ppo-timesteps 1048576 \
  --random-team-pool-size 8192 \
  --profile-gpu \
  --progress \
  --gpu-log-interval 8192 \
  --telemetry-log-interval 1024 \
  --wandb-project Pokémon-codesign \
  --wandb-entity YOUR_ENTITY \
  --wandb-name gpu-1024-envs \
  --wandb-mode online \
  --verbose
```

Each sharded rollout process owns up to `--envs-per-worker` environments and
steps their I/O-bound battles concurrently. `--showdown-shards S` launches `S`
Showdown servers and routes the environments round-robin over them; leaving it at
0 uses one server per 32 environments. The servers are started and stopped by the
environments themselves. Random opponent teams are
generated into one reusable pool so resets do not launch a Node process per
battle. Start below the machine's limits and increase environments, rollout
workers, and server shards while watching transitions per second and P95
latency; raw GPU utilization is not the primary throughput measure for the
small MLP controller.

The PPO rollout buffer contains `num_envs * rollout_steps` samples, and
`batch-size` must divide that value. `--profile-gpu` reports the selected GPU,
utilization when supported by the installed PyTorch build, and allocated,
reserved, and peak CUDA memory during training.

NGOpt evaluates `num_envs` embodiments concurrently. Their rollout returns are
reported to the optimizer, new teams are sampled and batch-validated, and each
environment is reset onto its replacement team before the next rollout.
The total number of NGOpt evaluations is exactly
`num_envs * ceil(ppo_timesteps / (num_envs * rollout_steps))`; population size
and parallelism both come directly from `--num-envs`.

The installed PyTorch build must include support for the requested accelerator.
Before a long run, verify the selected CUDA index with:

```bash
python -c 'import torch; print(torch.cuda.is_available(), torch.cuda.device_count())'
```

Enable W&B evaluation logging by supplying a project:

```bash
python examples/train_pokemon_league_sb3_ngopt.py \
  --generation 4 \
  --wandb-project Pokemon-codesign \
  --wandb-entity YOUR_ENTITY \
  --wandb-group gen4-league
```

W&B tracks the full experiment: PPO rollout and training statistics, NGOpt
population mean and incumbent returns, wall time, optional GPU statistics, and
promoted controller checkpoints. Scale telemetry starts before the first PPO update and includes
environment readiness, startup time, vector steps, buffer fill, completed
episodes, recent FPS, average FPS, and transitions per second. Replay recording
remains evaluation-only. After training, the final NGOpt recommendation is
evaluated by one environment and one representative battle is displayed in W&B
as an HTML Pokémon Showdown replay and uploaded in a downloadable
`evaluation-replays` artifact.
Use `--wandb-mode offline` to save locally for later `wandb sync`, or
`--wandb-mode disabled` to exercise the same configuration without logging.
Optional run organization is available through `--wandb-name` and
`--wandb-group`.

Switch baseline environments with one option:

```bash
python examples/train_sb3_ngopt.py --environment max-base-power
python examples/train_sb3_ngopt.py --environment simple-heuristics
```

The available values are `random`, `max-base-power`, and `simple-heuristics`.
Use the same value for training and `--watch` because each preset has a different
generation and ruleset.

`--verbose` prints each candidate, SB3's PPO rollout metrics, and evaluation
progress. Omit it for one summary line per candidate.
`--progress` displays a tqdm bar for every PPO training run with recent and
average environment-transition FPS. It can be combined with `--verbose` and
W&B logging.

The controller uses a compact human-visible battle observation: its own complete
team, revealed opponent information, known moves and PP, HP, status, stat
boosts, volatile effects, weather, field effects, side conditions, and
currently available move details. Categorical values are direct deterministic
integer IDs rather than one-hot vectors or learned embeddings. Zero represents
unknown or padding. Simulator-private opponent stats, nature, moves, item, and
ability are not exposed before Showdown reveals them. The legal-action mask
remains separate. Gym exposes these as one top-level `Dict` containing a
`MultiDiscrete` categorical vector, `Box` continuous vector, `MultiBinary`
binary vector, and `MultiBinary` action mask. The exact feature ranges and ID
mappings are saved as `observation_schema.json` beside each trained controller.

Observation contents are controlled by `ObservationConfig`. For example:

```python
ObservationConfig(
    include_stats=False,
    include_effects=False,
    max_side_conditions=8,
)
```

The example's selected configuration is `OBSERVATION_CONFIG` near the top of
`examples/train_sb3_ngopt.py`.

Controllers trained with the earlier 12-value observation are incompatible
with this encoder and must be retrained before using `--watch`.

The final recommended team and jointly trained PPO controller are written to
`runs/minimal_codesign/`.

## Pokémon League experiment

`examples/train_pokemon_league_sb3_ngopt.py` applies the same rollout-population
SB3 and NGOpt setup to a complete five-battle League episode:

```bash
python examples/train_pokemon_league_sb3_ngopt.py \
  --generation 4 \
  --difficulty 0 \
  --verbose
```

Use `--random-league` instead of `--generation` to sample one League per
episode while allowing Pokémon from every generation.

To watch the saved controller attempt the full League, leave the Showdown
server running, open `http://localhost:8000`, and use the same generation and
difficulty that were used for training:

```bash
python examples/train_pokemon_league_sb3_ngopt.py \
  --generation 4 \
  --difficulty 0 \  
  --watch \
  --turn-delay 1
```

The script prints a new direct battle URL for each League trainer. Use
`--watch-games` for multiple League attempts. Replays are saved under
`runs/pokemon_league/replays/`.

To watch slowed exhibition games with the saved best controller, leave the
Showdown server running, open `http://localhost:8000`, and run:

```bash
python examples/train_sb3_ngopt.py \
  --watch \
  --watch-games 3 \
  --turn-delay 1
```

The script prints the direct URL for each live battle and waits five seconds
before its first turn. HTML replays are saved under
`runs/minimal_codesign/replays/`.
