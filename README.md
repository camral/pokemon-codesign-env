# Pokemon Showdown Codesign Environment

**Pokemon Showdown** - A turn-based Pokemon battle simulator. This is based off [Pokemon Showdown](https://github.com/smogon/pokemon-showdown) and [Poke-Env](https://github.com/hsahovic/poke-env).
In this problem setup, the agent is tasked with designing a Pokemon team (embodiment) and learning a battle strategy (control policy) to maximize its win rate against various opponents.

The difficulties of the learning problems are yet to be fully explored, but the environment is designed to be flexible and allow for a wide range of co-design challenges.
Potential environment setups:
 - Randomized opponent teams with fixed battle rules.
 - Randomized battle rules with fixed opponent teams.
 - Randomized opponent teams and battle rules.
 - Curated opponent teams and battle rules, designed to test specific co-design strategies.
 - Self-play, where the agent competes against its own previous versions to iteratively improve both its team design and battle strategy.
 - Multi-agent co-design, where multiple agents design their own teams and strategies, competing against each other in a shared environment.

## Minimal SB3 + NGOpt experiment

The example in `examples/train_sb3_ngopt.py` uses NGOpt to choose the species
and nature of all six Pokemon. It trains a fresh masked PPO controller for each
candidate using `RandomPlayerEnv` by default. Each baseline environment gives
the opponent a newly generated random team for every battle.

Install the reinforcement-learning dependencies:

```bash
pip install -e '.[rl]'
```

Install the packaged Pokemon League formats into the bundled Showdown server:

```bash
pokemon-codesign-install-showdown pokemon-showdown
```

This uses Showdown's supported `config/custom-formats.ts` extension point and
builds the server. It does not patch the Showdown simulator. If the server
already has custom formats, the command preserves them and explains the import
that must be added manually.

Start the bundled Pokemon Showdown server in another terminal:

```bash
cd pokemon-showdown
node pokemon-showdown start --no-security
```

Then run the small experiment:

```bash
python examples/train_sb3_ngopt.py \
  --ng-budget 4 \
  --ppo-timesteps 2048 \
  --evaluation-episodes 8 \
  --verbose
```

Add `--launch-server` to start the bundled local Showdown server automatically.
The script reuses an existing server on port 8000 when present; otherwise it
starts one, waits until it is ready, and stops only that server when it exits.

All training examples use the CPU and one in-process environment by default.
Select any PyTorch device with `--device`, including a specific GPU:

```bash
python examples/train_sb3_ngopt.py --device cuda
python examples/train_pokemon_league_sb3_ngopt.py --device cuda:6
python examples/train_self_play_sb3_ngopt.py --device mps
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
  --launch-server \
  --verbose
```

The PPO rollout buffer contains `num_envs * rollout_steps` samples, and
`batch-size` must divide that value. `--profile-gpu` reports the selected GPU,
utilization when supported by the installed PyTorch build, and allocated,
reserved, and peak CUDA memory during training.

`--ng-workers N` evaluates up to N embodiment candidates concurrently and
requires `--vec-env subproc` so each candidate's poke-env state remains process
isolated. Each candidate owns its own PPO model and vector environment, so
increase this only when the server, CPU, and GPU memory can support them.
Usually one NGOpt worker with several vector environments is the best
configuration for a single GPU. The self-play example keeps frozen opponent
inference on CPU by default; use `--opponent-device` to change it.

The installed PyTorch build must include support for the requested accelerator.
Before a long run, verify the selected CUDA index with:

```bash
python -c 'import torch; print(torch.cuda.is_available(), torch.cuda.device_count())'
```

Enable W&B evaluation logging by supplying a project:

```bash
python examples/train_pokemon_league_sb3_ngopt.py \
  --generation 4 \
  --wandb-project pokemon-codesign \
  --wandb-entity YOUR_ENTITY \
  --wandb-group gen4-league \
  --launch-server
```

W&B tracks the full experiment: PPO rollout and training statistics, NGOpt
candidate returns and incumbent progress, candidate duration, wall time, the
current recommendation, optional GPU statistics, and promoted controller
checkpoints. Replay recording remains evaluation-only, and every candidate is
evaluated by one environment even when training uses multiple vector workers.
Each completed evaluation battle is displayed in W&B as an HTML Pokemon
Showdown replay and uploaded in a downloadable `evaluation-replays` artifact. A
Pokemon League evaluation can therefore log up to five replays per episode.
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

The best evaluated team and its PPO controller are written to
`runs/minimal_codesign/`. This is intentionally a nested baseline: every NGOpt
candidate receives a freshly initialized PPO controller, making embodiment
scores comparable at the cost of additional training time.

## Pokemon League experiment

`examples/train_pokemon_league_sb3_ngopt.py` applies the same nested SB3 and
NGOpt setup to a complete five-battle League episode:

```bash
python examples/train_pokemon_league_sb3_ngopt.py \
  --generation 4 \
  --difficulty 0 \
  --launch-server \
  --verbose
```

Use `--random-league` instead of `--generation` to sample one League per
episode while allowing Pokemon from every generation.

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
`--watch-games` for multiple League attempts, or add `--launch-server` to
manage the local server automatically. Replays are saved under
`runs/pokemon_league/replays/`.

## Self-play experiment

`examples/train_self_play_sb3_ngopt.py` performs iterative frozen-policy
self-play. The first round bootstraps against `RandomPlayer`; each later round
runs NGOpt and trains fresh PPO challengers against the previous round's fixed
champion team and controller.

```bash
python examples/train_self_play_sb3_ngopt.py \
  --self-play-rounds 3 \
  --ng-budget 4 \
  --launch-server \
  --verbose
```

Round snapshots are saved under `runs/self_play/round_N/`, and the latest
champion is also saved as `best_controller.zip` and `best_team.json`.

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
