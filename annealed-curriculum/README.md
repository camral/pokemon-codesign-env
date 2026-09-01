# Annealed curriculum baseline

This folder contains a minimal implementation of the annealed-reservoir co-design method. It trains one DQN battle controller while a population process converges from random Pokémon teams to one champion team.

## Representation

The Gym observation already contains the learner's complete visible team. `TeamEmbeddingExtractor` finds every `species_id` and `move_id` field in that schema. All species fields share one trainable embedding table and all move fields share another. The remaining categorical, continuous, and binary state is concatenated with those embeddings. Consequently, IDs are lookup keys rather than ordered quantities, without creating sparse one-hot vectors.

The baseline searches species, level, EVs, IVs, item, nature, ability, happiness, gender, and tera type when the format supports it. Moves are intentionally excluded from the persistent genome in this first iteration. `TrajectoryTeamDesigner` extends the canonical `TeamDesigner` to sample a reservoir member uniformly and sample fresh legal moves whenever an episode starts. The moves are exposed through the observation and remain fixed for that episode, making them trajectory-level context rather than an optimized physical parameter.

## Curriculum

Let `X` be `--evaluation-batch-size`, `Z` be the reservoir size, `Y` be `--state-samples`, and `C` be the mutation probability.

1. Sample `X` complete random teams and estimate each team's Q fitness over `Y` independently reset initial battle states.
2. Keep the best initial `Z` teams.
3. During phase `T1`, re-evaluate the elites with current network weights, fill the batch with mutations of uniformly selected reservoir teams, and keep the best annealed `Z`.
4. Give each environment the current reservoir. At each episode boundary it samples a team uniformly from the reservoir and samples that team's trajectory-level moves.
5. Anneal `C` linearly from 1 to 0 and `Z` linearly from `Z_initial` to 1.
6. During `T2`, stop mutation and train only with the champion.

Mutation decisions are generated as one NumPy mask over candidate, Pokémon, and attribute axes. Sampling legal values remains a short loop because abilities, genders, and tera types have ragged species-dependent domains. Every resulting batch is validated by one persistent Showdown validator process.

Fitness is always the literal minimum predicted legal-action Q-value across `Y` states. The Q network also applies the environment action mask during behavior selection and target computation.

## Run

Install the project with its RL dependencies, then run from the repository root:

```bash
python annealed-curriculum/train.py \
    --launch-server \
    --device cuda:0 \
    --num-envs 64 \
    --evaluation-batch-size 64 \
    --initial-reservoir-size 16 \
    --state-samples 8 \
    --total-steps 1000000 \
    --codesign-fraction 0.5 \
    --rollout-steps 32 \
    --vec-env sharded \
    --envs-per-worker 16 \
    --showdown-shards 4 \
    --wandb-project pokemon-codesign \
    --wandb-entity YOUR_ENTITY \
    --wandb-name annealed-curriculum \
    --progress \
    --verbose
```

`X` currently must equal `--num-envs`: this keeps evaluation genuinely batched and avoids constructing a second environment fleet. `T` and `T1` must be divisible by `num_envs × rollout_steps`, ensuring phase boundaries occur exactly between curriculum cycles. `--codesign-fraction 0` runs fixed-team practice after the initialized champion selection. Setting `--initial-reservoir-size 1` gives the rolling-champion variant.

Outputs are written to `runs/annealed_curriculum`: `controller.zip`, the optimizer-facing move-free `champion.json`, and `champion.team`, one representative trajectory-level moveset sampled for that champion.

## W&B logging

Set `--wandb-project` to enable W&B. The run records DQN/SB3 training metrics, curriculum phase and cycle, mutation probability, reservoir size, and minimum/mean/maximum Q fitness. At completion it stores the champion in the run summary and uploads the controller, JSON genome, and packed Showdown team as a model artifact. The existing `--wandb-entity`, `--wandb-group`, `--wandb-name`, and `--wandb-mode` options are supported. Omitting `--wandb-project` disables logging.

After training, `--evaluation-episodes` deterministic single-environment evaluation battles are played against the random baseline. Their mean return and one representative Showdown replay are logged to W&B.

## Current scope

This is an intentionally small research baseline. It uses initial battle resets for the `Y` state samples, linear schedules, and a random-player/random-team opponent. It does not yet checkpoint the complete reservoir. Moves deliberately remain trajectory-level context for this iteration.
