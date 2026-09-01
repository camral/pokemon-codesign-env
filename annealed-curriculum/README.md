# Annealed curriculum baseline

This folder contains a minimal implementation of the annealed-reservoir co-design method. It trains one DQN battle controller while a population process converges from random Pokémon teams to one champion team.

## Representation

The Gym observation already contains the learner's complete visible team. `TeamEmbeddingExtractor` finds every `species_id` and `move_id` field in that schema. All species fields share one trainable embedding table and all move fields share another. The remaining categorical, continuous, and binary state is concatenated with those embeddings. Consequently, IDs are lookup keys rather than ordered quantities, without creating sparse one-hot vectors.

The baseline searches species, level, EVs, IVs, item, nature, ability, happiness, gender, and tera type when the format supports it. Moves are intentionally excluded from mutation in this first iteration. They are sampled when a candidate is created, packed into that immutable candidate, exposed to the controller through the observation, and held fixed for its trajectories. This implements the proposed trajectory-context approximation without pretending that move IDs have an ordinal relationship.

## Curriculum

Let `X` be `--evaluation-batch-size`, `Z` be the reservoir size, `Y` be `--state-samples`, and `C` be the mutation probability.

1. Sample `X` complete random teams and estimate each team's Q fitness over `Y` independently reset initial battle states.
2. Keep the best initial `Z` teams.
3. During phase `T1`, re-evaluate the elites with current network weights, fill the batch with mutations of uniformly selected reservoir teams, and keep the best annealed `Z`.
4. Sample one reservoir team per environment, reset, and collect one DQN training cycle.
5. Anneal `C` linearly from 1 to 0 and `Z` linearly from `Z_initial` to 1.
6. During `T2`, stop mutation and train only with the champion.

Mutation decisions are generated as one NumPy mask over candidate, Pokémon, and attribute axes. Sampling legal values remains a short loop because abilities, genders, and tera types have ragged species-dependent domains. Every resulting batch is validated by one persistent Showdown validator process.

The default fitness is the literal minimum predicted legal-action Q-value across `Y` states. `--fitness percentile10` and `--fitness bottom-quartile` provide the two less brittle alternatives described in the proposal. The Q network also applies the environment action mask during behavior selection and target computation.

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
    --progress \
    --verbose
```

`X` currently must equal `--num-envs`: this keeps evaluation genuinely batched and avoids constructing a second environment fleet. `--codesign-fraction 0` runs fixed-team practice after the initialized champion selection. Setting `--initial-reservoir-size 1` gives the rolling-champion variant.

Outputs are written to `runs/annealed_curriculum`: `controller.zip`, the optimizer-facing `champion.json`, and the exact packed `champion.team` used by Showdown.

## Current scope

This is an intentionally small research baseline. It uses initial battle resets for the `Y` state samples, linear schedules, and a random-player/random-team opponent. It does not yet checkpoint the reservoir, log to W&B, or make moves an explicit outer-loop action. Those can be added without changing the curriculum core.
