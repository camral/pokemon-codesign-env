# Pokemon Co-Design Baselines: Source-of-Truth Plan

## Status and authority

This document is the source of truth for implementing Pokemon co-design training for the six algorithms in `../codesign-baselines`.

Implementation must remain faithful to the decisions recorded here. If a later discovery conflicts with this plan, update this document explicitly before changing the implementation.

## Implementation progress (2026-09-03)

Completed foundation:

- All-attribute `TeamDesignVectorAdapter`, derived from `TeamDesigner`, with deterministic conditional decoding, encoding, schema versioning, stable serialization, and candidate IDs.
- Shared batch candidate compilation through `BatchedTeamDesigner` followed by Showdown batch validation.
- Learner and random-opponent level sampling fixed at `Normal(70, 10)` and clipped to format bounds; level is absent from optimizer schemas.
- PPO + NGOpt parameterization now exposes all supported attributes and writes `team_design_schema.json`.
- PPO + NGOpt candidate fitness now uses an equal number of complete deterministic battles under a fixed controller snapshot; rollout sums are telemetry only.
- FastTD3 phase-projection primitives cover environment actions, replay extraction from `infos["executed_actions"]`, and differentiable actor/target-policy projection.
- Focused adapter, level, population-evaluation, and FastTD3 contract tests.

Still pending:

- Full shared metrics/artifact layer and held-out evaluation statistics.
- House of Dextra stages 1-3.
- CMA-ES trainer.
- FastTD3 environment and upstream collection/training-loop integration.
- Stackelberg PPO adapter/trainer.
- Transformer Transformer data schema, dataset tooling, trainer, and evaluator.

Current implementation directive: defer the remainder of House of Dextra after its graph adapter and recurrent-controller primitives; complete all other methods first.

## Settled decisions

### Canonical team representation

`TeamDesigner` and `BatchedTeamDesigner` are the only canonical Pokemon design representations.

- Do not create a second canonical team model.
- Optimizer-specific vectors, graph features, and tokens are adapters only.
- Every optimizer result must be decoded into the nested parameter dictionaries accepted by `TeamDesigner`.
- Every final team must be compiled by `TeamDesigner` and validated by Pokemon Showdown.

### Optimized attributes

Expose every battle-relevant attribute already supported as optimizable by `TeamDesigner`:

- Species
- Moves
- Item
- Nature
- Ability
- Happiness
- Gender
- Tera type when supported by the format
- EVs
- IVs

Level is not optimized.

### Levels

Both agent teams and random opponent teams use:

```python
level_mean = 70
level_std = 10
```

- Sample levels from this distribution independently when building teams.
- Clip sampled levels to the selected format's legal level range.
- Do not expose level to any embodiment optimizer.
- Do not calculate the level distribution from an individual opposing team.

### Parallel environments

Parallel environments normally explore different teams.

- PPO + NGOpt: one different NGOpt candidate per environment.
- CMA-ES: different candidates across workers; evaluation repeats of one candidate share its team and policy but use different seeds/opponents.
- House of Dextra: different GNN-selected designs across evaluation environments.
- FastTD3: each environment selects its own embodiment on the first transition.
- Stackelberg PPO: each environment/episode receives its own leader-selected embodiment.
- Transformer Transformer: training uses batches of different offline teams and trajectories; evaluation workers normally test different generated candidates.

The same team is duplicated across environments only for repeated evaluation of one specific candidate.

## Shared implementation

### Search configuration

Add a shared constructor for an all-attributes `TeamSearchConfig`.

- Set every supported attribute mode above to `OPTIMIZED`.
- Keep `level_mean=70` and `level_std=10`.
- Respect format generation, type restrictions, bans, team size, move count, EV limits, IV limits, and Tera support.
- Attributes unavailable in a format are inactive and absent from the optimizer adapter rather than meaningless search dimensions.

### Optimizer adapter

Add an adapter module under a package such as:

```text
src/pokemon_codesign_env/baseline_adapters/
    team_design.py
```

The adapter derives its schema from `TeamDesigner.get_optimizable_parameter_space()` and provides:

- Structured parameter sampling.
- Fixed numeric encoding and decoding for continuous optimizers.
- Legality-preserving mutation for graph search.
- Graph feature encoding.
- Token encoding for Transformer Transformer.
- Stable serialization and candidate identity.
- Batch compilation through `BatchedTeamDesigner`.

It is not an alternative team representation.

### Conditional decoding

Decode species before species-dependent attributes.

- Decode moves from the selected species' legal move list.
- Select moves without replacement.
- Decode ability from the selected species' legal abilities.
- Decode gender from the selected species' legal genders.
- Decode Tera type from the selected species' legal Tera types when supported.
- Round happiness to a legal integer.
- Project EVs and IVs into their per-stat caps and total budgets.

The vector schema must be deterministic, versioned, and saved with each run.

### Legality

Use three layers of legality handling:

1. Decode locally valid attribute values.
2. Repair predictable team-wide conflicts where the repair is deterministic and unbiased enough to document.
3. Batch-validate compiled teams with Pokemon Showdown.

Candidates that remain invalid receive the algorithm's invalid-candidate penalty and are replaced when necessary.

Do not rely on `RetryingTeamDesigner` to repair fully optimized candidates: retries do not change their optimized attributes.

Retain the existing poke-env species-trackability check so the observation system can distinguish team members reliably.

### Controller observation and action masking

Use the existing human-visible battle observation and legal-action mask.

- Controllers must never receive simulator-private opponent information.
- The complete resolved agent team remains visible through the observation.
- House of Dextra additionally receives an explicit stable encoding of the designed team as its morphology-conditioning input.
- Discrete Pokemon battle controllers must mask illegal actions before sampling or selecting an action.

### Common evaluation

All six methods share the same evaluation protocol where their algorithm permits it:

- Same battle format and opponent/task presets.
- Same held-out opponent seeds and teams.
- Same reward function.
- Same number of complete evaluation battles per candidate.
- Deterministic controller evaluation where appropriate.
- Report mean return, win rate, episode length, turn count, invalid-candidate rate, and repair rate.
- Report compute budgets in environment transitions, training battles, evaluation battles, and wall time.

Population search and controller training must be distinguished in metrics. Do not silently compare one optimizer call against one RL transition.

### Shared artifacts

Every completed run should emit applicable artifacts:

```text
best_team.json
best_controller.*
team_design_schema.json
observation_schema.json
metrics.json
replays/
```

## Method 1: PPO + NGOpt

### Entry point

```text
examples/train_pokemon_ppo_ngopt.py
```

### Controller

Use one shared SB3 masked PPO controller across the whole embodiment population.

### Embodiment search

1. Ask NGOpt for one candidate per parallel environment.
2. Decode each candidate through the shared adapter into `TeamDesigner` parameters.
3. Batch-compile and validate the population.
4. Assign a different valid team to each environment.
5. Train the shared PPO controller across that population.
6. Freeze controller weights for candidate scoring.
7. Evaluate candidates over an equal number of complete battles.
8. Report mean candidate returns to NGOpt.
9. Ask for and install the next population.

Reuse the existing masked policy, vector environments, Showdown sharding, logging, checkpointing, and replay support.

### Required change from the current example

The current example exposes only species and nature and scores candidates using summed rewards from a PPO rollout. Generalize the parameterization to all optimized attributes.

Use fixed complete-battle evaluations for optimizer fitness. Raw rollout sums mix controller learning, episode length, and the number of battles completed.

### Concern

The shared controller changes over training, so NGOpt sees a non-stationary objective. Log controller version with every candidate evaluation and periodically rescore the incumbent using the current controller.

## Method 2: CMA-ES

### Entry point

```text
examples/train_pokemon_cmaes.py
```

### Controller

CMA-ES jointly evolves a compact masked battle controller and the Pokemon team.

- Use a small fixed-topology MLP or linear controller.
- Flatten its weights into the baseline's `policy` block.
- Do not evolve the full SB3 PPO network initially; that search dimension is impractical.

### Embodiment and policy search

1. Decode the CMA-ES `design` block through the shared adapter.
2. Load the `policy` block into the compact controller.
3. Compile and validate the team.
4. Evaluate that exact team-policy pair over repeated battles.
5. Use different opponent seeds for repeats.
6. Evaluate different CMA-ES candidates concurrently across Showdown workers/shards.
7. Return mean reward to CMA-ES.
8. Save the recommended team and policy weights.

### Concerns

- CMA-ES is continuous, while most Pokemon design choices are categorical.
- Numeric categorical decoding creates discontinuities, flat regions, and artificial category adjacency.
- Evaluators must run in isolated processes with unique Showdown accounts; do not share poke-env clients across executor threads.
- This method will likely need a substantially larger battle budget than NGOpt.

## Method 3: House of Dextra

House of Dextra must follow the paper's controller procedure, not an invented online co-training loop.

### Entry points

```text
examples/train_pokemon_house_controller.py
examples/train_pokemon_house_of_dextra.py
examples/finetune_pokemon_house_controller.py
```

### Stage 1: Cross-embodiment controller pretraining

Pretrain a morphology-conditioned recurrent PPO policy across randomly sampled complete Pokemon teams.

- Use PPO.
- Follow the upstream recurrent controller shape unless a Pokemon-specific incompatibility is demonstrated: MLP layers `[512, 256, 128]` and a 256-unit GRU.
- Condition the policy explicitly on the complete team-design encoding.
- Retain the normal state-dependent Pokemon legal-action mask.
- Randomly sample all optimizable team attributes through `TeamDesigner`.
- Give parallel environments different random teams.
- Treat a battle format/ruleset as one embodiment family because its teams share an action and observation ABI.
- Train and save one cross-embodiment checkpoint per selected family/format when their ABIs differ.

### Stage 2: Graph Heuristic Search

Freeze the pretrained cross-embodiment controller during design search.

Represent each team as a fixed graph:

- One team/root node and six Pokemon nodes in a star graph, mirroring House of Dextra's fixed root-plus-slots representation.
- Pokemon node features expose every optimized attribute.
- Use stable binary encodings or learned categorical embeddings rather than treating integer IDs as meaningful continuous magnitudes.

Implement legal mutations:

- Replace species and regenerate or repair its dependent attributes.
- Replace one move.
- Replace item, nature, ability, gender, or Tera type.
- Change happiness.
- Transfer EV or IV points between stats while preserving constraints.

Search loop:

1. Mutate frontier designs to create a legal candidate pool.
2. Use the design-value GNN to rank the pool.
3. Select exploit and exploration candidates.
4. Evaluate different selected teams in parallel with the frozen cross-embodiment PPO checkpoint.
5. Train the design-value GNN to regress observed candidate performance.
6. Promote the best candidates and repeat.

Do not update the PPO controller inside each GNN search iteration.

### Stage 3: Final controller fine-tuning

After search:

1. Select the winning design using held-out evaluation.
2. Initialize its controller from the cross-embodiment PPO checkpoint.
3. Fine-tune a copy specifically on the winning team.
4. Preserve the frozen base checkpoint separately.
5. Evaluate and save both zero-shot base-policy performance and fine-tuned performance.

### Concerns

- The generic baseline adapter omits controller pretraining and fine-tuning; Pokemon-specific entry points must supply both.
- Existing upstream checkpoints are dexterous-hand checkpoints and cannot be used for Pokemon.
- The design-value GNN sees performance under the frozen base controller, which may rank teams differently from fully fine-tuned controllers. Report both zero-shot and post-fine-tuning results.

## Method 4: FastTD3

### Entry point

```text
examples/train_pokemon_fasttd3.py
```

### Combined action space

FastTD3 receives one fixed continuous action vector:

```text
action = [embodiment_action, battle_action]
```

The observation includes an unambiguous design/execution phase indicator.

### Design transition

The first step of every rollout selects the embodiment.

```text
actor output:    [embodiment_action, battle_action]
executed action: [embodiment_action, zeros]
```

On this transition:

1. Zero and discard the battle-action block.
2. Decode the embodiment block through the shared adapter and `TeamDesigner`.
3. Compile and validate the team.
4. Start the battle using that team.
5. Return zero reward for the design-selection transition.

The environment must not start the underlying Pokemon battle before this transition.

### Battle transitions

After embodiment selection:

```text
actor output:    [embodiment_action, battle_action]
executed action: [zeros, battle_action]
```

On these transitions:

1. Zero and discard the embodiment block.
2. Keep the selected team unchanged for the rest of the battle.
3. Apply the Pokemon legal-action mask to the battle-action block.
4. Convert it into the executed Showdown action.
5. Step the battle and return its reward.

Each parallel environment selects its own embodiment.

### Executed actions and replay

The replay buffer must contain the action actually retained after phase-dependent discarding, not the actor's original unprojected concatenated action.

1. The environment returns `executed_actions` in `infos`.
2. The FastTD3 collection loop stores `infos["executed_actions"]` in the transition.
3. The discarded action block is zero in replay.
4. Also record the selected discrete Showdown action ID for diagnostics.

Apply the same phase projection when computing:

- Current actor actions used by the actor loss.
- Target-policy actions used in Bellman targets.
- Evaluation actions.

Without this, the critic trains on projected actions while actor and target updates query it with impossible actions where both blocks are active.

For the battle block, initially retain the legal masked continuous scores in replay and separately record the selected discrete action ID. Do not replace the scores with a hard one-hot action unless a differentiable or otherwise justified actor-training treatment is added.

### Concerns

- Both the team decoder and battle argmax are discontinuous, while TD3 assumes continuous control.
- The generic FastTD3 adapter uses `SyncVectorEnv`, which is unsuitable for many network-bound Showdown environments. Add asynchronous or sharded collection.
- Phase projection must work correctly across vector autoresets and terminal observations.
- N-step replay must not cross the design boundary of a new episode incorrectly.

## Method 5: Stackelberg PPO

### Entry point

```text
examples/train_pokemon_stackelberg_ppo.py
```

### Design and control stages

Use six Pokemon graph nodes.

1. The leader selects all optimized team attributes once at the start of an episode.
2. Decode the leader output through the shared adapter and `TeamDesigner`.
3. Start the battle only after the embodiment has been selected.
4. The follower controls the fixed selected team until the battle terminates.
5. Convert follower outputs into global Pokemon battle-action scores and mask illegal actions.

Parallel environments receive different leader-generated teams.

### Required adapter changes

- Add lazy backend initialization so environment construction and reset do not start throwaway battles before the design action.
- Ensure the single leader design step can reach the entire normalized design range. The current additive update and default scale may expose only a small neighborhood.
- Add a Pokemon-appropriate global control head or an explicit principled mapping from node-major follower output to one global Showdown action.

### Concerns

- The upstream follower action is node-major, while a Pokemon battle action is global.
- Categorical team decisions are decoded from continuous leader outputs.
- The skeleton-action head is irrelevant for fixed six-slot Pokemon teams and must remain a documented no-op rather than influencing results.

## Method 6: Transformer Transformer

### Entry points

```text
examples/generate_pokemon_transformer_dataset.py
examples/train_pokemon_transformer_transformer.py
examples/evaluate_pokemon_transformer_transformer.py
```

### Representation

Do not encode Pokemon as MJCF or pretend existing RoboTokens describe teams. Add native Pokemon modalities while keeping `TeamDesigner` canonical.

Proposed modalities:

- `pokemon`: slot, species, item, nature, ability, happiness, gender, and Tera type.
- `move`: Pokemon slot, move slot, and move identity.
- `ev`: Pokemon slot and six-stat allocation.
- `iv`: Pokemon slot and six-stat allocation.
- `battle_state`: human-visible battle state over time.
- `opponent_task`: format and opponent/task descriptor.
- `control`: executed legal battle action.
- `action_mask`: legal actions at each timestep.

Level may be present as observed rollout context but is never a generated embodiment output.

### Dataset generation

1. Sample legal complete teams through `TeamDesigner`.
2. Sample both sides' levels from `Normal(70, 10)`.
3. Run an expert or sufficiently trained cross-embodiment controller.
4. Record team parameters, observations, masks, executed actions, rewards, opponent/task descriptors, and episode boundaries.
5. Write training and held-out Zarr datasets.

### Training tasks

- Control task: predict legal battle actions conditioned on the team and battle state.
- Team-generation task: generate every optimized team attribute conditioned on the opponent/task descriptor.
- Optionally train both heads over one shared transformer backbone, matching Transformer Transformer's combined hardware-generation/control principle.

### Evaluation

1. Generate several candidate teams for the same task.
2. Decode candidates into `TeamDesigner` parameters.
3. Compile and validate them.
4. Play them with the learned controller.
5. Rank them using held-out complete battles.

### Concerns

- Existing robot checkpoints are incompatible; train Pokemon models from scratch.
- New token groups, serialization, Zarr schema, adapters, and configurations are required.
- Dataset generation and training compute dominate this method.
- The task-conditioning schema must be fixed and versioned before generating data.

## Testing requirements

### Team adapter tests

- Structured parameter round trips.
- Numeric encode/decode determinism.
- Species-dependent move, ability, gender, and Tera legality.
- Unique moves.
- EV/IV cap and budget preservation.
- Unsupported-format attributes remain inactive.
- Level is absent from optimizer schemas.
- Sampled levels use mean 70 and standard deviation 10 before clipping.
- Fuzzed candidates compile and pass Showdown validation or are rejected deterministically.

### Environment tests

- Team replacement occurs only at intended boundaries.
- Different parallel environments can hold different teams.
- Legal-action masks prevent illegal controller actions.
- Design-phase transitions do not start battles prematurely.
- FastTD3 stores zeroed inactive action blocks.
- FastTD3 actor and target actions receive the same phase projection as replay actions.
- Stackelberg starts exactly one battle after its leader design action.

### Training smoke tests

For every method:

- Minimal CPU run.
- At least two environments where parallelism is part of the method.
- One optimizer/controller update.
- Checkpoint save and reload.
- Final team JSON recompiles successfully.
- Short deterministic held-out evaluation.

## Implementation order

1. Shared all-attribute search configuration and optimizer adapter.
2. Fixed `Normal(70, 10)` level handling for agent and random opponent builders.
3. Shared candidate compilation, validation, evaluation, metrics, and artifacts.
4. PPO + NGOpt.
5. CMA-ES.
6. FastTD3 combined-action environment and executed-action replay changes.
7. Stackelberg PPO.
8. Transformer Transformer dataset generation, training, and evaluation.
9. Unified cross-baseline benchmark runner.
10. Resume House of Dextra controller pretraining, graph search, and fine-tuning when requested.

## Implementation progress

- Shared all-attribute adapters, fixed level sampling, validation, evaluation metrics, and artifacts are implemented.
- PPO + NGOpt and CMA-ES entry points are implemented.
- FastTD3 has a two-phase combined action space, phase-projected actor actions, and executed-action replay storage.
- Stackelberg PPO has a lazy design-to-battle environment and upstream-problem adapter.
- Transformer Transformer has native Pokemon modalities, task conditioning, held-out splitting, training, generation, and evaluation entry points.
- The cross-baseline runner validates and combines the common artifact contract.
- House of Dextra remains intentionally deferred after its graph and recurrent-controller primitives.

## Known global concerns

- Pokemon's design space is highly categorical and conditional; continuous baselines will receive discontinuous objectives.
- Shared or pretrained cross-embodiment controllers measure how well a team works with that controller, not its theoretical best possible policy.
- Search comparisons become misleading if controller versions, evaluation battle counts, or opponent seeds differ.
- Showdown and poke-env are I/O-bound; algorithm-side GPU batching does not remove the need for server sharding and isolated accounts.
- Dependencies span this repository and `../codesign-baselines`; integration should use explicit installable packages or validated entry-point loading rather than fragile implicit imports.
- Some custom Showdown formats report a `max_move_count` larger than the four moves executable by the current battle controller ABI. Optimizer adapters therefore cap active move slots at four until the controller ABI explicitly supports more.
- In the current environment, importing `poke_env` starts a daemon event-loop thread that prevents otherwise-completed short Python and pytest processes from exiting naturally. Focused tests report success before an external timeout terminates the idle process; this shutdown behavior should be isolated and fixed separately.
