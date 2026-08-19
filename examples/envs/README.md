# Preconfigured environments

These examples combine a Showdown format, a configurable human-visible observation encoder, and a built-in poke-env opponent into a single-agent Gymnasium environment.

| Environment | Opponent | Default format | What it demonstrates |
| --- | --- | --- | --- |
| `RandomPlayerEnv` | `RandomPlayer` | `gen4customgame` | Generation 4 mechanics with unrestricted team co-design |
| `MaxBasePowerPlayerEnv` | `MaxBasePowerPlayer` | `gen6ou` | Generation 6 OU clauses and a damage-oriented opponent |
| `SimpleHeuristicsPlayerEnv` | `SimpleHeuristicsPlayer` | `gen9monotype` | Modern mechanics, Monotype team constraints, and poke-env's stronger scripted baseline |

The co-design agent always supplies its own team through `team`. Unless `opponent_team` is supplied, the environment asks Showdown for a new random team before every battle and validates it against the preset's target ruleset. Gen 4 Random Battle supplies the Custom Game opponent, Gen 6 Battle Factory supplies the OU opponent, and Gen 9 Monotype Random Battle supplies the Monotype opponent.

## Usage

Start Pokemon Showdown, then create an environment like any other Gymnasium environment:

```python
import numpy as np

from examples.envs import RandomPlayerEnv
from pokemon_codesign_env.configs import FormatConfig, TeamSearchConfig
from pokemon_codesign_env.team_designer import TeamDesigner

format_config = FormatConfig.from_showdown(RandomPlayerEnv.BATTLE_FORMAT)
learner_team = TeamDesigner(TeamSearchConfig(format=format_config))
env = RandomPlayerEnv(team=learner_team)
observation, info = env.reset(seed=0)

terminated = truncated = False
while not terminated and not truncated:
    legal_actions = np.flatnonzero(observation["action_mask"])
    action = env.np_random.choice(legal_actions)
    observation, reward, terminated, truncated, info = env.step(action)

env.close()
```

Each observation is a Gymnasium `Dict` containing `categorical`, `continuous`, `binary`, and `action_mask`. The state components come from `HumanVisibleBattleEncoder`; the opponent sees its own poke-env battle state and is not part of the RL observation.

## Configuration

Every preset accepts the same overrides:

```python
from examples.envs import MaxBasePowerPlayerEnv
from pokemon_codesign_env.configs import FormatConfig, ObservationConfig, TeamSearchConfig
from pokemon_codesign_env.team_designer import TeamDesigner

format_config = FormatConfig.from_showdown("gen7ou")
learner_team = TeamDesigner(TeamSearchConfig(format=format_config))
env = MaxBasePowerPlayerEnv(
    battle_format="gen7ou",
    opponent_team_format="gen7battlefactory",
    team=learner_team,
    observation_config=ObservationConfig(include_effects=False, include_stats=False),
    save_replays="runs/replays",
)
```

- `battle_format` changes the Showdown generation and ruleset.
- `opponent_team_format` changes the Showdown format used to generate random opponent teams.
- `observation_config` selects the visible state and slot limits.
- `showdown_path` selects a different local Showdown installation.
- `team` is required and sets the co-design agent's packed team or `TeamDesigner`.
- `opponent_team` overrides the automatically generated random opponent team.
- `opponent_kwargs` configures the built-in opponent.
- Other keyword arguments are forwarded to `SinglesEnv`, including `team`, `save_replays`, account and server configuration, logging, and timeouts.

All three presets use team-supplied formats so the co-design agent controls its embodiment. The built-in opponent policy and its random `TeamDesigner` are separate: `RandomPlayer`, `MaxBasePowerPlayer`, and `SimpleHeuristicsPlayer` choose battle actions, while the designer generates their teams.
