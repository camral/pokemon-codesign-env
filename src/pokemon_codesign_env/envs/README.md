# Baseline environments

These are the reusable baseline environments for `pokemon-codesign-env`. Each combines the shared human-visible observation pipeline with a built-in poke-env opponent. The presets differ only in opponent policy and Showdown format; observation and Gymnasium behavior come from the shared base environment.

| Environment | Opponent policy | Battle format | Random opponent-team source |
| --- | --- | --- | --- |
| `RandomPlayerEnv` | `RandomPlayer` | `gen4customgame` | `gen4randombattle` |
| `MaxBasePowerPlayerEnv` | `MaxBasePowerPlayer` | `gen6ou` | `gen6battlefactory` |
| `SimpleHeuristicsPlayerEnv` | `SimpleHeuristicsPlayer` | `gen9monotype` | `gen9monotyperandombattle` |

The co-design agent supplies its designed team through `team`. Unless `opponent_team` is supplied, Showdown generates a fresh random opponent team before every battle and validates it against the battle format.

## Usage

Start Pokemon Showdown, then create an environment like any Gymnasium environment:

```python
import numpy as np

from pokemon_codesign_env.configs import FormatConfig, TeamSearchConfig
from pokemon_codesign_env.envs import RandomPlayerEnv
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

Each observation is a Gymnasium `Dict` containing `categorical`, `continuous`, `binary`, and `action_mask`.

## Format configuration

The class attributes define each baseline's format:

```python
class MaxBasePowerPlayerEnv(BaselineOpponentEnv):
    BATTLE_FORMAT = "gen6ou"
    OPPONENT_TEAM_FORMAT = "gen6battlefactory"
    OPPONENT_CLASS = MaxBasePowerPlayer
```

- `BATTLE_FORMAT` is the team-supplied format used for the actual battle.
- `OPPONENT_TEAM_FORMAT` is a compatible Showdown format with a native random-team generator.
- `OPPONENT_CLASS` selects the fixed baseline control policy.

The format can be overridden at construction time when needed:

```python
format_config = FormatConfig.from_showdown("gen7ou")
learner_team = TeamDesigner(TeamSearchConfig(format=format_config))
env = MaxBasePowerPlayerEnv(team=learner_team, battle_format="gen7ou", opponent_team_format="gen7battlefactory")
```

`team` is required and accepts a packed team or `TeamDesigner`. `opponent_team` can replace automatic random generation. Remaining keyword arguments are forwarded to `SinglesEnv`; `observation_config` still customizes the shared observation pipeline without changing the baseline preset.
