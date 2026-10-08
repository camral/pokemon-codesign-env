# Environment classes

These are the environment classes behind `pce.make` and `pce.make_vec`, which also start a Showdown server for you. Use them directly only when you need more control.

## Baseline environments

The baseline environments share one observation and differ only in opponent and battle format.

| Class | Opponent | Battle format | Opponent team format |
| --- | --- | --- | --- |
| `RandomPlayerEnv` | `SwitchWeightedRandomPlayer` (switches 20% of the time when it could move) | `gen4customgame` | `gen4randombattle` |
| `MaxBasePowerPlayerEnv` | `MaxBasePowerPlayer` | `gen6customgame` | `gen6randombattle` |
| `SimpleHeuristicsPlayerEnv` | `SimpleHeuristicsPlayer` | `gen9customgame` | `gen9randombattle` |

Your team goes in `team`. Unless you pass `opponent_team`, Showdown generates a fresh random opponent team before every battle.

### Use

Used directly, the classes expect a Showdown server on port 8000:

```bash
node pokemon-showdown/pokemon-showdown start --no-security
```

Then:

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

Observations are a Gymnasium `Dict` of `categorical`, `continuous`, `binary` and `action_mask`.

### Format

Each baseline's format comes from its class attributes:

```python
class MaxBasePowerPlayerEnv(BaselineOpponentEnv):
    BATTLE_FORMAT = "gen6customgame"
    OPPONENT_TEAM_FORMAT = "gen6randombattle"
    OPPONENT_CLASS = MaxBasePowerPlayer
```

- `BATTLE_FORMAT`: the team-supplied format the battle is played in.
- `OPPONENT_TEAM_FORMAT`: a compatible format with a random-team generator, used for the opponent.
- `OPPONENT_CLASS`: the opponent's policy.

You can override the format at construction:

```python
format_config = FormatConfig.from_showdown("gen7ou")
learner_team = TeamDesigner(TeamSearchConfig(format=format_config))
env = MaxBasePowerPlayerEnv(team=learner_team, battle_format="gen7ou", opponent_team_format="gen7battlefactory")
```

`team` is required and takes a packed team or a `TeamDesigner`. `opponent_team` replaces the random opponent team, `observation_config` customizes the observation, and any other keyword arguments go to `SinglesEnv`.

## Pokémon League environments

`PokemonLeagueEnv` runs one generation's League: an episode faces the four Elite Four trainers and then the Champion, all played by `SimpleHeuristicsPlayer`. A win moves on to the next trainer; a loss, or beating the Champion, ends the episode.

Each League format bans Pokémon from later generations, so `TeamDesigner`'s species space only holds Pokémon up to that generation, and Showdown enforces the same rule on teams you submit.

Install the League formats once before starting Showdown:

```bash
pokemon-codesign-install-showdown /path/to/pokemon-showdown
```

This adds `config/pokemon-codesign-formats.ts` and builds the server without touching simulator files. If you already have a `config/custom-formats.ts`, it is left alone and you are told to import `Formats` from `./pokemon-codesign-formats` and spread it into that file's `Formats` array.

```python
from pokemon_codesign_env.configs import FormatConfig, TeamSearchConfig
from pokemon_codesign_env.envs import PokemonLeagueEnv, get_pokemon_league
from pokemon_codesign_env.team_designer import TeamDesigner

league = get_pokemon_league(4)
format_config = FormatConfig.from_showdown(league.battle_format)
learner_team = TeamDesigner(TeamSearchConfig(format=format_config, team_size=6))
env = PokemonLeagueEnv(team=learner_team, generation=4, difficulty=0)
```

### Random League

With `random_league=True` and no `generation`, each episode samples a League and keeps it for all five battles. These episodes use the Gen 9 League format, so every Pokémon is available and the spaces stay the same:

```python
format_config = FormatConfig.from_showdown(PokemonLeagueEnv.RANDOM_BATTLE_FORMAT)
learner_team = TeamDesigner(TeamSearchConfig(format=format_config, team_size=6))
env = PokemonLeagueEnv(team=learner_team, random_league=True, difficulty=0)
```

### Difficulty

| `difficulty` | Between trainers |
|---|---|
| `0` | The team is fully healed. |
| `1` | HP, fainted Pokémon, non-volatile status, move PP and lost held items carry over. |

Difficulty 1 stores this state in the packed team's Poké Ball field, so that field is reserved in hard League tasks. Temporary effects (stat changes, weather, volatile status, Mega Evolution, Dynamax, Terastallization) reset as usual.

### Observation

League observations add `league`, a `MultiDiscrete` of the current trainer index, trainers remaining and difficulty, and `league_generation`, the League's generation.

### One battle

For a single battle against one named trainer, use `PokemonLeagueBattleEnv`:

```python
from pokemon_codesign_env.envs import PokemonLeagueBattleEnv

env = PokemonLeagueBattleEnv(team=learner_team, generation=4, opponent="Cynthia")
```

### Leagues

| Generation | Game | Trainers |
| --- | --- | --- |
| 1 | Red/Blue | Lorelei, Bruno, Agatha, Lance, Blue |
| 2 | Crystal | Will, Koga, Bruno, Karen, Lance |
| 3 | Emerald | Sidney, Phoebe, Glacia, Drake, Wallace |
| 4 | Platinum | Aaron, Bertha, Flint, Lucian, Cynthia |
| 5 | Black/White rematch | Shauntal, Grimsley, Caitlin, Marshal, Alder |
| 6 | X/Y | Wikstrom, Malva, Drasna, Siebold, Diantha |
| 7 | Ultra Sun/Ultra Moon | Molayne, Olivia, Acerola, Kahili, Hau |
| 8 | Sword Champion Cup | Bede, Nessa, Bea, Raihan, Leon |
| 9 | Scarlet/Violet | Rika, Poppy, Larry, Hassel, Geeta |

### Trainer data

Trainer teams reproduce the games' species order, levels, moves, abilities, held items, natures, genders, EVs and IVs. Gen 1 and 2 convert the fixed trainer DVs to Showdown IVs with zero stat experience, and leave fields that didn't exist yet unset. Starter-dependent Champions use one representative team. Mega Evolution, Z-moves, Gigantamax and Terastallization are kept where they apply, and Galar, which has no Elite Four, uses its last four Champion Cup opponents instead.

Packed teams use the standard stat order (HP, Attack, Defense, Special Attack, Special Defense, Speed). Every `LeaguePokemon` stores all its values, implicit ones included, so `get_pokemon_league(generation)` returns the complete opponent configuration.

The data lives in `envs/leagues/gen1.json` to `gen9.json`, which `pokemon_league_env.py` parses into `PokemonLeague`, `LeagueTrainer` and `LeaguePokemon` objects.

Sources (game data, not competitive sets):
- the `pret` disassemblies for [Red/Blue](https://github.com/pret/pokered), [Crystal](https://github.com/pret/pokecrystal) and [Emerald](https://github.com/pret/pokeemerald);
- the [PokeFeur vanilla ROM exports](https://github.com/JonatanUHC/PokeFeur) for later games;
- the [Project Pokemon USUM trainer dump](https://projectpokemon.org/research/usum/trainers.txt) for Ultra Sun/Ultra Moon IVs and EVs.

Engine-derived values such as early DVs and deterministic natures are converted to their Showdown equivalents.
