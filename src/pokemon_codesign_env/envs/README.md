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

## Pokemon League environments

`PokemonLeagueEnv` provides one configurable League task for every generation. One episode contains all four League opponents followed by the Champion, each controlled by `SimpleHeuristicsPlayer`. Winning a battle advances directly to the next trainer; losing or defeating the Champion ends the episode. Each League format bans future-generation Pokemon, so `TeamDesigner.get_parameter_space("species")` automatically contains only Pokemon introduced in that generation or earlier. The same restriction applies to manually submitted teams during Showdown validation.

Install the packaged formats once before starting Showdown:

```bash
pokemon-codesign-install-showdown /path/to/pokemon-showdown
```

The installer adds `config/pokemon-codesign-formats.ts` through Showdown's supported custom-format loader and builds the server. No simulator source files are modified. An existing unrelated `config/custom-formats.ts` is never overwritten; import `Formats` from `./pokemon-codesign-formats` and spread it into that file's `Formats` array when prompted.

```python
from pokemon_codesign_env.configs import FormatConfig, TeamSearchConfig
from pokemon_codesign_env.envs import PokemonLeagueEnv, get_pokemon_league
from pokemon_codesign_env.team_designer import TeamDesigner

league = get_pokemon_league(4)
format_config = FormatConfig.from_showdown(league.battle_format)
learner_team = TeamDesigner(TeamSearchConfig(format=format_config, team_size=6))
env = PokemonLeagueEnv(team=learner_team, generation=4, difficulty=0)
```

Set `random_league=True` and omit `generation` to sample one League at the start of every episode. The selected League remains fixed for all five battles. Random-League episodes use the Generation 9 League format so every Pokemon is available and the Gymnasium spaces remain identical across episodes:

```python
format_config = FormatConfig.from_showdown(PokemonLeagueEnv.RANDOM_BATTLE_FORMAT)
learner_team = TeamDesigner(TeamSearchConfig(format=format_config, team_size=6))
env = PokemonLeagueEnv(team=learner_team, random_league=True, difficulty=0)
```

`difficulty=0` heals the designed team before every trainer. `difficulty=1` carries current HP, fainted Pokemon, non-volatile status, move PP, and lost held items into the next battle. The state is encoded in the standard packed-team Poké Ball metadata and restored by the custom format; this metadata field is therefore reserved for hard League tasks. Temporary battle effects such as stat changes, weather, volatile status, Mega Evolution, Dynamax, and Terastallization reset normally. The observation contains a `league` `MultiDiscrete` value with the current trainer index, number of trainers remaining, and difficulty, plus `league_generation`, which identifies the selected League.

`PokemonLeagueBattleEnv` remains available for experiments that need an isolated battle against one named trainer:

```python
from pokemon_codesign_env.envs import PokemonLeagueBattleEnv

env = PokemonLeagueBattleEnv(team=learner_team, generation=4, opponent="Cynthia")
```

The available presets are:

| Generation | Game | League opponents |
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

The trainer teams reproduce the selected games' species order, levels, moves, abilities, held items, natures, genders, EVs, and IVs. Generation I and II use the games' fixed trainer DVs converted to Showdown IVs and zero stat experience; fields that did not exist yet remain unset. Starter-dependent Champions use one documented representative team. Mega Evolution, Z-Moves, Gigantamax, and Terastallization are retained where applicable. Galar has no Elite Four, so its four final Champion Cup opponents occupy the equivalent slots.

The packed Showdown representation follows the standard stat order: HP, Attack, Defense, Special Attack, Special Defense, Speed. A game's implicit or uniform trainer values are still stored on every `LeaguePokemon`, so inspecting `get_pokemon_league(generation)` returns the complete opponent configuration without relying on random-battle fallbacks.

The configurations live in `envs/leagues/gen1.json` through `gen9.json`. Each file contains the League metadata, its ordered trainers, and every Pokemon field. `pokemon_league_env.py` parses these files into the immutable `PokemonLeague`, `LeagueTrainer`, and `LeaguePokemon` objects used by the environment.

The data comes from game-derived sources rather than competitive sets: the `pret` disassemblies for [Red/Blue](https://github.com/pret/pokered), [Crystal](https://github.com/pret/pokecrystal), and [Emerald](https://github.com/pret/pokeemerald); the [PokeFeur vanilla ROM exports](https://github.com/JonatanUHC/PokeFeur) for the later games; and the [Project Pokemon USUM trainer dump](https://projectpokemon.org/research/usum/trainers.txt) for Ultra Sun/Ultra Moon's per-stat IV and EV values. Values derived by the game engine, such as early-generation DVs and deterministic natures, are converted to their equivalent Showdown fields.
