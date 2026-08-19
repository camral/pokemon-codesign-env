import json
from dataclasses import dataclass
from enum import IntEnum
from importlib.resources import files
from pathlib import Path

import numpy as np
from gymnasium import Env, spaces
from poke_env.data import to_id_str
from poke_env.player import SimpleHeuristicsPlayer
from poke_env.teambuilder import Teambuilder, TeambuilderPokemon

from pokemon_codesign_env.envs.base import BaselineOpponentEnv
from pokemon_codesign_env.showdown import validate_team

StatSpread = tuple[int, int, int, int, int, int]
LEAGUE_STATE_PREFIX = "pcg1"
STATUS_CODES = {
    None: "0",
    "brn": "b",
    "frz": "f",
    "par": "a",
    "psn": "p",
    "slp": "s",
    "tox": "t",
}


@dataclass(frozen=True, slots=True)
class LeaguePokemon:
    """One Pokemon exactly as configured by its source game."""

    species: str
    level: int
    moves: tuple[str, ...]
    ability: str | None = None
    item: str | None = None
    nature: str | None = None
    evs: StatSpread = (0, 0, 0, 0, 0, 0)
    ivs: StatSpread = (31, 31, 31, 31, 31, 31)
    gender: str | None = None
    gmax: bool = False
    tera_type: str | None = None


@dataclass(frozen=True, slots=True)
class LeagueTrainer:
    """One Elite Four, Champion Cup, or Champion opponent."""

    name: str
    team: tuple[LeaguePokemon, ...]
    role: str = "elite_four"


class LeagueDifficulty(IntEnum):
    EASY = 0
    HARD = 1


@dataclass(frozen=True, slots=True)
class LeaguePokemonState:
    hp: int
    status: str | None
    move_pp: tuple[int, ...]
    item: str | None


@dataclass(frozen=True, slots=True)
class PokemonLeague:
    """The representative Pokemon League used for one generation."""

    generation: int
    region: str
    game: str
    trainers: tuple[LeagueTrainer, ...]
    note: str | None = None

    @property
    def battle_format(self) -> str:
        return f"gen{self.generation}pokemonleague"

    def get_trainer(self, name: str) -> LeagueTrainer:
        trainer_id = to_id_str(name)
        for trainer in self.trainers:
            if to_id_str(trainer.name) == trainer_id:
                return trainer
        available = ", ".join(trainer.name for trainer in self.trainers)
        raise ValueError(
            f"Unknown Generation {self.generation} League trainer {name!r}. "
            f"Available: {available}."
        )


def _load_pokemon(data: dict) -> LeaguePokemon:
    return LeaguePokemon(
        species=data["species"],
        level=data["level"],
        moves=tuple(data["moves"]),
        ability=data["ability"],
        item=data["item"],
        nature=data["nature"],
        evs=tuple(data["evs"]),
        ivs=tuple(data["ivs"]),
        gender=data["gender"],
        gmax=data["gmax"],
        tera_type=data["tera_type"],
    )


def _load_league(generation: int) -> PokemonLeague:
    path = files(__package__).joinpath("leagues", f"gen{generation}.json")
    data = json.loads(path.read_text(encoding="utf-8"))
    trainers = tuple(
        LeagueTrainer(
            name=trainer["name"],
            role=trainer["role"],
            team=tuple(_load_pokemon(pokemon) for pokemon in trainer["team"]),
        )
        for trainer in data["trainers"]
    )
    return PokemonLeague(
        generation=data["generation"],
        region=data["region"],
        game=data["game"],
        trainers=trainers,
        note=data["note"],
    )


POKEMON_LEAGUES = {generation: _load_league(generation) for generation in range(1, 10)}


def get_pokemon_league(generation: int) -> PokemonLeague:
    """Return the representative League configuration for a generation."""
    try:
        return POKEMON_LEAGUES[generation]
    except KeyError:
        raise ValueError(f"generation must be 1-9, got {generation}") from None


class LeagueTeam(Teambuilder):
    """Build a deterministic legal Showdown team for a League trainer."""

    def __init__(
        self,
        league: PokemonLeague,
        trainer: LeagueTrainer,
        showdown_path: Path | None = None,
        battle_format: str | None = None,
    ):
        super().__init__()
        self.league = league
        self.trainer = trainer
        self.battle_format = battle_format or league.battle_format
        self.showdown_path = (
            showdown_path or Path(__file__).resolve().parents[3] / "pokemon-showdown"
        )
        self._packed_team = None

    def yield_team(self) -> str:
        if self._packed_team is None:
            team = [
                TeambuilderPokemon(
                    species=pokemon.species,
                    item=pokemon.item,
                    ability=pokemon.ability,
                    moves=list(pokemon.moves),
                    nature=pokemon.nature,
                    evs=list(pokemon.evs),
                    ivs=list(pokemon.ivs),
                    gender=pokemon.gender,
                    level=pokemon.level,
                    gmax=pokemon.gmax,
                    tera_type=pokemon.tera_type,
                )
                for pokemon in self.trainer.team
            ]
            self._packed_team = self.join_team(team)
            validate_team(self.battle_format, self._packed_team, self.showdown_path)
        return self._packed_team


class PokemonLeagueBattleEnv(BaselineOpponentEnv):
    """Battle one configured Pokemon League trainer."""

    BATTLE_FORMAT = "gen9customgame"
    OPPONENT_TEAM_FORMAT = "gen9randombattle"
    OPPONENT_CLASS = SimpleHeuristicsPlayer

    def __init__(
        self,
        team,
        generation: int,
        opponent: str,
        showdown_path: Path | None = None,
        battle_format: str | None = None,
        **env_kwargs,
    ):
        self.league = get_pokemon_league(generation)
        self.league_trainer = self.league.get_trainer(opponent)
        battle_format = battle_format or self.league.battle_format
        opponent_team = LeagueTeam(
            self.league,
            self.league_trainer,
            showdown_path,
            battle_format,
        )
        super().__init__(
            team=team,
            battle_format=battle_format,
            showdown_path=showdown_path,
            opponent_team=opponent_team,
            **env_kwargs,
        )


def _resolve_team(team: str | Teambuilder) -> str:
    return team.yield_team() if isinstance(team, Teambuilder) else team


def _snapshot_team(battle) -> tuple[LeaguePokemonState, ...]:
    states = []
    for pokemon in battle.team.values():
        status = None
        if not pokemon.fainted and pokemon.status is not None:
            status = pokemon.status.name.lower()
        item = pokemon.item
        if item in {None, "unknown_item"}:
            item = None
        states.append(
            LeaguePokemonState(
                hp=pokemon.current_hp,
                status=status,
                move_pp=tuple(move.current_pp for move in pokemon.base_moves.values()),
                item=item,
            )
        )
    return tuple(states)


def _apply_team_state(team: str, states: tuple[LeaguePokemonState, ...]) -> str:
    packed_pokemon = team.split("]")
    if len(packed_pokemon) != len(states):
        raise ValueError(
            f"Persistent team contains {len(packed_pokemon)} Pokemon but state contains "
            f"{len(states)}."
        )

    restored_team = []
    for packed, state in zip(packed_pokemon, states, strict=True):
        fields = packed.split("|")
        if len(fields) != 12:
            raise ValueError("Cannot apply persistent state to an invalid packed team.")
        try:
            status_code = STATUS_CODES[state.status]
        except KeyError:
            raise ValueError(f"Cannot persist unsupported status {state.status!r}.") from None
        if not 0 <= state.hp < 36**3:
            raise ValueError(f"Cannot persist HP outside 0-{36**3 - 1}: {state.hp}")
        if any(not 0 <= pp < 36**2 for pp in state.move_pp):
            raise ValueError(f"Cannot persist move PP outside 0-{36**2 - 1}.")

        encoded_state = "".join(
            [
                LEAGUE_STATE_PREFIX,
                np.base_repr(state.hp, base=36).lower().zfill(3),
                status_code,
                *(np.base_repr(pp, base=36).lower().zfill(2) for pp in state.move_pp),
            ]
        )
        fields[2] = state.item or ""
        misc = fields[11].split(",")
        misc.extend([""] * (6 - len(misc)))
        misc[2] = encoded_state
        fields[11] = ",".join(misc[:6])
        restored_team.append((state.hp == 0, "|".join(fields)))
    restored_team.sort(key=lambda pokemon: pokemon[0])
    return "]".join(packed for _, packed in restored_team)


class PokemonLeagueEnv(Env):
    """Face every trainer in a fixed or randomly sampled League as one episode."""

    metadata = {"render_modes": ["human"]}
    RANDOM_BATTLE_FORMAT = "gen9pokemonleague"

    def __init__(
        self,
        team,
        generation: int | None = None,
        difficulty: int = 0,
        random_league: bool = False,
        showdown_path: Path | None = None,
        **env_kwargs,
    ):
        if team is None:
            raise ValueError("team must contain the co-design agent's designed team")
        if random_league and generation is not None:
            raise ValueError("generation must be omitted when random_league is enabled")
        if not random_league and generation is None:
            raise ValueError("generation is required when random_league is disabled")
        try:
            self.difficulty = LeagueDifficulty(difficulty)
        except ValueError:
            raise ValueError(f"difficulty must be 0 or 1, got {difficulty}") from None

        self.team = team
        self.random_league = random_league
        initial_generation = 9 if random_league else generation
        self.league = get_pokemon_league(initial_generation)
        self.battle_format = (
            self.RANDOM_BATTLE_FORMAT if random_league else self.league.battle_format
        )
        self.current_trainer_index = 0
        self._base_team = None
        self._current_team = None
        self._episode_active = False
        self._battle_env = PokemonLeagueBattleEnv(
            team=team,
            generation=initial_generation,
            opponent=self.league.trainers[0].name,
            showdown_path=showdown_path,
            battle_format=self.battle_format,
            **env_kwargs,
        )
        self.action_space = self._battle_env.action_space
        self.observation_space = spaces.Dict(
            {
                **self._battle_env.observation_space.spaces,
                "league": spaces.MultiDiscrete(
                    [len(self.league.trainers), len(self.league.trainers) + 1, 2]
                ),
                "league_generation": spaces.Discrete(10),
            }
        )

    @property
    def current_trainer(self) -> LeagueTrainer:
        return self.league.trainers[self.current_trainer_index]

    def _league_observation(self, observation):
        return {
            **observation,
            "league": np.asarray(
                [
                    self.current_trainer_index,
                    len(self.league.trainers) - self.current_trainer_index,
                    int(self.difficulty),
                ],
                dtype=np.int64,
            ),
            "league_generation": np.int64(self.league.generation),
        }

    def _league_info(self, info, **updates):
        return {
            **info,
            "generation": self.league.generation,
            "random_league": self.random_league,
            "difficulty": int(self.difficulty),
            "trainer_index": self.current_trainer_index,
            "trainer": self.current_trainer.name,
            "trainers_defeated": self.current_trainer_index,
            **updates,
        }

    def _prepare_battle(self, team: str):
        battle = self._battle_env.env
        battle.agent1.update_team(team)
        battle.agent2.update_team(
            LeagueTeam(
                self.league,
                self.current_trainer,
                self._battle_env.format_config.showdown_path,
                self.battle_format,
            )
        )

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if self.random_league:
            generation = int(self.np_random.integers(1, 10))
            self.league = get_pokemon_league(generation)
        self.current_trainer_index = 0
        self._base_team = _resolve_team(self.team)
        self._current_team = self._base_team
        self._prepare_battle(self._current_team)
        observation, info = self._battle_env.reset(seed=seed, options=options)
        self._episode_active = True
        return self._league_observation(observation), self._league_info(info)

    def step(self, action):
        if not self._episode_active:
            raise RuntimeError("reset() must be called before step().")

        observation, reward, terminated, truncated, info = self._battle_env.step(action)
        if truncated:
            self._episode_active = False
            return (
                self._league_observation(observation),
                reward,
                False,
                True,
                self._league_info(info, league_complete=False),
            )
        if not terminated:
            return (
                self._league_observation(observation),
                reward,
                False,
                False,
                self._league_info(info),
            )

        battle = self._battle_env.env.battle1
        won = bool(battle and battle.won)
        if not won or self.current_trainer_index == len(self.league.trainers) - 1:
            self._episode_active = False
            return (
                self._league_observation(observation),
                reward,
                True,
                False,
                self._league_info(
                    info,
                    battle_won=won,
                    league_complete=won,
                    trainers_defeated=self.current_trainer_index + int(won),
                ),
            )

        if self.difficulty is LeagueDifficulty.HARD:
            next_team = _apply_team_state(self._current_team, _snapshot_team(battle))
        else:
            next_team = self._base_team
        self._current_team = next_team

        defeated_trainer = self.current_trainer.name
        self.current_trainer_index += 1
        self._prepare_battle(next_team)
        observation, next_info = self._battle_env.reset()
        return (
            self._league_observation(observation),
            reward,
            False,
            False,
            self._league_info(
                {**info, **next_info},
                battle_won=True,
                defeated_trainer=defeated_trainer,
                league_complete=False,
            ),
        )

    def render(self):
        return self._battle_env.render()

    def close(self):
        self._episode_active = False
        self._battle_env.close()
