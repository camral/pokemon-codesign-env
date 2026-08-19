import json
from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path

from poke_env.data import to_id_str
from poke_env.player import SimpleHeuristicsPlayer
from poke_env.teambuilder import Teambuilder, TeambuilderPokemon

from pokemon_codesign_env.envs.base import BaselineOpponentEnv
from pokemon_codesign_env.showdown import validate_team

StatSpread = tuple[int, int, int, int, int, int]


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
        return f"gen{self.generation}customgame"

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
    ):
        super().__init__()
        self.league = league
        self.trainer = trainer
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
            validate_team(self.league.battle_format, self._packed_team, self.showdown_path)
        return self._packed_team


class PokemonLeagueEnv(BaselineOpponentEnv):
    """Battle one configured Pokemon League trainer using the heuristic controller."""

    BATTLE_FORMAT = "gen9customgame"
    OPPONENT_TEAM_FORMAT = "gen9randombattle"
    OPPONENT_CLASS = SimpleHeuristicsPlayer

    def __init__(
        self,
        team,
        generation: int,
        opponent: str,
        showdown_path: Path | None = None,
        **env_kwargs,
    ):
        self.league = get_pokemon_league(generation)
        self.league_trainer = self.league.get_trainer(opponent)
        opponent_team = LeagueTeam(self.league, self.league_trainer, showdown_path)
        super().__init__(
            team=team,
            battle_format=self.league.battle_format,
            showdown_path=showdown_path,
            opponent_team=opponent_team,
            **env_kwargs,
        )
