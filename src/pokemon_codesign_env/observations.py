from collections.abc import Sequence
from dataclasses import asdict

import numpy as np
from gymnasium import spaces
from poke_env.battle import (
    Effect,
    Field,
    MoveCategory,
    PokemonGender,
    PokemonType,
    SideCondition,
    Status,
    Weather,
)
from poke_env.battle.side_condition import STACKABLE_CONDITIONS
from poke_env.data import GenData
from poke_env.data.normalize import to_id_str

from pokemon_codesign_env.configs import FormatConfig, ObservationConfig
from pokemon_codesign_env.showdown import get_parameter_values

STATS = ("hp", "atk", "def", "spa", "spd", "spe")
BOOSTS = ("accuracy", "atk", "def", "evasion", "spa", "spd", "spe")


class HumanVisibleBattleEncoder:
    """Encode configurable human-visible state as structured Gym observations."""

    def __init__(
        self,
        species: Sequence[str],
        moves: Sequence[str],
        items: Sequence[str | None],
        abilities: Sequence[str | None],
        natures: Sequence[str | None],
        observation_config: ObservationConfig | None = None,
    ):
        self.config = observation_config or ObservationConfig()
        self.species = self._vocabulary(species)
        self.moves = self._vocabulary(moves)
        self.items = self._vocabulary(items)
        self.abilities = self._vocabulary(abilities)
        self.natures = self._vocabulary(natures)
        self.statuses = self._enum_vocabulary(Status)
        self.genders = self._enum_vocabulary(PokemonGender)
        self.types = self._enum_vocabulary(PokemonType)
        self.effects = self._enum_vocabulary(Effect)
        self.fields = self._enum_vocabulary(Field)
        self.side_conditions = self._enum_vocabulary(SideCondition)
        self.weather = self._enum_vocabulary(Weather)
        self.move_categories = self._enum_vocabulary(MoveCategory)

        self.species_ids = self._ids(self.species)
        self.move_ids = self._ids(self.moves)
        self.item_ids = self._ids(self.items)
        self.ability_ids = self._ids(self.abilities)
        self.nature_ids = self._ids(self.natures)
        self.status_ids = self._ids(self.statuses)
        self.gender_ids = self._ids(self.genders)
        self.type_ids = self._ids(self.types)
        self.effect_ids = self._ids(self.effects)
        self.field_ids = self._ids(self.fields)
        self.side_condition_ids = self._ids(self.side_conditions)
        self.weather_ids = self._ids(self.weather)
        self.move_category_ids = self._ids(self.move_categories)

        self._feature_names = {"categorical": [], "continuous": [], "binary": []}
        self._categorical_highs: list[int] = []
        self._continuous_lows: list[float] = []
        self._continuous_highs: list[float] = []
        self._build_feature_schema()
        if not any(self._feature_names.values()):
            raise ValueError("ObservationConfig must enable at least one observation feature.")

        self.feature_names = tuple(
            name for names in self._feature_names.values() for name in names
        )
        self.feature_indices = {
            name: (group, idx)
            for group, names in self._feature_names.items()
            for idx, name in enumerate(names)
        }
        self.observation_space = self._build_observation_space()

    @classmethod
    def from_showdown(
        cls,
        format_config: FormatConfig,
        generations_allowed: tuple[int, ...] | None = None,
        types_allowed: tuple[str, ...] | None = None,
        observation_config: ObservationConfig | None = None,
    ) -> "HumanVisibleBattleEncoder":
        """Build deterministic ID tables from the configured Showdown format."""

        def values(parameter: str) -> list[str | None]:
            return get_parameter_values(
                format_config.format_id,
                parameter,
                format_config.showdown_path,
                generations_allowed=generations_allowed,
                types_allowed=types_allowed,
            )

        return cls(
            species=values("species"),
            moves=values("moves"),
            items=values("item"),
            abilities=values("ability"),
            natures=values("nature"),
            observation_config=observation_config,
        )

    @staticmethod
    def _vocabulary(values: Sequence[str | None]) -> tuple[str, ...]:
        return tuple(sorted({to_id_str(value) for value in values if value is not None}))

    @staticmethod
    def _enum_vocabulary(enum_type) -> tuple[str, ...]:
        return tuple(sorted(member.name.lower() for member in enum_type))

    @staticmethod
    def _ids(values: Sequence[str]) -> dict[str, int]:
        return {value: idx for idx, value in enumerate(values, start=1)}

    def _categorical(self, name: str, maximum_id: int) -> None:
        self._feature_names["categorical"].append(name)
        self._categorical_highs.append(maximum_id)

    def _continuous(self, name: str, low: float = 0.0, high: float = 1.0) -> None:
        self._feature_names["continuous"].append(name)
        self._continuous_lows.append(low)
        self._continuous_highs.append(high)

    def _binary(self, name: str) -> None:
        self._feature_names["binary"].append(name)

    def _build_feature_schema(self) -> None:
        if self.config.include_turn:
            self._continuous("battle.turn")
        if self.config.include_mechanics:
            for name in (
                "force_switch",
                "trapped",
                "wait",
                "team_preview",
                "can_mega_evolve",
                "can_z_move",
                "can_dynamax",
                "can_terastallize",
            ):
                self._binary(f"battle.{name}")
        if self.config.include_weather:
            self._categorical("battle.weather_id", len(self.weather))
        if self.config.include_fields:
            for slot in range(self.config.max_fields):
                self._categorical(f"battle.field_id.{slot}", len(self.fields))
        if self.config.include_side_conditions:
            for side in ("own_side", "opponent_side"):
                for slot in range(self.config.max_side_conditions):
                    self._categorical(
                        f"battle.{side}.condition_id.{slot}",
                        len(self.side_conditions),
                    )
                    self._continuous(f"battle.{side}.condition_value.{slot}")
        if self.config.include_available_moves:
            for slot in range(self.config.move_slots):
                prefix = f"available_move.{slot}"
                self._categorical(f"{prefix}.move_id", len(self.moves))
                self._continuous(f"{prefix}.base_power")
                self._continuous(f"{prefix}.accuracy")
                self._continuous(f"{prefix}.pp_fraction")
                self._continuous(f"{prefix}.priority", low=-1.0)
                self._categorical(f"{prefix}.category_id", len(self.move_categories))
                self._categorical(f"{prefix}.type_id", len(self.types))

        for side, included in (
            ("own_team", self.config.include_own_team),
            ("opponent_team", self.config.include_opponent_team),
        ):
            if included:
                for slot in range(self.config.team_size):
                    self._append_pokemon_schema(f"{side}.{slot}")

    def _append_pokemon_schema(self, prefix: str) -> None:
        if self.config.include_presence:
            self._binary(f"{prefix}.present")
            self._binary(f"{prefix}.revealed")
        if self.config.include_identity:
            self._continuous(f"{prefix}.level")
            self._binary(f"{prefix}.shiny")
            self._categorical(f"{prefix}.species_id", len(self.species))
            self._categorical(f"{prefix}.type_id.0", len(self.types))
            self._categorical(f"{prefix}.type_id.1", len(self.types))
            self._categorical(f"{prefix}.gender_id", len(self.genders))
        if self.config.include_health:
            self._binary(f"{prefix}.active")
            self._binary(f"{prefix}.fainted")
            self._continuous(f"{prefix}.hp_fraction")
        if self.config.include_status:
            self._categorical(f"{prefix}.status_id", len(self.statuses))
            self._continuous(f"{prefix}.status_counter")
        if self.config.include_item:
            self._categorical(f"{prefix}.item_id", len(self.items))
        if self.config.include_ability:
            self._categorical(f"{prefix}.ability_id", len(self.abilities))
        if self.config.include_nature:
            self._categorical(f"{prefix}.nature_id", len(self.natures))
        if self.config.include_moves:
            self._categorical(f"{prefix}.last_move_id", len(self.moves))
            for slot in range(self.config.move_slots):
                self._categorical(f"{prefix}.move_id.{slot}", len(self.moves))
            for slot in range(self.config.move_slots):
                self._continuous(f"{prefix}.move_pp.{slot}")
        if self.config.include_action_state:
            self._binary(f"{prefix}.first_turn")
            self._binary(f"{prefix}.must_recharge")
            self._binary(f"{prefix}.preparing")
        if self.config.include_boosts:
            for boost in BOOSTS:
                self._continuous(f"{prefix}.boost.{boost}", low=-1.0)
        if self.config.include_stats:
            for stat in STATS:
                self._continuous(f"{prefix}.stats.{stat}")
        if self.config.include_effects:
            for slot in range(self.config.max_effects):
                self._categorical(f"{prefix}.effect_id.{slot}", len(self.effects))

    def _build_observation_space(self) -> spaces.Dict:
        observation_spaces = {}
        if self._feature_names["categorical"]:
            observation_spaces["categorical"] = spaces.MultiDiscrete(
                np.array(self._categorical_highs, dtype=np.int64) + 1,
                dtype=np.int64,
            )
        if self._feature_names["continuous"]:
            observation_spaces["continuous"] = spaces.Box(
                low=np.array(self._continuous_lows, dtype=np.float32),
                high=np.array(self._continuous_highs, dtype=np.float32),
                dtype=np.float32,
            )
        if self._feature_names["binary"]:
            observation_spaces["binary"] = spaces.MultiBinary(
                len(self._feature_names["binary"])
            )
        return spaces.Dict(observation_spaces)

    @staticmethod
    def _normalized(value: float, scale: float) -> float:
        return float(np.clip(value / scale, -1.0, 1.0))

    @staticmethod
    def _enum_id(value, ids: dict[str, int]) -> int:
        return ids.get(value.name.lower(), 0) if value is not None else 0

    @staticmethod
    def _string_id(value: str | None, ids: dict[str, int]) -> int:
        return ids.get(to_id_str(value), 0) if value is not None else 0

    @staticmethod
    def _padded(values: Sequence, size: int, padding=0) -> list:
        values = list(values)[:size]
        return values + [padding] * (size - len(values))

    def _set(self, state: dict[str, float], name: str, value) -> None:
        if name in self.feature_indices:
            state[name] = value

    def _encode_available_move(self, move, slot: int, state: dict[str, float]) -> None:
        if move is None:
            return
        prefix = f"available_move.{slot}"
        self._set(state, f"{prefix}.move_id", self._string_id(move.id, self.move_ids))
        self._set(state, f"{prefix}.base_power", self._normalized(move.base_power, 300))
        self._set(state, f"{prefix}.accuracy", float(move.accuracy))
        self._set(
            state,
            f"{prefix}.pp_fraction",
            self._normalized(move.current_pp, max(1, move.max_pp)),
        )
        move_entry = getattr(move, "entry", None)
        priority = move_entry.get("priority", 0) if move_entry is not None else move.priority
        self._set(state, f"{prefix}.priority", self._normalized(priority, 7))
        self._set(
            state,
            f"{prefix}.category_id",
            self._enum_id(move.category, self.move_category_ids),
        )
        self._set(state, f"{prefix}.type_id", self._enum_id(move.type, self.type_ids))

    def _encode_pokemon(
        self,
        pokemon,
        own: bool,
        prefix: str,
        state: dict[str, float],
    ) -> None:
        if pokemon is None:
            return
        details_visible = own or pokemon.revealed

        if self.config.include_presence:
            self._set(state, f"{prefix}.present", 1)
            self._set(state, f"{prefix}.revealed", int(details_visible))
        if self.config.include_identity:
            self._set(state, f"{prefix}.level", self._normalized(pokemon.level, 100))
            self._set(state, f"{prefix}.shiny", int(pokemon.shiny))
            self._set(
                state,
                f"{prefix}.species_id",
                self._string_id(pokemon.species, self.species_ids),
            )
            for slot, type_ in enumerate(self._padded(pokemon.types, 2, None)):
                self._set(
                    state,
                    f"{prefix}.type_id.{slot}",
                    self._enum_id(type_, self.type_ids),
                )
            self._set(
                state,
                f"{prefix}.gender_id",
                self._enum_id(pokemon.gender, self.gender_ids),
            )

        if details_visible:
            if self.config.include_health:
                self._set(state, f"{prefix}.active", int(bool(pokemon.active)))
                self._set(state, f"{prefix}.fainted", int(pokemon.fainted))
                self._set(
                    state,
                    f"{prefix}.hp_fraction",
                    float(pokemon.current_hp_fraction),
                )
            if self.config.include_status:
                self._set(
                    state,
                    f"{prefix}.status_id",
                    self._enum_id(pokemon.status, self.status_ids),
                )
                self._set(
                    state,
                    f"{prefix}.status_counter",
                    self._normalized(pokemon.status_counter, 16),
                )
            if self.config.include_item:
                item = pokemon.item
                if item == GenData.from_gen(pokemon.gen).UNKNOWN_ITEM:
                    item = None
                self._set(
                    state,
                    f"{prefix}.item_id",
                    self._string_id(item, self.item_ids),
                )
            if self.config.include_ability:
                self._set(
                    state,
                    f"{prefix}.ability_id",
                    self._string_id(pokemon.ability, self.ability_ids),
                )
            if self.config.include_moves:
                last_move = pokemon.last_move
                self._set(
                    state,
                    f"{prefix}.last_move_id",
                    self._string_id(last_move.id, self.move_ids)
                    if last_move is not None
                    else 0,
                )
                moves = self._padded(
                    list(pokemon.moves.values()), self.config.move_slots, None
                )
                for slot, move in enumerate(moves):
                    if move is not None:
                        self._set(
                            state,
                            f"{prefix}.move_id.{slot}",
                            self._string_id(move.id, self.move_ids),
                        )
                        self._set(
                            state,
                            f"{prefix}.move_pp.{slot}",
                            self._normalized(move.current_pp, max(1, move.max_pp)),
                        )
            if self.config.include_action_state:
                self._set(state, f"{prefix}.first_turn", int(pokemon.first_turn))
                self._set(
                    state,
                    f"{prefix}.must_recharge",
                    int(pokemon.must_recharge),
                )
                self._set(state, f"{prefix}.preparing", int(pokemon.preparing))
            if self.config.include_boosts:
                for boost in BOOSTS:
                    self._set(
                        state,
                        f"{prefix}.boost.{boost}",
                        self._normalized(pokemon.boosts.get(boost, 0), 6),
                    )
            if self.config.include_effects:
                effects = sorted(
                    self._enum_id(effect, self.effect_ids) for effect in pokemon.effects
                )
                padded_effects = self._padded(effects, self.config.max_effects)
                for slot, effect_id in enumerate(padded_effects):
                    self._set(state, f"{prefix}.effect_id.{slot}", effect_id)

        if own and self.config.include_nature:
            self._set(
                state,
                f"{prefix}.nature_id",
                self._string_id(pokemon.nature, self.nature_ids),
            )
        if own and self.config.include_stats:
            for stat in STATS:
                self._set(
                    state,
                    f"{prefix}.stats.{stat}",
                    self._normalized(pokemon.stats.get(stat) or 0, 1000),
                )

    def _encode_side_conditions(
        self,
        conditions: dict,
        side: str,
        state: dict[str, float],
    ) -> None:
        ordered = sorted(conditions, key=lambda condition: condition.name)
        for slot, condition in enumerate(ordered[: self.config.max_side_conditions]):
            maximum = STACKABLE_CONDITIONS.get(condition)
            value = (
                self._normalized(conditions[condition], maximum) if maximum is not None else 1.0
            )
            self._set(
                state,
                f"battle.{side}.condition_id.{slot}",
                self._enum_id(condition, self.side_condition_ids),
            )
            self._set(state, f"battle.{side}.condition_value.{slot}", value)

    def _assemble(self, state: dict[str, float]) -> dict[str, np.ndarray]:
        observation = {}
        if self._feature_names["categorical"]:
            observation["categorical"] = np.array(
                [state.get(name, 0) for name in self._feature_names["categorical"]],
                dtype=np.int64,
            )
        if self._feature_names["continuous"]:
            observation["continuous"] = np.array(
                [state.get(name, 0.0) for name in self._feature_names["continuous"]],
                dtype=np.float32,
            )
        if self._feature_names["binary"]:
            observation["binary"] = np.array(
                [state.get(name, 0) for name in self._feature_names["binary"]],
                dtype=np.int8,
            )
        return observation

    def get_feature(self, observation: dict[str, np.ndarray], name: str):
        """Return one named feature from a structured observation."""
        group, index = self.feature_indices[name]
        return observation[group][index]

    def get_schema(self) -> dict:
        """Return the exact structured spaces and deterministic ID mappings."""
        return {
            "encoder": type(self).__name__,
            "observation_config": asdict(self.config),
            "feature_count": len(self.feature_names),
            "feature_names": self._feature_names,
            "categorical_nvec": [high + 1 for high in self._categorical_highs],
            "continuous_low": self._continuous_lows,
            "continuous_high": self._continuous_highs,
            "id_mappings": {
                "species": [None, *self.species],
                "moves": [None, *self.moves],
                "items": [None, *self.items],
                "abilities": [None, *self.abilities],
                "natures": [None, *self.natures],
                "statuses": [None, *self.statuses],
                "genders": [None, *self.genders],
                "types": [None, *self.types],
                "effects": [None, *self.effects],
                "fields": [None, *self.fields],
                "side_conditions": [None, *self.side_conditions],
                "weather": [None, *self.weather],
                "move_categories": [None, *self.move_categories],
            },
        }

    def encode(self, battle) -> dict[str, np.ndarray]:
        """Return configured state without simulator-private opponent information."""
        state: dict[str, float] = {}
        if self.config.include_turn:
            self._set(state, "battle.turn", self._normalized(battle.turn, 100))
        if self.config.include_mechanics:
            for name, value in (
                ("force_switch", battle.force_switch),
                ("trapped", battle.trapped),
                ("wait", battle.wait),
                ("team_preview", battle.teampreview),
                ("can_mega_evolve", battle.can_mega_evolve),
                ("can_z_move", battle.can_z_move),
                ("can_dynamax", battle.can_dynamax),
                ("can_terastallize", battle.can_tera),
            ):
                self._set(state, f"battle.{name}", int(bool(value)))
        if self.config.include_weather:
            weather = next(iter(battle.weather), None)
            self._set(
                state,
                "battle.weather_id",
                self._enum_id(weather, self.weather_ids),
            )
        if self.config.include_fields:
            fields = sorted(
                self._enum_id(field, self.field_ids) for field in battle.fields
            )
            for slot, field_id in enumerate(
                self._padded(fields, self.config.max_fields)
            ):
                self._set(state, f"battle.field_id.{slot}", field_id)
        if self.config.include_side_conditions:
            self._encode_side_conditions(battle.side_conditions, "own_side", state)
            self._encode_side_conditions(
                battle.opponent_side_conditions, "opponent_side", state
            )
        if self.config.include_available_moves:
            moves = self._padded(
                list(battle.available_moves), self.config.move_slots, None
            )
            for slot, move in enumerate(moves):
                self._encode_available_move(move, slot, state)
        if self.config.include_own_team:
            own_team = self._padded(
                list(battle.team.values()), self.config.team_size, None
            )
            for slot, pokemon in enumerate(own_team):
                self._encode_pokemon(pokemon, True, f"own_team.{slot}", state)
        if self.config.include_opponent_team:
            opponent_team = self._padded(
                list(battle.opponent_team.values()), self.config.team_size, None
            )
            for slot, pokemon in enumerate(opponent_team):
                self._encode_pokemon(pokemon, False, f"opponent_team.{slot}", state)

        observation = self._assemble(state)
        if not self.observation_space.contains(observation):
            raise RuntimeError("Encoded observation does not match its configured Gym space.")
        return observation
