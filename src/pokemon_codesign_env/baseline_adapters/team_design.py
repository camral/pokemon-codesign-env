"""Optimizer views derived from the canonical :class:`TeamDesigner` schema."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

import numpy as np

from pokemon_codesign_env.configs import (
    AttributeMode,
    FormatConfig,
    ItemMode,
    SelectionMode,
    TeamSearchConfig,
)
from pokemon_codesign_env.team_designer import TeamDesigner

STATS = ("hp", "atk", "def", "spa", "spd", "spe")
DEFAULT_LEVEL_MEAN = 70
DEFAULT_LEVEL_STD = 10.0
TEAM_DESIGN_SCHEMA_VERSION = 1


def make_all_attribute_search_config(
    format_config: FormatConfig,
    *,
    team_size: int = 6,
    generations_allowed: tuple[int, ...] | None = None,
    types_allowed: tuple[str, ...] | None = None,
) -> TeamSearchConfig:
    """Expose every supported battle-relevant attribute except level."""

    if generations_allowed is None:
        generations_allowed = tuple(range(1, format_config.generation + 1))
    return TeamSearchConfig(
        format=format_config,
        team_size=team_size,
        level_mean=DEFAULT_LEVEL_MEAN,
        level_std=DEFAULT_LEVEL_STD,
        species=SelectionMode.OPTIMIZED,
        evs=SelectionMode.OPTIMIZED,
        ivs=AttributeMode.OPTIMIZED,
        moves=SelectionMode.OPTIMIZED,
        items=ItemMode.OPTIMIZED,
        natures=AttributeMode.OPTIMIZED,
        abilities=AttributeMode.OPTIMIZED,
        happiness=AttributeMode.OPTIMIZED,
        gender=SelectionMode.OPTIMIZED,
        tera_types=(
            AttributeMode.OPTIMIZED if format_config.supports_tera_type else AttributeMode.DEFAULT
        ),
        generations_allowed=generations_allowed,
        types_allowed=types_allowed,
    )


@dataclass(frozen=True, slots=True)
class VectorField:
    """One stable slice in an optimizer-facing vector."""

    pokemon: str
    parameter: str
    width: int


class TeamDesignVectorAdapter:
    """Map bounded numeric vectors to canonical ``TeamDesigner`` parameters.

    Values are in ``[0, 1]``. Categorical values are represented by equal-width
    intervals. Species is decoded first so dependent parameters are selected
    from the chosen species' legal catalog.
    """

    def __init__(self, search_config: TeamSearchConfig, move_slots: int = 4):
        if move_slots < 1:
            raise ValueError("move_slots must be positive")
        self.search_config = search_config
        self.move_slots = move_slots
        self.designer = TeamDesigner(search_config, validate_teams=False)
        self.parameter_space = self.designer.get_optimizable_parameter_space()
        self.fields = self._build_fields()
        self.dimension = sum(field.width for field in self.fields)

    def _build_fields(self) -> tuple[VectorField, ...]:
        fields = []
        for pokemon, spaces in self.parameter_space.items():
            if "species" not in spaces:
                raise ValueError("The all-attribute vector requires optimized species")
            ordered = (
                "species",
                "moves",
                "item",
                "nature",
                "ability",
                "happiness",
                "gender",
                "tera_type",
                "evs",
                "ivs",
            )
            for parameter in ordered:
                if parameter not in spaces:
                    continue
                space = spaces[parameter]
                if space["type"] == "choice_set":
                    width = min(self.move_slots, int(space["max_size"]))
                elif space["type"] == "stat_allocation":
                    width = len(space["stats"])
                else:
                    width = 1
                fields.append(VectorField(pokemon, parameter, width))
        return tuple(fields)

    def decode(self, vector: np.ndarray) -> dict[str, dict]:
        """Decode one bounded vector into values accepted by ``TeamDesigner``."""

        values = np.asarray(vector, dtype=np.float64)
        if values.shape != (self.dimension,):
            raise ValueError(
                f"Expected a vector with shape ({self.dimension},), got {values.shape}"
            )
        if not np.all(np.isfinite(values)):
            raise ValueError("Team design vectors must contain only finite values")
        values = np.clip(values, 0.0, 1.0)
        result = {pokemon: {} for pokemon in self.parameter_space}
        offset = 0
        for field in self.fields:
            raw = values[offset : offset + field.width]
            offset += field.width
            species = result[field.pokemon].get("species")
            space = self.designer.get_parameter_space(
                field.parameter, self._dependency_species(field.parameter, species)
            )
            result[field.pokemon][field.parameter] = self._decode_field(raw, space)
        self.designer.set_optimizable_parameter_values(result)
        return result

    def _decode_field(self, raw: np.ndarray, space: dict):
        kind = space["type"]
        if kind == "categorical":
            return self._categorical(raw[0], space["values"])
        if kind == "integer":
            return int(round(space["min"] + float(raw[0]) * (space["max"] - space["min"])))
        if kind == "choice_set":
            remaining = list(space["values"])
            count = min(len(raw), int(space["max_size"]), len(remaining))
            if count < int(space["min_size"]):
                raise ValueError("A species does not have enough legal choices")
            selected = []
            for scalar in raw[:count]:
                choice = self._categorical(scalar, remaining)
                selected.append(choice)
                remaining.remove(choice)
            return selected
        if kind == "stat_allocation":
            rounded = np.rint(raw * int(space["max_per_stat"])).astype(np.int64)
            allocation = self._limit_allocation(rounded, int(space["budget"]))
            return {
                stat: int(value) for stat, value in zip(space["stats"], allocation, strict=True)
            }
        raise ValueError(f"Unsupported parameter-space type: {kind!r}")

    @staticmethod
    def _categorical(value: float, choices: list):
        if not choices:
            raise ValueError("Cannot decode an empty categorical space")
        index = min(int(float(value) * len(choices)), len(choices) - 1)
        return choices[index]

    @staticmethod
    def _limit_allocation(values: np.ndarray, budget: int) -> np.ndarray:
        total = int(values.sum())
        if total <= budget:
            return values
        scaled = values.astype(np.float64) * (budget / total)
        allocation = np.floor(scaled).astype(np.int64)
        remainder = budget - int(allocation.sum())
        fractions = scaled - allocation
        for index in np.argsort(-fractions, kind="stable")[:remainder]:
            allocation[index] += 1
        return allocation

    def encode(self, parameters: dict[str, dict]) -> np.ndarray:
        """Encode canonical parameter values at category-bin centres."""

        self.designer.set_optimizable_parameter_values(parameters)
        encoded = []
        for field in self.fields:
            value = parameters[field.pokemon][field.parameter]
            species = parameters[field.pokemon].get("species")
            space = self.designer.get_parameter_space(
                field.parameter, self._dependency_species(field.parameter, species)
            )
            encoded.extend(self._encode_field(value, space, field.width))
        return np.asarray(encoded, dtype=np.float64)

    def _encode_field(self, value, space: dict, width: int) -> list[float]:
        kind = space["type"]
        if kind == "categorical":
            return [self._categorical_centre(value, space["values"])]
        if kind == "integer":
            span = space["max"] - space["min"]
            return [0.0 if span == 0 else (value - space["min"]) / span]
        if kind == "choice_set":
            remaining = list(space["values"])
            result = []
            for choice in value:
                result.append(self._categorical_centre(choice, remaining))
                remaining.remove(choice)
            result.extend([0.0] * (width - len(result)))
            return result
        if kind == "stat_allocation":
            cap = int(space["max_per_stat"])
            return [0.0 if cap == 0 else value[stat] / cap for stat in space["stats"]]
        raise ValueError(f"Unsupported parameter-space type: {kind!r}")

    @staticmethod
    def _categorical_centre(value, choices: list) -> float:
        index = choices.index(value)
        return (index + 0.5) / len(choices)

    @staticmethod
    def _dependency_species(parameter: str, species: str | None) -> str | None:
        return species if parameter in {"moves", "ability", "gender", "tera_type"} else None

    def schema(self) -> dict:
        """Return the stable optimizer-vector layout for run artifacts."""

        return {
            "version": TEAM_DESIGN_SCHEMA_VERSION,
            "range": [0.0, 1.0],
            "dimension": self.dimension,
            "level_optimized": False,
            "level_mean": self.search_config.level_mean,
            "level_std": self.search_config.level_std,
            "move_slots": self.move_slots,
            "fields": [
                {"pokemon": field.pokemon, "parameter": field.parameter, "width": field.width}
                for field in self.fields
            ],
        }

    def serialize(self, parameters: dict[str, dict]) -> str:
        """Return a stable canonical serialization after validating a design."""

        self.designer.set_optimizable_parameter_values(parameters)
        return json.dumps(parameters, sort_keys=True, separators=(",", ":"))

    def candidate_id(self, parameters: dict[str, dict]) -> str:
        """Return a compact identity shared by optimizers, metrics, and artifacts."""

        return hashlib.sha256(self.serialize(parameters).encode()).hexdigest()
