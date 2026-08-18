from dataclasses import dataclass, field
from enum import StrEnum

@dataclass
class PokemonParameters:
    """Parameters defining a single Pokemon."""

    # Without Defaults
    species: str
    moves: list[str]
    item: str | None
    nature: str | None
    ability: str | None
    happiness: int | None
    gender: str | None

    # With Defaults
    level: int = 100
    evs: dict[str, int] = field(default_factory=lambda: {"hp": 0, "atk": 0, "def": 0, "spa": 0, "spd": 0, "spe": 0})
    ivs: dict[str, int] = field(default_factory=lambda: {"hp": 0, "atk": 0, "def": 0, "spa": 0, "spd": 0, "spe": 0})

class AttributeMode(StrEnum):
    """How a team attribute is chosen during search."""

    DEFAULT = "default"
    RANDOM = "random"
    OPTIMIZED = "optimized"

class ItemMode(StrEnum):
    """Item selection mode; items may additionally be suppressed entirely."""

    DEFAULT = "default"
    RANDOM = "random"
    OPTIMIZED = "optimized"
    NONE = "none"

class GenderMode(StrEnum):
    """Gender selection mode; there is no format default to fall back on."""

    RANDOM = "random"
    OPTIMIZED = "optimized"

@dataclass(frozen=True, slots=True)
class TeamSearchConfig:
    """Parameters defining the team design search space."""

    # Numerical parameters
    team_size: int = 6
    ev_budget: int = 510
    ev_per_stat_cap: int = 252
    iv_max: int = 31
    iv_budget: int = 186
    level_mean: int = 50
    level_std: float = 0.0

    # What is being optimized?
    evs: AttributeMode = AttributeMode.DEFAULT
    ivs: AttributeMode = AttributeMode.DEFAULT
    moves: AttributeMode = AttributeMode.DEFAULT
    items: ItemMode = ItemMode.DEFAULT
    natures: AttributeMode = AttributeMode.DEFAULT
    abilities: AttributeMode = AttributeMode.DEFAULT
    happiness: AttributeMode = AttributeMode.DEFAULT
    gender: GenderMode = GenderMode.RANDOM

    # Restrictions on Pokemon selection (None = unrestricted)
    generations_allowed: tuple[int, ...] | None = None
    types_allowed: tuple[str, ...] | None = None

    def __post_init__(self) -> None:
        # Dataclasses do not coerce, so a bad string would otherwise survive as a str.
        for field_name, enum_type in _MODE_FIELDS.items():
            value = getattr(self, field_name)
            if not isinstance(value, enum_type):
                try:
                    coerced = enum_type(value)
                except ValueError:
                    allowed = ", ".join(repr(m.value) for m in enum_type)
                    raise ValueError(f"{field_name} must be one of {allowed}, got {value!r}") from None
                object.__setattr__(self, field_name, coerced)

        if not 1 <= self.team_size <= 6:
            raise ValueError(f"team_size must be 1-6, got {self.team_size}")
        if self.ev_budget < 0:
            raise ValueError(f"ev_budget must be >= 0, got {self.ev_budget}")
        if self.level_std < 0:
            raise ValueError(f"level_std must be >= 0, got {self.level_std}")
        if self.level_mean < 1 or self.level_mean > 100:
            raise ValueError(f"level_mean must be 1-100, got {self.level_mean}")
        if self.iv_budget < 0:
            raise ValueError(f"iv_budget must be >= 0, got {self.iv_budget}")
        if self.iv_max < 0 or self.iv_max > 31:
            raise ValueError(f"iv_max must be 0-31, got {self.iv_max}")


_MODE_FIELDS: dict[str, type[StrEnum]] = {
    "evs": AttributeMode,
    "ivs": AttributeMode,
    "moves": AttributeMode,
    "items": ItemMode,
    "natures": AttributeMode,
    "abilities": AttributeMode,
    "happiness": AttributeMode,
    "gender": GenderMode,
}
