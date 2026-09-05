"""Gymnasium spaces describing the embodiment design space of a team search.

`TeamSearchConfig` and `TeamDesigner` describe the searchable team parameters with plain
descriptor dictionaries. A codesign Gymnasium environment has to expose the same information
as a `gymnasium.Space` so that an embodiment designer can sample it, test membership, and
translate it to whatever representation its optimizer needs.

Three constraints have no equivalent in the stock spaces and are handled here:

* an EV or IV allocation is capped per stat *and* by a total budget,
* `moves`, `ability`, `gender` and `tera_type` are only legal for particular species, so the
  leaf spaces hold the union over every legal species and `TeamDesignSpace.contains` applies
  the species-conditional restriction,
* poke-env tracks Pokemon by fuzzy identifier matching, so a team whose species cannot be told
  apart produces corrupted observations even though Showdown accepts it.
"""

from functools import lru_cache

import numpy as np
from gymnasium import spaces
from gymnasium.spaces import Space
from poke_env.data import GenData
from poke_env.data.normalize import to_id_str

from pokemon_codesign_env.configs import TeamSearchConfig
from pokemon_codesign_env.showdown import get_parameter_values
from pokemon_codesign_env.team_designer import TeamDesigner


@lru_cache(maxsize=None)
def species_are_trackable(species: tuple[str, ...], generation: int) -> bool:
    """Return whether poke-env's fuzzy identifier matching can distinguish every Pokemon."""
    pokedex = GenData.from_gen(generation).pokedex
    base_species = [to_id_str(pokedex.get(to_id_str(name), {}).get("baseSpecies", name)) for name in species]
    for index, base_name in enumerate(base_species):
        for other_index, other_name in enumerate(species):
            if index != other_index and (base_name == to_id_str(other_name) or base_name in {to_id_str(part) for part in other_name.split("-")}):
                return False
    return True


class Categorical(Space):
    """One value drawn from an explicit ordered list of Pokemon Showdown values."""

    def __init__(self, values, seed=None):
        self.values = tuple(values)
        if not self.values:
            raise ValueError("Categorical requires at least one value")
        self._indices = {value: index for index, value in enumerate(self.values)}
        super().__init__((), None, seed)

    @property
    def is_np_flattenable(self) -> bool:
        return False

    @property
    def index_nvec(self) -> tuple[int, ...]:
        return (len(self.values),)

    def sample(self, mask=None, probability=None):
        return self.values[int(self.np_random.integers(len(self.values)))]

    def contains(self, x) -> bool:
        try:
            return x in self._indices
        except TypeError:
            return False

    def to_index(self, value) -> tuple[int, ...]:
        if not self.contains(value):
            raise ValueError(f"{value!r} is not a member of {self}")
        return (self._indices[value],)

    def from_index(self, indices):
        index = int(np.asarray(indices).ravel()[0])
        if not 0 <= index < len(self.values):
            raise ValueError(f"index {index} does not name a value of {self}")
        return self.values[index]

    def __repr__(self) -> str:
        return f"Categorical({len(self.values)} values)"

    def __eq__(self, other) -> bool:
        return isinstance(other, Categorical) and self.values == other.values


class IntegerParam(Space):
    """A bounded integer parameter that yields Python ints rather than NumPy scalars."""

    def __init__(self, minimum: int, maximum: int, seed=None):
        if maximum < minimum:
            raise ValueError(f"maximum must be at least minimum, got {minimum}-{maximum}")
        self.minimum = int(minimum)
        self.maximum = int(maximum)
        super().__init__((), None, seed)

    @property
    def is_np_flattenable(self) -> bool:
        return False

    @property
    def index_nvec(self) -> tuple[int, ...]:
        return (self.maximum - self.minimum + 1,)

    def sample(self, mask=None, probability=None) -> int:
        return int(self.np_random.integers(self.minimum, self.maximum + 1))

    def contains(self, x) -> bool:
        return type(x) is int and self.minimum <= x <= self.maximum

    def to_index(self, value) -> tuple[int, ...]:
        if not self.contains(value):
            raise ValueError(f"{value!r} is not a member of {self}")
        return (value - self.minimum,)

    def from_index(self, indices) -> int:
        index = int(np.asarray(indices).ravel()[0])
        if not 0 <= index < self.index_nvec[0]:
            raise ValueError(f"index {index} is outside {self}")
        return self.minimum + index

    def __repr__(self) -> str:
        return f"IntegerParam({self.minimum}, {self.maximum})"

    def __eq__(self, other) -> bool:
        return isinstance(other, IntegerParam) and (self.minimum, self.maximum) == (other.minimum, other.maximum)


class ChoiceSet(Space):
    """An ordered selection of distinct values, such as a Pokemon's move set."""

    def __init__(self, values, min_size: int, max_size: int, unique: bool = True, seed=None):
        self.values = tuple(values)
        self.min_size = int(min_size)
        self.max_size = int(max_size)
        self.unique = bool(unique)
        if not self.values:
            raise ValueError("ChoiceSet requires at least one value")
        if self.min_size < 0 or self.max_size < self.min_size:
            raise ValueError(f"invalid choice set size range {self.min_size}-{self.max_size}")
        if self.unique and self.min_size > len(self.values):
            raise ValueError(f"cannot choose {self.min_size} distinct values from {len(self.values)}")
        self._indices = {value: index for index, value in enumerate(self.values)}
        super().__init__((), None, seed)

    @property
    def is_np_flattenable(self) -> bool:
        return False

    @property
    def index_nvec(self) -> tuple[int, ...]:
        return (len(self.values),) * self.max_size

    def sample(self, mask=None, probability=None) -> list:
        size = min(self.max_size, len(self.values))
        indices = self.np_random.choice(len(self.values), size=size, replace=False)
        return [self.values[int(index)] for index in indices]

    def contains(self, x) -> bool:
        if not isinstance(x, list) or not self.min_size <= len(x) <= self.max_size:
            return False
        try:
            if self.unique and len(x) != len(set(x)):
                return False
            return all(value in self._indices for value in x)
        except TypeError:
            return False

    def to_index(self, value) -> tuple[int, ...]:
        if not self.contains(value):
            raise ValueError(f"{value!r} is not a member of {self}")
        # Padding by repeating the final choice round-trips, because from_index drops duplicates.
        indices = [self._indices[choice] for choice in value]
        return tuple(indices + [indices[-1]] * (self.max_size - len(indices)))

    def from_index(self, indices) -> list:
        chosen = []
        for raw in np.asarray(indices).ravel()[: self.max_size]:
            index = int(raw)
            if not 0 <= index < len(self.values):
                raise ValueError(f"index {index} does not name a value of {self}")
            value = self.values[index]
            # to_index pads a short selection by repeating its last choice, so a repeat is the
            # padding rather than a duplicate. TeamDesignSpace.from_index re-encodes the result to
            # confirm the whole point really denotes this selection.
            if value not in chosen:
                chosen.append(value)
        if len(chosen) < self.min_size:
            raise ValueError(f"indices name {len(chosen)} distinct values, fewer than the minimum {self.min_size}")
        return chosen

    def __repr__(self) -> str:
        return f"ChoiceSet({len(self.values)} values, {self.min_size}-{self.max_size})"

    def __eq__(self, other) -> bool:
        return (
            isinstance(other, ChoiceSet)
            and (self.values, self.min_size, self.max_size, self.unique) == (other.values, other.min_size, other.max_size, other.unique)
        )


class StatAllocation(Space):
    """A per-stat allocation bounded by a per-stat cap and a shared total budget."""

    def __init__(self, stats, minimum: int, max_per_stat: int, budget: int, seed=None):
        self.stats = tuple(stats)
        self.minimum = int(minimum)
        self.max_per_stat = int(max_per_stat)
        self.budget = int(budget)
        if not self.stats:
            raise ValueError("StatAllocation requires at least one stat")
        if self.max_per_stat < self.minimum:
            raise ValueError(f"max_per_stat must be at least minimum, got {self.minimum}-{self.max_per_stat}")
        if self.budget < self.minimum * len(self.stats):
            raise ValueError(f"budget {self.budget} cannot cover the minimum for {len(self.stats)} stats")
        super().__init__((), None, seed)

    @property
    def is_np_flattenable(self) -> bool:
        return False

    @property
    def index_nvec(self) -> tuple[int, ...]:
        return (self.max_per_stat - self.minimum + 1,) * len(self.stats)

    def sample(self, mask=None, probability=None) -> dict[str, int]:
        # Mirrors TeamDesigner.sample_parameter so sampled allocations always spend the budget.
        order = list(self.stats)
        self.np_random.shuffle(order)
        remaining_budget = self.budget
        values = {}
        for index, stat in enumerate(order):
            remaining_stats = len(order) - index - 1
            low = max(self.minimum, remaining_budget - self.max_per_stat * remaining_stats)
            high = min(self.max_per_stat, remaining_budget)
            values[stat] = int(self.np_random.integers(low, high + 1))
            remaining_budget -= values[stat]
        return {stat: values[stat] for stat in self.stats}

    def contains(self, x) -> bool:
        if not isinstance(x, dict) or set(x) != set(self.stats):
            return False
        if any(type(value) is not int for value in x.values()):
            return False
        if any(value < self.minimum or value > self.max_per_stat for value in x.values()):
            return False
        return sum(x.values()) <= self.budget

    def to_index(self, value) -> tuple[int, ...]:
        if not self.contains(value):
            raise ValueError(f"{value!r} is not a member of {self}")
        return tuple(value[stat] - self.minimum for stat in self.stats)

    def from_index(self, indices) -> dict[str, int]:
        raw = np.asarray(indices).ravel()[: len(self.stats)]
        values = []
        for entry in raw:
            index = int(entry)
            if not 0 <= index < self.index_nvec[0]:
                raise ValueError(f"index {index} is outside the per-stat range of {self}")
            values.append(self.minimum + index)
        if sum(values) > self.budget:
            raise ValueError(f"allocation sums to {sum(values)}, over the budget of {self.budget}")
        return {stat: values[index] for index, stat in enumerate(self.stats)}

    def __repr__(self) -> str:
        return f"StatAllocation({len(self.stats)} stats, max {self.max_per_stat}, budget {self.budget})"

    def __eq__(self, other) -> bool:
        return (
            isinstance(other, StatAllocation)
            and (self.stats, self.minimum, self.max_per_stat, self.budget) == (other.stats, other.minimum, other.max_per_stat, other.budget)
        )


def space_from_descriptor(descriptor: dict) -> Space:
    """Translate one TeamDesigner parameter descriptor into its Gymnasium space."""
    descriptor_type = descriptor["type"]
    if descriptor_type == "categorical":
        return Categorical(descriptor["values"])
    if descriptor_type == "integer":
        return IntegerParam(descriptor["min"], descriptor["max"])
    if descriptor_type == "choice_set":
        return ChoiceSet(descriptor["values"], descriptor["min_size"], descriptor["max_size"], descriptor["unique"])
    if descriptor_type == "stat_allocation":
        return StatAllocation(descriptor["stats"], descriptor["min"], descriptor["max_per_stat"], descriptor["budget"])
    raise ValueError(f"Unsupported parameter space type: {descriptor_type!r}")


class TeamDesignSpace(spaces.Dict):
    """The embodiment design space of a `TeamSearchConfig`.

    Members are exactly the dictionaries `TeamDesigner.set_optimizable_parameter_values` accepts,
    so a sampled or decoded embodiment can be handed straight to a designer.
    """

    DEPENDENT_PARAMETERS = ("moves", "ability", "gender", "tera_type")
    SAMPLE_ATTEMPTS = 100

    def __init__(self, search_config: TeamSearchConfig, seed=None):
        self.search_config = search_config
        descriptors = TeamDesigner(search_config, validate_teams=False).get_optimizable_parameter_space()
        # gymnasium.spaces.Dict orders keys alphabetically, which interleaves pokemon_10 and
        # pokemon_2 on large teams, so the codec uses its own numeric ordering throughout.
        self.pokemon_keys = tuple(sorted(descriptors, key=lambda name: int(name.rsplit("_", 1)[1])))
        super().__init__(
            {key: spaces.Dict({param: space_from_descriptor(descriptor) for param, descriptor in descriptors[key].items()}) for key in self.pokemon_keys},
            seed=seed,
        )
        self.parameter_keys = {key: tuple(self.spaces[key].spaces) for key in self.pokemon_keys}
        self.optimizes_species = all("species" in self.parameter_keys[key] for key in self.pokemon_keys)
        self._value_cache: dict[tuple[str, str | None], tuple] = {}
        self._dependent_spaces: dict[tuple[str, str], Space] = {}
        self._compilable: dict[str, bool] = {}
        self._flat_space: spaces.MultiDiscrete | None = None

    @property
    def generation(self) -> int:
        return self.search_config.format.generation

    def legal_values(self, param: str, species: str | None) -> tuple:
        """Return the values of a parameter that are legal for one species."""
        cache_key = (param, species)
        if cache_key not in self._value_cache:
            format_config = self.search_config.format
            self._value_cache[cache_key] = tuple(get_parameter_values(
                format_config.format_id, param, format_config.showdown_path, species,
                self.search_config.generations_allowed, self.search_config.types_allowed,
            ))
        return self._value_cache[cache_key]

    def dependent_space(self, pokemon: str, param: str, species: str) -> Space:
        """Return the species-restricted space for one dependent parameter."""
        cache_key = (param, species)
        if cache_key not in self._dependent_spaces:
            values = self.legal_values(param, species)
            union = self.spaces[pokemon][param]
            if isinstance(union, ChoiceSet):
                self._dependent_spaces[cache_key] = ChoiceSet(values, union.min_size, min(union.max_size, len(values)), union.unique)
            else:
                self._dependent_spaces[cache_key] = Categorical(values)
        return self._dependent_spaces[cache_key]

    def _dependent_parameters(self, pokemon: str) -> tuple[str, ...]:
        return tuple(param for param in self.parameter_keys[pokemon] if param in self.DEPENDENT_PARAMETERS)

    @property
    def required_domains(self) -> tuple[str, ...]:
        """Dependent parameters TeamDesigner must resolve for a species, optimized or not.

        `yield_team` draws moves and gender from a species-restricted domain whatever their
        mode, and reads a tera type whenever the format supports one, so a species with an
        empty domain for any of them cannot be compiled into a team even though every
        individual parameter value is legal. Abilities are exempt because their default
        resolution tolerates a species with no listed ability.
        """
        required = ["moves", "gender"]
        if self.search_config.abilities.value != "default":
            required.append("ability")
        if self.search_config.format.supports_tera_type:
            required.append("tera_type")
        return tuple(required)

    def species_is_compilable(self, species: str) -> bool:
        """Return whether TeamDesigner can resolve every dependent parameter for a species."""
        if species not in self._compilable:
            self._compilable[species] = all(self.legal_values(param, species) for param in self.required_domains)
        return self._compilable[species]

    def _species_is_usable(self, pokemon: str, species: str, chosen: list[str]) -> bool:
        """Reject species with an empty dependent domain or an untrackable teammate."""
        if not self.species_is_compilable(species):
            return False
        if any(not self.legal_values(param, species) for param in self._dependent_parameters(pokemon)):
            return False
        return species_are_trackable(tuple(chosen) + (species,), self.generation)

    def _resolve_species(self, pokemon: str, preferred: int, chosen: list[str]) -> str:
        """Return the species an index names, provided the team can still be built from it."""
        species_space = self.spaces[pokemon]["species"]
        candidate = species_space.from_index([preferred])
        if not self.species_is_compilable(candidate):
            raise ValueError(f"{pokemon}: {candidate} has no legal value for one of {self.required_domains}")
        if not species_are_trackable(tuple(chosen) + (candidate,), self.generation):
            raise ValueError(f"{pokemon}: {candidate} cannot be told apart from another Pokemon on the team")
        return candidate

    def sample(self, mask=None, probability=None) -> dict:
        embodiment = {}
        chosen_species: list[str] = []
        for pokemon in self.pokemon_keys:
            values = {}
            species = None
            if "species" in self.parameter_keys[pokemon]:
                species_space = self.spaces[pokemon]["species"]
                for _ in range(self.SAMPLE_ATTEMPTS):
                    candidate = species_space.sample()
                    if self._species_is_usable(pokemon, candidate, chosen_species):
                        species = candidate
                        break
                if species is None:
                    species = self._resolve_species(pokemon, int(self.np_random.integers(len(species_space.values))), chosen_species)
                chosen_species.append(species)
                values["species"] = species
            for param in self.parameter_keys[pokemon]:
                if param == "species":
                    continue
                if species is not None and param in self.DEPENDENT_PARAMETERS:
                    values[param] = self.dependent_space(pokemon, param, species).sample()
                else:
                    values[param] = self.spaces[pokemon][param].sample()
            embodiment[pokemon] = values
        return embodiment

    def contains(self, x) -> bool:
        if not isinstance(x, dict) or set(x) != set(self.pokemon_keys):
            return False
        for pokemon in self.pokemon_keys:
            values = x[pokemon]
            if not isinstance(values, dict) or set(values) != set(self.parameter_keys[pokemon]):
                return False
            for param, value in values.items():
                if not self.spaces[pokemon][param].contains(value):
                    return False
            species = values.get("species")
            if species is None:
                continue
            if not self.species_is_compilable(species):
                return False
            for param in self._dependent_parameters(pokemon):
                legal = self.legal_values(param, species)
                selected = values[param] if isinstance(values[param], list) else [values[param]]
                if any(choice not in legal for choice in selected):
                    return False
        return self.team_is_trackable(x)

    def _explain_dependent_illegality(self, embodiment: dict) -> None:
        """Name the species-parameter pairing that makes an embodiment illegal, if any.

        Every parameter is indexed against the union of its values, so a decoded point can pair a
        species with a move it cannot learn. Reporting which pairing failed is far more use to an
        embodiment designer than being told the point is not a member.
        """
        for pokemon in self.pokemon_keys:
            species = embodiment[pokemon].get("species")
            if species is None:
                continue
            for param in self._dependent_parameters(pokemon):
                legal = set(self.legal_values(param, species))
                chosen = embodiment[pokemon][param]
                for value in (chosen if isinstance(chosen, list) else [chosen]):
                    if value not in legal:
                        raise ValueError(f"{pokemon}: {species} has no legal {param} {value!r}")

    def team_is_trackable(self, embodiment: dict) -> bool:
        """Return whether poke-env can distinguish every Pokemon in an embodiment."""
        if not self.optimizes_species:
            return True
        return species_are_trackable(tuple(embodiment[pokemon]["species"] for pokemon in self.pokemon_keys), self.generation)

    @property
    def flat_space(self) -> spaces.MultiDiscrete:
        """A flat MultiDiscrete view for optimizers that need a fixed-width representation."""
        if self._flat_space is None:
            nvec = [width for pokemon in self.pokemon_keys for param in self.parameter_keys[pokemon] for width in self.spaces[pokemon][param].index_nvec]
            self._flat_space = spaces.MultiDiscrete(np.asarray(nvec, dtype=np.int64))
        return self._flat_space

    def to_index(self, embodiment: dict) -> np.ndarray:
        """Encode an embodiment as a point in `flat_space`."""
        if not self.contains(embodiment):
            raise ValueError("embodiment is not a member of the design space")
        indices = [
            index
            for pokemon in self.pokemon_keys
            for param in self.parameter_keys[pokemon]
            for index in self._to_index_for(pokemon, param, embodiment[pokemon])
        ]
        return np.asarray(indices, dtype=np.int64)

    def _to_index_for(self, pokemon: str, param: str, values: dict) -> tuple[int, ...]:
        # Every parameter, dependent or not, is indexed against the union of its values, so an
        # index always names the same move or ability whatever species sits beside it. Indexing
        # the species-restricted list instead would make the same index mean a different move for
        # every species, so changing a species would silently reinterpret its whole move set.
        return self.spaces[pokemon][param].to_index(values[param])

    def from_index(self, indices) -> dict:
        """Decode a point of `flat_space` into the embodiment it names.

        `flat_space` is a box and this space is a strict subset of it, because a box cannot say
        that EVs share a budget, that species must be distinguishable, or that a move must come
        from the pool of the species beside it. A point that satisfies none of those names no
        design, and raises rather than being adjusted into one: an embodiment designer has to be
        told its proposal was rejected, not handed a different team and a score for it.
        """
        flat = np.asarray(indices, dtype=np.int64).ravel()
        expected = len(self.flat_space.nvec)
        if flat.size != expected:
            raise ValueError(f"Expected {expected} indices, got {flat.size}")

        raw = {}
        cursor = 0
        for pokemon in self.pokemon_keys:
            raw[pokemon] = {}
            for param in self.parameter_keys[pokemon]:
                width = len(self.spaces[pokemon][param].index_nvec)
                raw[pokemon][param] = flat[cursor:cursor + width]
                cursor += width

        embodiment = {}
        chosen_species: list[str] = []
        for pokemon in self.pokemon_keys:
            values = {}
            species = None
            if "species" in self.parameter_keys[pokemon]:
                species = self._resolve_species(pokemon, int(raw[pokemon]["species"][0]), chosen_species)
                chosen_species.append(species)
                values["species"] = species
            for param in self.parameter_keys[pokemon]:
                if param == "species":
                    continue
                # Decoded from the union, so an index names the same value regardless of species.
                # A move the species cannot learn is a rejected pairing, which contains() reports.
                values[param] = self.spaces[pokemon][param].from_index(raw[pokemon][param])
            embodiment[pokemon] = values

        # The leaves police their own domains, but only the whole embodiment shows the constraints
        # that span parameters. Re-encoding proves the decode was faithful rather than merely
        # plausible: a point that does not come back unchanged named no design to begin with.
        self._explain_dependent_illegality(embodiment)
        if not self.contains(embodiment):
            raise ValueError("indices do not name a member of the design space")
        if not np.array_equal(self.to_index(embodiment), flat):
            raise ValueError("indices do not name this embodiment exactly")
        return embodiment

    def __repr__(self) -> str:
        return f"TeamDesignSpace({len(self.pokemon_keys)} Pokemon, {len(self.flat_space.nvec)} indices)"
