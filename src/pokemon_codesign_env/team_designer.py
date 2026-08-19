from copy import deepcopy
from dataclasses import fields

import numpy as np
from poke_env.teambuilder import Teambuilder, TeambuilderPokemon
from pokemon_codesign_env.configs import TeamSearchConfig, PokemonParameters
from pokemon_codesign_env.showdown import get_parameter_values, validate_team

class TeamDesigner(Teambuilder):
    """
    This class expresses the team design process as a search problem.
    """
    def __init__(self, search_config: TeamSearchConfig):
        super().__init__()
        self.search_config = search_config
        self.team: list[PokemonParameters] = []
        self.optimizable_attributes = {}
        self.optimizable_parameter_values = {}
        self.parameter_spaces = {}
        self.sampled_observations = {}
        self.rng = np.random.default_rng()
        
        self.define_optimizable_parameters()

    def define_optimizable_parameters(self):
        """
        This function uses the search_config to define the parameters that are used to navigate the search space for the team design process.
        :return:
        """

        optimizable_attributes = self.search_config.get_optimizable_attributes()

        for idx in range(self.search_config.team_size):
            self.optimizable_attributes[f"pokemon_{idx}"] = {}

            for attribute in optimizable_attributes:
                if attribute in {field.name for field in fields(PokemonParameters)}:
                    self.optimizable_attributes[f"pokemon_{idx}"][attribute] = getattr(self.search_config, attribute)
                else:
                    # Some attributes have different names in the search_config and PokemonParameters, so we need to map them accordingly.
                    if attribute == "items":
                        self.optimizable_attributes[f"pokemon_{idx}"]["item"] = getattr(self.search_config, attribute)
                    elif attribute == "natures":
                        self.optimizable_attributes[f"pokemon_{idx}"]["nature"] = getattr(self.search_config, attribute)
                    elif attribute == "abilities":
                        self.optimizable_attributes[f"pokemon_{idx}"]["ability"] = getattr(self.search_config, attribute)
                    elif attribute == "tera_types":
                        self.optimizable_attributes[f"pokemon_{idx}"]["tera_type"] = getattr(
                            self.search_config, attribute
                        )
                    else:
                        raise ValueError(f"Attribute {attribute} is not a valid attribute in PokemonParameters.")

    def get_parameter_space(self, param: str, species: str | None = None) -> dict:
        """
        The function returns the parameter space for a given parameter. The parameter space is defined by the search_config.
        :param param:
        :return:
        """
        cache_key = (param, species)
        if cache_key in self.parameter_spaces:
            return self.parameter_spaces[cache_key]

        if param == "evs":
            parameter_space = {
                "type": "stat_allocation",
                "stats": ["hp", "atk", "def", "spa", "spd", "spe"],
                "min": 0,
                "max_per_stat": self.search_config.ev_per_stat_cap,
                "budget": self.search_config.ev_budget,
            }
        elif param == "ivs":
            parameter_space = {
                "type": "stat_allocation",
                "stats": ["hp", "atk", "def", "spa", "spd", "spe"],
                "min": 0,
                "max_per_stat": self.search_config.iv_max,
                "budget": self.search_config.iv_budget,
            }
        elif param == "moves":
            parameter_space = {
                "type": "choice_set",
                "values": get_parameter_values(
                    self.search_config.format.format_id,
                    param,
                    self.search_config.format.showdown_path,
                    species,
                    self.search_config.generations_allowed,
                    self.search_config.types_allowed,
                ),
                "depends_on": "species",
                "min_size": 1,
                "max_size": self.search_config.format.max_move_count,
                "unique": True,
            }
        elif param in {"species", "item", "nature", "ability", "gender", "tera_type"}:
            parameter_space = {
                "type": "categorical",
                "values": get_parameter_values(
                    self.search_config.format.format_id,
                    param,
                    self.search_config.format.showdown_path,
                    species,
                    self.search_config.generations_allowed,
                    self.search_config.types_allowed,
                ),
            }
            if param in {"ability", "gender", "tera_type"}:
                parameter_space["depends_on"] = "species"
        elif param == "happiness":
            parameter_space = {
                "type": "integer",
                "min": 0,
                "max": 255,
                "default": 255,
            }
        else:
            raise ValueError(f"Parameter {param} does not have a defined search space.")

        self.parameter_spaces[cache_key] = parameter_space
        return parameter_space

    def get_optimizable_parameter_space(self) -> dict:
        """
        This function returns the parameter spaces used to navigate the team design search space.
        :return: A dictionary containing the spaces for parameters that can be optimized.
        """
        return {pokemon: {param: self.get_parameter_space(param) for param in parameters}
            for pokemon, parameters in self.optimizable_attributes.items() if parameters}

    def get_optimizable_parameter_values(self) -> dict:
        """Return the current concrete values for all optimizable parameters."""
        return deepcopy(self.optimizable_parameter_values)

    def set_optimizable_parameter_values(self, parameters: dict):
        """
        This function sets the concrete values used to navigate the team design search space.
        :param parameters: A dictionary containing values for all optimizable parameters.
        """
        if not isinstance(parameters, dict):
            raise TypeError("parameters must be a dictionary.")

        expected_pokemon = {pokemon for pokemon, pokemon_attributes in self.optimizable_attributes.items() if pokemon_attributes}

        received_pokemon = set(parameters)
        if received_pokemon != expected_pokemon:
            missing = sorted(expected_pokemon - received_pokemon)
            unexpected = sorted(received_pokemon - expected_pokemon)
            raise ValueError(f"Invalid Pokemon parameters. Missing: {missing}, unexpected: {unexpected}.")

        for pokemon, pokemon_parameters in parameters.items():
            if not isinstance(pokemon_parameters, dict):
                raise TypeError(f"Parameters for {pokemon} must be a dictionary.")

            expected_parameters = set(self.optimizable_attributes[pokemon])
            received_parameters = set(pokemon_parameters)
            if received_parameters != expected_parameters:
                missing = sorted(expected_parameters - received_parameters)
                unexpected = sorted(received_parameters - expected_parameters)
                raise ValueError(f"Invalid parameters for {pokemon}. Missing: {missing}, unexpected: {unexpected}.")

            species = pokemon_parameters.get("species")
            for param, value in pokemon_parameters.items():
                parameter_space = self.get_parameter_space(param, species)
                parameter_type = parameter_space["type"]

                if parameter_type == "categorical":
                    if value not in parameter_space["values"]:
                        raise ValueError(f"Invalid value for {pokemon}.{param}: {value!r}.")
                elif parameter_type == "integer":
                    if type(value) is not int or not parameter_space["min"] <= value <= parameter_space["max"]:
                        raise ValueError(f"Invalid value for {pokemon}.{param}: {value!r}.")
                elif parameter_type == "choice_set":
                    if not isinstance(value, list):
                        raise TypeError(f"{pokemon}.{param} must be a list.")
                    if not parameter_space["min_size"] <= len(value) <= parameter_space["max_size"]:
                        raise ValueError(f"Invalid number of values for {pokemon}.{param}: {len(value)}.")
                    if parameter_space["unique"] and len(value) != len(set(value)):
                        raise ValueError(f"Values for {pokemon}.{param} must be unique.")
                    if any(choice not in parameter_space["values"] for choice in value):
                        raise ValueError(f"Invalid value for {pokemon}.{param}: {value!r}.")
                elif parameter_type == "stat_allocation":
                    if not isinstance(value, dict):
                        raise TypeError(f"{pokemon}.{param} must be a dictionary.")
                    if set(value) != set(parameter_space["stats"]):
                        raise ValueError(f"{pokemon}.{param} must contain exactly {parameter_space['stats']}.")
                    if any(type(stat_value) is not int for stat_value in value.values()):
                        raise TypeError(f"Values for {pokemon}.{param} must be integers.")
                    if any(stat_value < parameter_space["min"] or stat_value > parameter_space["max_per_stat"] for stat_value in value.values()):
                        raise ValueError(f"A value for {pokemon}.{param} is outside its allowed range.")
                    if sum(value.values()) > parameter_space["budget"]:
                        raise ValueError(f"Values for {pokemon}.{param} exceed its budget.")
                else:
                    raise ValueError(f"Unsupported parameter space type: {parameter_type!r}.")

        self.optimizable_parameter_values = deepcopy(parameters)

    def sample_parameter(self, parameter_space: dict):
        """Sample a value from a parameter space."""
        parameter_type = parameter_space["type"]

        if parameter_type == "categorical":
            if not parameter_space["values"]:
                raise ValueError("Cannot sample from an empty categorical space.")
            idx = int(self.rng.integers(len(parameter_space["values"])))
            return parameter_space["values"][idx]
        elif parameter_type == "integer":
            return int(self.rng.integers(parameter_space["min"], parameter_space["max"] + 1))
        elif parameter_type == "choice_set":
            size = min(parameter_space["max_size"], len(parameter_space["values"]))
            if size < parameter_space["min_size"]:
                raise ValueError("Cannot sample enough values from the choice set.")
            indices = self.rng.choice(len(parameter_space["values"]), size=size, replace=False)
            return [parameter_space["values"][int(idx)] for idx in indices]
        elif parameter_type == "stat_allocation":
            stats = list(parameter_space["stats"])
            self.rng.shuffle(stats)
            remaining_budget = parameter_space["budget"]
            values = {}

            for idx, stat in enumerate(stats):
                remaining_stats = len(stats) - idx - 1
                min_value = max(parameter_space["min"], remaining_budget - parameter_space["max_per_stat"] * remaining_stats)
                max_value = min(parameter_space["max_per_stat"], remaining_budget)
                values[stat] = int(self.rng.integers(min_value, max_value + 1))
                remaining_budget -= values[stat]

            return {stat: values[stat] for stat in parameter_space["stats"]}
        else:
            raise ValueError(f"Unsupported parameter space type: {parameter_type!r}.")

    def get_parameter_value(self, pokemon: str, param: str, species: str | None = None):
        """Resolve an optimized, random, absent, or default parameter value."""
        config_name = {"item": "items", "nature": "natures", "ability": "abilities", "tera_type": "tera_types"}.get(param, param)
        mode = getattr(self.search_config, config_name)

        if mode.value == "optimized":
            return deepcopy(self.optimizable_parameter_values[pokemon][param])
        elif mode.value == "random":
            return self.sample_parameter(self.get_parameter_space(param, species))
        elif mode.value == "none":
            return None
        elif param == "item":
            return None
        elif param == "nature":
            return None if self.search_config.format.generation <= 2 else "Serious"
        elif param == "ability":
            abilities = self.get_parameter_space(param, species)["values"]
            return abilities[0] if abilities else None
        elif param == "happiness":
            return 255
        elif param == "ivs":
            return {stat: self.search_config.format.iv_max for stat in ("hp", "atk", "def", "spa", "spd", "spe")}
        elif param == "tera_type":
            if not self.search_config.format.supports_tera_type:
                return None
            tera_types = self.get_parameter_space(param, species)["values"]
            return tera_types[0]
        else:
            raise ValueError(f"Parameter {param} does not have a usable default.")

    def sample_observations(self) -> dict:
        """Sample species-independent default and random attributes for the next team."""
        observable_parameters = {
            "species": "species",
            "evs": "evs",
            "ivs": "ivs",
            "item": "items",
            "nature": "natures",
            "happiness": "happiness",
        }
        observations = {}

        for idx in range(self.search_config.team_size):
            pokemon_name = f"pokemon_{idx}"
            pokemon_observations = {}

            for param, config_name in observable_parameters.items():
                mode = getattr(self.search_config, config_name)
                if mode.value in {"default", "random"}:
                    pokemon_observations[param] = self.get_parameter_value(pokemon_name, param)

            level = round(self.rng.normal(self.search_config.level_mean, self.search_config.level_std))
            pokemon_observations["level"] = min(max(level, self.search_config.format.min_level), self.search_config.format.max_level)
            observations[pokemon_name] = pokemon_observations

        self.sampled_observations = observations
        return deepcopy(observations)

    def _resolve_observed_parameter(self, pokemon: str, param: str, observations: dict, species: str | None = None):
        """Use an observed value when available, otherwise resolve it normally."""
        if param in observations:
            return deepcopy(observations[param])
        return self.get_parameter_value(pokemon, param, species)

    def yield_team(self) -> str:
        """
        This function is what is called before each battle starts to define the team that is used.
        Therefore, this is the "generate" or "compile" function.

        In this function, the parameters that the class exposes will be consumed/evaluated to generate and return a team.
        :return: A string representing the generated team.
        """

        expected_pokemon = {pokemon for pokemon, attributes in self.optimizable_attributes.items() if attributes}
        if expected_pokemon != set(self.optimizable_parameter_values):
            raise ValueError("Optimizable parameters must be set before yielding a team.")

        if not self.sampled_observations:
            self.sample_observations()

        sampled_observations = self.sampled_observations
        resolved_team = []
        for idx in range(self.search_config.team_size):
            pokemon_name = f"pokemon_{idx}"
            pokemon_observations = sampled_observations[pokemon_name]

            species = self._resolve_observed_parameter(pokemon_name, "species", pokemon_observations)
            moves = self.get_parameter_value(pokemon_name, "moves", species)
            ability = self.get_parameter_value(pokemon_name, "ability", species)
            gender = self.get_parameter_value(pokemon_name, "gender", species)
            item = self._resolve_observed_parameter(pokemon_name, "item", pokemon_observations)
            nature = self._resolve_observed_parameter(pokemon_name, "nature", pokemon_observations)
            happiness = self._resolve_observed_parameter(pokemon_name, "happiness", pokemon_observations)
            evs = self._resolve_observed_parameter(pokemon_name, "evs", pokemon_observations)
            ivs = self._resolve_observed_parameter(pokemon_name, "ivs", pokemon_observations)
            tera_type = self.get_parameter_value(pokemon_name, "tera_type", species)

            level = pokemon_observations["level"]
            resolved_team.append(PokemonParameters(
                species=species,
                moves=moves,
                item=item,
                nature=nature,
                ability=ability,
                happiness=happiness,
                gender=gender,
                tera_type=tera_type,
                level=level,
                evs=evs,
                ivs=ivs,
            ))

        previous_team = self.team
        self.team = resolved_team
        try:
            packed_team = self.join_team([
                TeambuilderPokemon(
                    species=pokemon.species,
                    item=pokemon.item,
                    ability=pokemon.ability,
                    moves=pokemon.moves,
                    nature=pokemon.nature,
                    evs=[pokemon.evs[stat] for stat in ("hp", "atk", "def", "spa", "spd", "spe")],
                    gender=pokemon.gender,
                    ivs=[pokemon.ivs[stat] for stat in ("hp", "atk", "def", "spa", "spd", "spe")],
                    level=pokemon.level,
                    happiness=pokemon.happiness,
                    tera_type=pokemon.tera_type)
                for pokemon in self.team])
            validate_team(self.search_config.format.format_id, packed_team, self.search_config.format.showdown_path)
        except Exception:
            self.team = previous_team
            self.sampled_observations = {}
            raise

        self.sampled_observations = {}
        return packed_team


class BatchedTeamDesigner:
    """Coordinate an ordered batch of TeamDesigner instances."""

    def __init__(self, search_config: TeamSearchConfig = None, n_teams: int = None, team_designers: list[TeamDesigner] = None):
        if team_designers is None:
            if search_config is None or n_teams is None:
                raise ValueError("Either team_designers or both search_config and n_teams must be provided.")
            team_designers = [TeamDesigner(search_config) for _ in range(n_teams)]
            self.n_teams = n_teams
        else:
            self.n_teams = len(team_designers)

        if not isinstance(team_designers, list):
            raise TypeError("team_designers must be a list.")
        if not team_designers:
            raise ValueError("team_designers must contain at least one TeamDesigner.")
        if any(not isinstance(designer, TeamDesigner) for designer in team_designers):
            raise TypeError("team_designers must contain only TeamDesigner instances.")
        if len({id(designer) for designer in team_designers}) != len(team_designers):
            raise ValueError("Each batch position must have its own TeamDesigner instance.")

        self.team_designers = team_designers
        self.search_config = search_config

    def get_optimizable_parameter_space(self) -> list[dict]:
        """Return one optimizer-facing parameter space per TeamDesigner."""
        return [designer.get_optimizable_parameter_space() for designer in self.team_designers]

    def get_optimizable_parameter_values(self) -> list[dict]:
        """Return one concrete optimizer parameter dictionary per TeamDesigner."""
        return [designer.get_optimizable_parameter_values() for designer in self.team_designers]

    def set_optimizable_parameter_values(self, parameter_values: list[dict]) -> None:
        """Assign one concrete optimizer parameter dictionary to each TeamDesigner."""
        if not isinstance(parameter_values, list):
            raise TypeError("parameter_values must be a list.")
        if len(parameter_values) != len(self.team_designers):
            raise ValueError(f"Expected {len(self.team_designers)} parameter sets, got {len(parameter_values)}.")

        for designer, team_parameters in zip(self.team_designers, parameter_values, strict=True):
            designer.set_optimizable_parameter_values(team_parameters)

    def sample_observations(self) -> list[dict]:
        """Sample one observation dictionary per TeamDesigner."""
        return [designer.sample_observations() for designer in self.team_designers]
