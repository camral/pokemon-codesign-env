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
        self.optimizable_attributes = {}
        self.optimizable_parameter_values = {}
        self.parameter_spaces = {}
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
        elif param in {"species", "item", "nature", "ability", "gender"}:
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
            if param in {"ability", "gender"}:
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

    def get_optimizable_parameters(self) -> dict:
        """
        This function returns the parameters that are used to navigate the search space for the team design process.
        :return: A dictionary containing the parameters that can be optimized.
        """
        return {pokemon: {param: self.get_parameter_space(param) for param in parameters}
            for pokemon, parameters in self.optimizable_attributes.items() if parameters}

    def set_optimizable_parameters(self, parameters: dict):
        """
        This function sets the parameters that are used to navigate the search space for the team design process.
        :param parameters: A dictionary containing the parameters that can be optimized.
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
        config_name = {"item": "items", "nature": "natures", "ability": "abilities"}.get(param, param)
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
        else:
            raise ValueError(f"Parameter {param} does not have a usable default.")

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

        team = []
        for idx in range(self.search_config.team_size):
            pokemon_name = f"pokemon_{idx}"
            species = self.get_parameter_value(pokemon_name, "species")
            moves = self.get_parameter_value(pokemon_name, "moves", species)
            ability = self.get_parameter_value(pokemon_name, "ability", species)
            gender = self.get_parameter_value(pokemon_name, "gender", species)
            item = self.get_parameter_value(pokemon_name, "item", species)
            nature = self.get_parameter_value(pokemon_name, "nature", species)
            happiness = self.get_parameter_value(pokemon_name, "happiness", species)
            evs = self.get_parameter_value(pokemon_name, "evs", species)
            ivs = self.get_parameter_value(pokemon_name, "ivs", species)

            level = round(self.rng.normal(self.search_config.level_mean, self.search_config.level_std))
            level = min(max(level, self.search_config.format.min_level), self.search_config.format.max_level)
            pokemon = PokemonParameters(
                species=species,
                moves=moves,
                item=item,
                nature=nature,
                ability=ability,
                happiness=happiness,
                gender=gender,
                level=level,
                evs=evs,
                ivs=ivs,
            )
            team.append(TeambuilderPokemon(
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
            ))

        packed_team = self.join_team(team)
        validate_team(self.search_config.format.format_id, packed_team, self.search_config.format.showdown_path)
        return packed_team
