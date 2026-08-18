from dataclasses import fields

import numpy as np
from poke_env.teambuilder import Teambuilder
from pokemon_codesign_env.configs import TeamSearchConfig, PokemonParameters
from pokemon_codesign_env.showdown import get_parameter_values

class TeamDesigner(Teambuilder):
    """
    This class expresses the team design process as a search problem.
    """
    def __init__(self, search_config: TeamSearchConfig):
        super().__init__()
        self.search_config = search_config
        self.optimizable_parameters = {}
        self.parameter_spaces = {}
        
        self.define_optimizable_parameters()

    def define_optimizable_parameters(self):
        """
        This function uses the search_config to define the parameters that are used to navigate the search space for the team design process.
        :return:
        """

        optimizable_attributes = self.search_config.get_optimizable_attributes()

        for idx in range(self.search_config.team_size):
            self.optimizable_parameters[f"pokemon_{idx}"] = {}

            for attribute in optimizable_attributes:
                if attribute in {field.name for field in fields(PokemonParameters)}:
                    self.optimizable_parameters[f"pokemon_{idx}"][attribute] = getattr(self.search_config, attribute)
                else:
                    # Some attributes have different names in the search_config and PokemonParameters, so we need to map them accordingly.
                    if attribute == "items":
                        self.optimizable_parameters[f"pokemon_{idx}"]["item"] = getattr(self.search_config, attribute)
                    elif attribute == "natures":
                        self.optimizable_parameters[f"pokemon_{idx}"]["nature"] = getattr(self.search_config, attribute)
                    elif attribute == "abilities":
                        self.optimizable_parameters[f"pokemon_{idx}"]["ability"] = getattr(self.search_config, attribute)
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
            for pokemon, parameters in self.optimizable_parameters.items() if parameters}

    def set_optimizable_parameters(self, parameters: dict):
        """
        This function sets the parameters that are used to navigate the search space for the team design process.
        :param parameters: A dictionary containing the parameters that can be optimized.
        """
        raise NotImplementedError

    def yield_team(self) -> str:
        """
        This function is what is called before each battle starts to define the team that is used.
        Therefore this is the "generate" or "compile" function.

        In this function, the parameters that the class exposes will be consumed/evaluated to generate and return a team.
        :return: A string representing the generated team.
        """

        raise NotImplementedError
