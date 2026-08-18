import numpy as np
from poke_env.teambuilder import Teambuilder
from pokemon_codesign_env.configs import TeamSearchConfig

class TeamDesigner(Teambuilder):
    """
    This class expresses the team design process as a search problem.
    """
    def __init__(self, search_config: TeamSearchConfig):
        super().__init__()
        self.search_config = search_config
        self.optimizable_parameters = {}
        
        self.define_optimizable_parameters()

    def define_optimizable_parameters(self):
        """
        This function uses the search_config to define the parameters that are used to navigate the search space for the team design process.
        :return:
        """
        raise NotImplementedError

    def get_optimizable_parameters(self) -> dict:
        """
        This function returns the parameters that are used to navigate the search space for the team design process.
        :return: A dictionary containing the parameters that can be optimized.
        """
        raise NotImplementedError

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
