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

    def yield_team(self) -> str:
        """
        This function is what is called before each battle starts to define the team that is used.
        Therefore this is the "generate" or "compile" function.

        In this function, the parameters that the class exposes will be consumed/evaluated to generate and return a team.
        :return: A string representing the generated team.
        """

        raise NotImplementedError
