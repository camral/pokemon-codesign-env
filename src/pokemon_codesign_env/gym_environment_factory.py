import gymnasium
from gymnasium import spaces
import numpy as np

from pathlib import Path

from pokemon_codesign_env.configs import TeamSearchConfig, FormatConfig

class PokEnv(gymnasium.Env):
    def __init__(self, opponent: str = "random", format_config: FormatConfig = FormatConfig(), embodiment_config: TeamSearchConfig = TeamSearchConfig()):
        super().__init__()