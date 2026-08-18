from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class TeamSearchConfig:
    """Parameters defining the team design search space."""

    # Numerical parameters
    team_size: int = 6
    EV_budget: int = 510
    IV_budget: int = 31
    level_mean: int = 50
    level_std: int = 0

    # What is being optimized? The options are "default", "random", and "optimized"
    EVs: str = "default" # Options: "default", "random", "optimized"
    IVs: str = "default" # Options: "default", "random", "optimized"
    moves: str = "default" # Options: "default", "random", "optimized"
    items: str = "default" # Options: "default", "random", "optimized", "none"
    natures: str = "default" # Options: "default", "random", "optimized"
    abilities: str = "default" # Options: "default", "random", "optimized"
    happiness: str = "default" # Options: "default", "random", "optimized"
    sex: str = "random" # Options: "random", "optimized"

    # Restrictions on Pokémon selection
    generations_allowed: list[int] = None # Options: None (all generations allowed), or a list of generations to allow
    types_allowed: list[str] = None # Options: None (all types allowed), or a list of types to allow

