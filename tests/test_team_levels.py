import numpy as np
from poke_env.teambuilder import Teambuilder, TeambuilderPokemon

from pokemon_codesign_env.team_levels import resample_packed_team_levels
from pokemon_codesign_env.team_pool import RandomTeamPool


def packed_team(level: int = 100) -> str:
    return Teambuilder.join_team(
        [
            TeambuilderPokemon(species="Pikachu", moves=["Thunderbolt"], level=level),
            TeambuilderPokemon(species="Raichu", moves=["Thunderbolt"], level=level),
        ]
    )


def levels(team: str) -> list[int | None]:
    return [pokemon.level for pokemon in Teambuilder.parse_packed_team(team)]


def test_resample_packed_team_levels_sets_level_fifty() -> None:
    actual = resample_packed_team_levels(
        packed_team(), np.random.default_rng(3), minimum=1, maximum=100
    )

    assert levels(actual) == [50, 50]


def test_random_team_pool_sets_level_fifty_on_each_yield() -> None:
    builder = RandomTeamPool((packed_team(),)).builder(seed=9)

    first = levels(builder.yield_team())
    second = levels(builder.yield_team())
    assert first + second == [50, 50, 50, 50]
