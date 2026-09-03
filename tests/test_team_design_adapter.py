from functools import lru_cache

import numpy as np

from pokemon_codesign_env.baseline_adapters.team_design import (
    DEFAULT_LEVEL_MEAN,
    DEFAULT_LEVEL_STD,
    TeamDesignVectorAdapter,
    make_all_attribute_search_config,
)
from pokemon_codesign_env.configs import AttributeMode
from pokemon_codesign_env.team_designer import TeamDesigner


@lru_cache(maxsize=1)
def make_adapter() -> TeamDesignVectorAdapter:
    from pokemon_codesign_env.configs import FormatConfig

    config = make_all_attribute_search_config(FormatConfig.from_showdown("gen4customgame"))
    return TeamDesignVectorAdapter(config)


def test_all_supported_attributes_are_optimized_but_level_is_not() -> None:
    adapter = make_adapter()
    config = adapter.search_config

    assert config.level_mean == DEFAULT_LEVEL_MEAN == 70
    assert config.level_std == DEFAULT_LEVEL_STD == 10.0
    assert set(config.get_optimizable_attributes()) == {
        "species",
        "evs",
        "ivs",
        "moves",
        "items",
        "natures",
        "abilities",
        "happiness",
        "gender",
    }
    assert config.tera_types is AttributeMode.DEFAULT
    assert adapter.schema()["level_optimized"] is False
    assert adapter.schema()["version"] == 1
    assert adapter.schema()["move_slots"] == 4
    assert all(field.parameter != "level" for field in adapter.fields)
    assert next(field for field in adapter.fields if field.parameter == "moves").width == 4


def test_candidate_identity_is_stable() -> None:
    adapter = make_adapter()
    parameters = adapter.decode(np.full(adapter.dimension, 0.5))

    assert adapter.candidate_id(parameters) == adapter.candidate_id(parameters)
    assert len(adapter.candidate_id(parameters)) == 64


def test_numeric_vector_round_trip_preserves_decoded_parameters() -> None:
    adapter = make_adapter()
    vector = np.random.default_rng(7).uniform(size=adapter.dimension)

    decoded = adapter.decode(vector)
    encoded = adapter.encode(decoded)

    assert encoded.shape == (adapter.dimension,)
    assert np.all((encoded >= 0.0) & (encoded <= 1.0))
    assert adapter.decode(encoded) == decoded


def test_decoded_team_respects_dependent_and_stat_spaces() -> None:
    adapter = make_adapter()
    parameters = adapter.decode(np.random.default_rng(11).uniform(size=adapter.dimension))
    designer = TeamDesigner(adapter.search_config, validate_teams=True)
    designer.set_optimizable_parameter_values(parameters)

    packed_team = designer.yield_team()

    assert packed_team
    assert len(designer.team) == adapter.search_config.team_size
    for pokemon in designer.team:
        assert len(pokemon.moves) == len(set(pokemon.moves))
        assert len(pokemon.moves) <= adapter.search_config.format.max_move_count
        assert sum(pokemon.evs.values()) <= adapter.search_config.ev_budget
        assert sum(pokemon.ivs.values()) <= adapter.search_config.iv_budget
        assert (
            adapter.search_config.format.min_level
            <= pokemon.level
            <= adapter.search_config.format.max_level
        )
