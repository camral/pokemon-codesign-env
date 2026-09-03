import numpy as np

from pokemon_codesign_env.baseline_adapters.house_of_dextra import PokemonTeamGraphAdapter
from pokemon_codesign_env.baseline_adapters.team_design import (
    TeamDesignVectorAdapter,
    make_all_attribute_search_config,
)
from pokemon_codesign_env.configs import FormatConfig


def make_adapter() -> TeamDesignVectorAdapter:
    config = make_all_attribute_search_config(FormatConfig.from_showdown("gen4customgame"))
    return TeamDesignVectorAdapter(config)


def test_house_graph_has_root_and_six_bidirectionally_connected_slots():
    graph_adapter = PokemonTeamGraphAdapter(make_adapter())
    design = graph_adapter.initial_design(np.random.default_rng(3))

    nodes, edges = graph_adapter.encode(design)

    assert nodes.shape == (7, graph_adapter.feature_dimension)
    assert edges.shape == (2, 12)
    assert set(map(tuple, edges.T)) == {
        edge for slot in range(1, 7) for edge in ((0, slot), (slot, 0))
    }
    assert nodes[0, 0] == 1
    assert np.all(nodes[1:, 1] == 1)


def test_house_mutation_preserves_canonical_constraints():
    vector_adapter: TeamDesignVectorAdapter = make_adapter()
    graph_adapter = PokemonTeamGraphAdapter(vector_adapter)
    rng = np.random.default_rng(9)
    design = graph_adapter.initial_design(rng)

    for _ in range(50):
        design = graph_adapter.mutate(design, rng)
        vector_adapter.designer.set_optimizable_parameter_values(design)


def test_house_serialization_and_schema_are_stable():
    graph_adapter = PokemonTeamGraphAdapter(make_adapter())
    design = graph_adapter.initial_design(np.random.default_rng(4))

    assert graph_adapter.serialize(design) == graph_adapter.serialize(design)
    assert graph_adapter.schema()["nodes"] == 7
    assert graph_adapter.schema()["team_design_schema"]["level_optimized"] is False
