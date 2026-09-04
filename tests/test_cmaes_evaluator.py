import pytest

from pokemon_codesign_env.cmaes_evaluator import PokemonCMAEvaluator


def test_candidate_repeat_seed_panel_is_stable():
    evaluator = PokemonCMAEvaluator("random", seed=20, evaluation_repeats=3)

    seeds = []
    for index in range(6):
        evaluator.evaluation_index = index
        seeds.append(evaluator.evaluation_seed)

    assert seeds == [20, 21, 22, 20, 21, 22]


def test_repeat_count_must_be_positive():
    with pytest.raises(ValueError, match="evaluation_repeats"):
        PokemonCMAEvaluator("random", seed=0, evaluation_repeats=0)


def test_candidate_ids_are_stably_distributed_over_showdown_shards():
    evaluator = PokemonCMAEvaluator("random", seed=0, showdown_port=8000, showdown_shards=4)

    assert evaluator.port_for("0000000000000000") == 8000
    assert evaluator.port_for("0000000000000003") == 8003
    assert evaluator.port_for("0000000000000004") == 8000
