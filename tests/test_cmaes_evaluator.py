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
