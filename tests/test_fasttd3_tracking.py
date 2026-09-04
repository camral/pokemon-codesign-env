import numpy as np

from pokemon_codesign_env.fasttd3_tracking import (
    FastTD3RunTracker,
    executed_action_array,
)


def test_tracker_keeps_independent_parallel_teams_and_selects_winner():
    tracker = FastTD3RunTracker(2)
    infos = {
        "phase": np.asarray(["design", "design"], dtype=object),
        "_phase": np.asarray([True, True]),
        "invalid_design": np.asarray([False, False]),
        "_invalid_design": np.asarray([True, True]),
        "selected_design": np.asarray([[0.1, 0.2], [0.8, 0.9]]),
        "_selected_design": np.asarray([True, True]),
        "executed_actions": np.asarray([[0.5, 0.0], [-0.5, 0.0]]),
    }
    tracker.update(infos, [0, 0], [False, False])
    tracker.update([{"phase": "battle"}, {"phase": "battle"}], [1, 3], [True, True])

    assert tracker.completed_returns == [1.0, 3.0]
    np.testing.assert_array_equal(tracker.best_parameters, [0.8, 0.9])
    assert tracker.environment_transitions == 4
    np.testing.assert_array_equal(executed_action_array(infos, 2), [[0.5, 0], [-0.5, 0]])


def test_tracker_counts_invalid_designs():
    tracker = FastTD3RunTracker(1)
    tracker.update(
        [
            {
                "phase": "design",
                "invalid_design": True,
                "executed_actions": np.zeros(2),
            }
        ],
        [-100],
        [True],
    )

    assert tracker.invalid_candidate_rate == 1.0


def test_executed_actions_are_read_from_same_step_final_info():
    infos = {
        "final_info": np.asarray([{"executed_actions": np.asarray([0.0, 0.7])}], dtype=object),
        "_final_info": np.asarray([True]),
    }

    np.testing.assert_allclose(executed_action_array(infos, 1), [[0.0, 0.7]])
