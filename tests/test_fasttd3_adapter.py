import numpy as np
import pytest

from pokemon_codesign_env.baseline_adapters.fasttd3 import (
    project_actions,
    project_tensor_actions,
    replay_actions,
)


def test_project_actions_keeps_only_the_phase_appropriate_block():
    actions = np.arange(15, dtype=np.float32).reshape(3, 5)

    projected = project_actions(actions, np.array([True, False, True]), 2)

    np.testing.assert_array_equal(projected[0], [0, 1, 0, 0, 0])
    np.testing.assert_array_equal(projected[1], [0, 0, 7, 8, 9])
    np.testing.assert_array_equal(projected[2], [10, 11, 0, 0, 0])
    np.testing.assert_array_equal(actions, np.arange(15, dtype=np.float32).reshape(3, 5))


def test_project_actions_supports_one_environment():
    np.testing.assert_array_equal(
        project_actions(np.ones(5), design_phase=False, embodiment_dimension=2),
        [0, 0, 1, 1, 1],
    )


def test_tensor_projection_preserves_gradients_only_through_active_blocks():
    torch = pytest.importorskip("torch")
    actions = torch.ones((2, 5), requires_grad=True)

    project_tensor_actions(actions, [True, False], 2).sum().backward()

    np.testing.assert_array_equal(actions.grad.numpy(), [[1, 1, 0, 0, 0], [0, 0, 1, 1, 1]])


def test_replay_actions_uses_executed_actions_from_infos():
    proposed = np.ones((2, 4), dtype=np.float32)
    expected = np.array([[1, 1, 0, 0], [0, 0, 1, 1]], dtype=np.float32)

    np.testing.assert_array_equal(
        replay_actions(proposed, [{"executed_actions": row} for row in expected]), expected
    )
    np.testing.assert_array_equal(
        replay_actions(proposed, {"executed_actions": expected}), expected
    )


def test_replay_actions_rejects_missing_or_wrong_shape():
    with pytest.raises(KeyError):
        replay_actions(np.ones((1, 4)), [{}])
    with pytest.raises(ValueError):
        replay_actions(np.ones((1, 4)), {"executed_actions": np.ones((1, 3))})
