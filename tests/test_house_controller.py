import numpy as np
import torch
from gymnasium import spaces

from pokemon_codesign_env.house_controller import MorphologyConditionedGRUActorCritic


def test_house_controller_masks_actions_and_resets_gru_state():
    observation_space = spaces.Dict(
        {
            "continuous": spaces.Box(-1, 1, shape=(3,), dtype=np.float32),
            "action_mask": spaces.MultiBinary(4),
        }
    )
    policy = MorphologyConditionedGRUActorCritic(
        observation_space, action_count=4, morphology_dimension=5
    )
    observations = {
        "continuous": torch.ones((2, 3)),
        "action_mask": torch.tensor([[1, 0, 1, 0], [0, 1, 0, 0]]),
    }
    morphology = torch.zeros((2, 5))
    hidden = torch.ones((2, 256))

    distribution, values, next_hidden = policy.step(
        observations, morphology, hidden, torch.tensor([True, False])
    )

    assert distribution.logits[0, 1].isneginf()
    assert distribution.logits[1, 0].isneginf()
    assert values.shape == (2,)
    assert next_hidden.shape == (2, 256)
    assert not torch.equal(next_hidden[0], next_hidden[1])


def test_house_controller_uses_required_architecture():
    observation_space = spaces.Dict(
        {
            "binary": spaces.MultiBinary(2),
            "action_mask": spaces.MultiBinary(3),
        }
    )
    policy = MorphologyConditionedGRUActorCritic(observation_space, 3, 7)

    assert policy.gru.hidden_size == 256
    assert [layer.out_features for layer in policy.mlp if isinstance(layer, torch.nn.Linear)] == [
        512,
        256,
        128,
    ]
