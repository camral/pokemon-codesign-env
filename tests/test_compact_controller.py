import numpy as np
from gymnasium import spaces

from pokemon_codesign_env.compact_controller import CompactBattleController


def test_compact_controller_never_selects_an_illegal_action():
    observation_space = spaces.Dict(
        {"state": spaces.Box(-1, 1, (2,)), "action_mask": spaces.MultiBinary(3)}
    )
    controller = CompactBattleController(observation_space, 3, hidden_size=2)
    controller.set_parameters(np.ones(controller.parameter_count))

    action = controller.predict({"state": [1, 1], "action_mask": [0, 1, 0]})

    assert action == 1
