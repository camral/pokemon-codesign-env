import numpy as np
from poke_env.teambuilder import Teambuilder

import pokemon_codesign_env.house_problem as house_problem


def _observations(problem, count, masks):
    values = {
        key: np.zeros((count, *problem.encoder.observation_space[key].shape), dtype=np.float32)
        for key in problem.observation_keys
    }
    values["action_mask"] = np.asarray(masks, dtype=bool)
    return values


def test_house_problem_exposes_complete_team_graph_at_fixed_level():
    problem = house_problem.make_problem({"environment": "random"})
    rng = np.random.default_rng(7)
    design = problem.sample_design(problem.family_name, rng)

    nodes, edges = problem.encode(design)
    morphology = problem.morphology([design])
    problem.vector_adapter.designer.set_optimizable_parameter_values(design)
    packed = problem.vector_adapter.designer.yield_team()

    assert nodes.shape == (7, problem.graph_adapter.feature_dimension)
    assert edges.shape == (2, 12)
    assert morphology.shape == (1, nodes.size)
    assert problem.vector_adapter.schema()["level_optimized"] is False
    assert problem.vector_adapter.schema()["level_mean"] == 50
    assert problem.vector_adapter.schema()["level_std"] == 0
    assert {member.level for member in Teambuilder.parse_packed_team(packed)} == {50}


def test_house_controller_batch_executes_highest_scoring_legal_action(monkeypatch):
    problem = house_problem.make_problem({"environment": "random"})
    designs = [
        problem.sample_design(problem.family_name, np.random.default_rng(seed)) for seed in (1, 2)
    ]
    masks = np.zeros((2, problem.action_count), dtype=bool)
    masks[0, (1, 4)] = True
    masks[1, (2, 3)] = True

    class FakeVecEnv:
        def __init__(self, factories, **kwargs):
            self.factories = factories
            self.kwargs = kwargs
            self.actions = None

        def seed(self, seed):
            self.seed_value = seed

        def reset(self):
            return _observations(problem, 2, masks)

        def step(self, actions):
            self.actions = actions
            return (
                _observations(problem, 2, masks),
                np.asarray([1.0, -1.0]),
                np.asarray([True, False]),
                ({"actor": 0}, {"actor": 1}),
            )

        def close(self):
            pass

    monkeypatch.setattr(house_problem, "ShardedVecEnv", FakeVecEnv)
    batch = house_problem.PokemonControllerBatch(problem, designs, training=False, seed=11)
    observation = batch.reset()
    scores = np.zeros((2, problem.action_count), dtype=np.float32)
    scores[0, 4] = 0.5
    scores[0, 3] = 100
    scores[1, 2] = 0.1
    scores[1, 3] = 0.2

    next_observation, rewards, dones, info = batch.step(scores)

    assert observation.shape == next_observation.shape == (2, problem.observation_dimension)
    assert batch.env.actions.tolist() == [4, 3]
    assert rewards.tolist() == [1.0, -1.0]
    assert dones.tolist() == [True, False]
    assert info["selected_battle_action"].tolist() == [4, 3]
