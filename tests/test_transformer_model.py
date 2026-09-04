import pytest
import torch

from pokemon_codesign_env.transformer_model import PokemonTransformerTransformer


def test_transformer_outputs_designs_and_masks_control_actions():
    model = PokemonTransformerTransformer(12, 8, 4, 2, width=16, layers=1, heads=2)
    logits, design = model(
        torch.rand(3, 12),
        torch.rand(3, 8),
        torch.tensor([0, 1, 0]),
        torch.tensor([[1, 0, 0, 0], [0, 1, 1, 0], [0, 0, 0, 1]]),
    )

    assert logits.shape == (3, 4)
    assert design.shape == (3, 12)
    assert torch.all((design >= 0) & (design <= 1))
    assert logits[0, 1].isneginf()


def test_generated_team_depends_on_task_not_supplied_design_or_battle_state():
    model = PokemonTransformerTransformer(12, 8, 4, 2, width=16, layers=1, heads=2)
    model.eval()
    task = torch.tensor([1, 1])

    _, generated = model(
        torch.stack((torch.zeros(12), torch.ones(12))),
        torch.stack((torch.zeros(8), torch.ones(8))),
        task,
    )

    torch.testing.assert_close(generated[0], generated[1])


def test_native_team_features_are_required_when_configured():
    model = PokemonTransformerTransformer(
        4, 3, 2, 1, width=8, layers=1, heads=2, native_dimension=5
    )
    design = torch.zeros(1, 4)
    state = torch.zeros(1, 3)
    tasks = torch.zeros(1, dtype=torch.long)

    with pytest.raises(ValueError, match="native_team"):
        model(design, state, tasks)
    logits, generated = model(design, state, tasks, native_team=torch.zeros(1, 5))
    assert logits.shape == (1, 2)
    assert generated.shape == (1, 4)


def test_native_checkpoint_round_trip(tmp_path):
    model = PokemonTransformerTransformer(
        4, 3, 2, 1, width=8, layers=1, heads=2, native_dimension=5
    )
    path = tmp_path / "model.pt"
    torch.save({"configuration": model.configuration, "state_dict": model.state_dict()}, path)

    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    restored = PokemonTransformerTransformer(**checkpoint["configuration"])
    restored.load_state_dict(checkpoint["state_dict"])

    for expected, actual in zip(model.parameters(), restored.parameters(), strict=True):
        torch.testing.assert_close(expected, actual)
