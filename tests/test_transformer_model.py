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
