import json

import numpy as np
import pytest
import zarr

from pokemon_codesign_env.transformer_data import (
    PokemonTrajectoryWriter,
    encode_team_modalities,
    held_out_episode_indices,
    native_team_features,
)


def _parameters(species):
    return {
        f"pokemon_{slot}": {
            "species": f"{species}{slot}",
            "moves": [f"move{slot}", "shared"],
            "item": "item",
            "nature": "nature",
            "ability": "ability",
            "happiness": 200,
            "gender": "M",
            "tera_type": None,
            "evs": {stat: slot for stat in ("hp", "atk", "def", "spa", "spd", "spe")},
            "ivs": {stat: 31 for stat in ("hp", "atk", "def", "spa", "spd", "spe")},
        }
        for slot in range(6)
    }


def test_writer_materializes_native_modalities_and_split(tmp_path):
    writer = PokemonTrajectoryWriter(3, 2, 4)
    writer.add_episode(
        np.array([0.1, 0.2, 0.3]),
        _parameters("mon"),
        "random",
        [(np.array([1, 2]), np.array([1, 0, 1, 0]), 2, 1.0)],
        held_out=True,
    )
    path = tmp_path / "data.zarr"
    writer.write(path, schema={"version": 1})
    root = zarr.open_group(str(path), mode="r")

    assert root["pokemon"].shape == (1, 6, 7)
    assert root["move"].shape == (1, 6, 4)
    assert root["ev"].shape == root["iv"].shape == (1, 6, 6)
    assert bool(root["held_out"][0])
    assert root["opponent_task"][0] > 0
    assert "species" in json.loads(root.attrs["vocabularies"])


def test_held_out_split_is_reproducible_and_never_empty_or_complete():
    first = held_out_episode_indices(5, 0.01, 7)
    second = held_out_episode_indices(5, 0.01, 7)
    assert first == second
    assert len(first) == 1
    assert len(held_out_episode_indices(5, 0.99, 7)) == 4

    with pytest.raises(ValueError):
        held_out_episode_indices(1, 0.2, 7)


def test_native_features_round_trip_through_saved_vocabulary(tmp_path):
    parameters = _parameters("native")
    writer = PokemonTrajectoryWriter(3, 2, 4)
    writer.add_episode(np.zeros(3), parameters, "random", [(np.zeros(2), np.ones(4), 0, 0.0)])
    path = tmp_path / "native.zarr"
    writer.write(path, schema={})
    root = zarr.open_group(str(path), mode="r")
    vocabularies = json.loads(root.attrs["vocabularies"])
    encoded = encode_team_modalities(parameters, vocabularies)

    np.testing.assert_array_equal(encoded[0], root["pokemon"][0])
    np.testing.assert_array_equal(encoded[1], root["move"][0])
    features, scales = native_team_features(*encoded)
    assert features.shape == scales.shape == (138,)
