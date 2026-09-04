import json

import numpy as np
import zarr

from pokemon_codesign_env.transformer_data import PokemonTrajectoryWriter


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
