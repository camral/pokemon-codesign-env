from pokemon_codesign_env.showdown import _poke_env_can_represent


def test_constructor_compatibility_catches_malformed_species():
    assert _poke_env_can_represent("pikachu", 6)
    assert not _poke_env_can_represent("vivillonfancy", 6)
    assert not _poke_env_can_represent("missingno", 6)
    assert not _poke_env_can_represent("basculegion", 8)
    assert _poke_env_can_represent("vivillonfancy", 9)
