from pathlib import Path

import pytest

from pokemon_codesign_env.scaling import _require_built_showdown


def test_showdown_cluster_requires_installer_build(tmp_path: Path):
    with pytest.raises(RuntimeError, match="codesign-gym-install-showdown"):
        _require_built_showdown(tmp_path)

    for name in ("dist/sim/dex.js", "dist/data/learnsets.js", "dist/config/custom-formats.js"):
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
    _require_built_showdown(tmp_path)
