import importlib.util
import json
from pathlib import Path


def _runner_module():
    path = Path(__file__).resolve().parents[1] / "examples" / "run_pokemon_codesign_benchmarks.py"
    spec = importlib.util.spec_from_file_location("benchmark_runner", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_collect_results_requires_and_combines_common_artifacts(tmp_path):
    module = _runner_module()
    run = tmp_path / "method"
    run.mkdir()
    for name in module.REQUIRED_ARTIFACTS:
        payload = {"mean_return": 3.0} if name == "metrics.json" else {}
        (run / name).write_text(json.dumps(payload), encoding="utf-8")

    results = module.collect_results([{"name": "method", "output_dir": str(run)}])

    assert results[0]["mean_return"] == 3.0
