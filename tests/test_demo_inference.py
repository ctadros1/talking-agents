import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location("demo_inference", Path(__file__).parents[1] / "scripts/demo_inference.py")
demo = importlib.util.module_from_spec(spec)
spec.loader.exec_module(demo)


def test_demo_rejects_output_that_would_overwrite_scientific_artifacts(tmp_path):
    (tmp_path / "results").mkdir()
    with pytest.raises(ValueError, match="output"):
        demo.new_output(tmp_path, tmp_path / "results")
    with pytest.raises(ValueError, match="output"):
        demo.new_output(tmp_path, tmp_path / "runs")
    with pytest.raises(ValueError, match="output"):
        demo.new_output(tmp_path, tmp_path / "viewer")
    safe = demo.new_output(tmp_path, tmp_path / "demo_output" / "fresh")
    assert safe.is_dir()
    with pytest.raises(ValueError, match="exist"):
        demo.new_output(tmp_path, safe)
