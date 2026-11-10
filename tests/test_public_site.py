"""Public adapter and distribution boundaries; no scientific results are changed."""

import hashlib
import importlib.util
import json
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_browser_weights_are_bound_to_the_included_final_checkpoints():
    for condition in ("comm", "no_comm"):
        manifest = json.loads((ROOT / f"web/models/{condition}.json").read_text())
        checkpoint = ROOT / f"demo_checkpoints/v1/{condition}/seed_0019/final.pt"
        assert hashlib.sha256(checkpoint.read_bytes()).hexdigest() == manifest["checkpoint_sha256"]
        binary = (ROOT / f"web/models/{condition}.bin").read_bytes()
        assert hashlib.sha256(binary).hexdigest() == manifest["binary_sha256"]
        assert manifest["seed"] == 19
        assert len(manifest["agents"]) == 2


def test_site_build_publishes_only_public_assets_and_exact_saved_results():
    spec = importlib.util.spec_from_file_location("public_build", ROOT / "scripts/build_site.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    out = module.build()
    files = [p.relative_to(out).as_posix() for p in out.rglob("*") if p.is_file()]
    assert "index.html" in files
    assert "replays/vendor/three.min.js" in files
    assert not any(".env" in p or "checkpoints" in p or "fixtures" in p or "node_modules" in p for p in files)
    assert all(p.startswith(("models/", "data/", "replays/", "downloads/")) or "/" not in p for p in files)
    data = json.loads((out / "data/results.json").read_text())
    for key, file in (
        ("v1", "summary.json"),
        ("v2", "v2/summary.json"),
        ("vocal", "vocal/summary.json"),
        ("ablation", "ablation_unbiased/summary.json"),
    ):
        assert data[key] == json.loads((ROOT / "results" / file).read_text())
    with zipfile.ZipFile(out / "downloads/results-and-protocol.zip") as archive:
        assert "results/per_run.csv" in archive.namelist()
        assert "docs/PREREGISTRATION_V2.md" in archive.namelist()
        assert not any(n.startswith(("runs/", ".git/", "node_modules/")) for n in archive.namelist())


def test_public_snapshot_preserves_all_frozen_scientific_bytes():
    provenance = json.loads((ROOT / "research-provenance.json").read_text())
    for name, expected in provenance["frozen_files"].items():
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == expected, name
