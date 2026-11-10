"""Recompute all saved statistical summaries and check frozen scientific bytes."""

from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
from pathlib import Path

from emcomm import analysis, analysis_v2

ROOT = Path(__file__).resolve().parents[1]


def verify():
    provenance = json.loads((ROOT / "research-provenance.json").read_text())
    for name, expected in provenance["frozen_files"].items():
        actual = hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError(f"Frozen scientific file changed: {name}")
    print(f"PASS: {len(provenance['frozen_files'])} frozen scientific files match the original source.")
    for label, folder, module in (
        ("v1", "results", analysis),
        ("v2", "results/v2", analysis_v2),
        ("vocal", "results/vocal", analysis),
    ):
        path = ROOT / folder
        actual = module.summarize(module.load_rows(path / "per_run.csv"))
        expected = json.loads((path / "summary.json").read_text())
        if actual != expected:
            raise ValueError(f"{label} summary differs from saved table")
        print(f"PASS: {label} statistical summary reproduced exactly.")
    spec = importlib.util.spec_from_file_location("ablation", ROOT / "scripts/analyze_ablation.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    with (ROOT / "results/per_run.csv").open() as file:
        original = list(csv.DictReader(file))
    with (ROOT / "results/ablation_unbiased/per_run.csv").open() as file:
        control = list(csv.DictReader(file))
    actual = module.summarize_comparison(original, control)
    saved = json.loads((ROOT / "results/ablation_unbiased/summary.json").read_text())
    if actual != {key: saved[key] for key in actual}:
        raise ValueError("Supplementary objective ablation differs from saved table")
    print("PASS: exploratory objective ablation reproduced exactly.")
    print("All checks passed. No stored result was changed.")


if __name__ == "__main__":
    verify()
