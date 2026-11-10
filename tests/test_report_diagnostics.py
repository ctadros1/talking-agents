import importlib.util
import json
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location(
    "report_diagnostics", Path(__file__).parents[1] / "scripts/report_diagnostics.py"
)
module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(module)


@pytest.mark.parametrize(
    "success,pick,target,expected",
    [
        (True, 2, 2, "success"),
        (False, -1, 2, "no_pick"),
        (False, 1, 2, "wrong_pick"),
        (False, 2, 2, "correct_pick_no_delivery"),
    ],
)
def test_mutually_exclusive_outcomes(success, pick, target, expected):
    assert module.outcome({"success": success, "runner_pick": pick, "target": target}) == expected


def test_success_with_wrong_pick_is_rejected():
    with pytest.raises(ValueError):
        module.outcome({"success": True, "runner_pick": 1, "target": 2})


def test_counts_must_cover_exactly_fixed_episodes():
    with pytest.raises(ValueError):
        module.summarize_counts(
            [
                {
                    "condition": "comm",
                    "split": "held_out",
                    "seed": 0,
                    "n": 200,
                    "success": 1,
                    "no_pick": 1,
                    "wrong_pick": 1,
                    "correct_pick_no_delivery": 1,
                }
            ]
        )


def test_negative_counts_are_rejected_even_if_total_matches():
    rows = [
        {
            "condition": condition,
            "split": split,
            "seed": seed,
            "n": 200,
            "success": 200,
            "no_pick": -1,
            "wrong_pick": 1,
            "correct_pick_no_delivery": 0,
        }
        for condition in ("comm", "oracle")
        for split in ("held_out", "unseen")
        for seed in range(20)
    ]
    with pytest.raises(ValueError):
        module.summarize_counts(rows)


def test_saved_partition_preserves_canonical_success_and_reproduces_summary():
    root = Path(__file__).parents[1]
    counts = json.loads((root / "results/report_diagnostics/outcome_counts.json").read_text())
    saved = json.loads((root / "results/report_diagnostics/summary.json").read_text())
    canonical = json.loads((root / "results/v2/summary.json").read_text())
    rebuilt = module.summarize_counts(counts["rows"])
    assert rebuilt["conditions"] == saved["conditions"]
    assert len(counts["source_sha256"]) == 40
    for condition in ("comm", "oracle"):
        for split in ("held_out", "unseen"):
            assert saved["conditions"][condition][split]["success"]["mean"] == pytest.approx(
                canonical["conditions"][condition][split]["success"]["mean"], abs=1e-12
            )
