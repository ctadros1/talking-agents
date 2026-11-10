import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    "submission_ablation", Path(__file__).parents[1] / "scripts/analyze_ablation.py"
)
ablation = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ablation)


def rows(success, modes=("intact", "permuted")):
    return [
        {
            "condition": "comm",
            "seed": str(seed),
            "split": "held_out",
            "intervention": mode,
            "success": str(success if mode == "intact" else 0.2),
        }
        for seed in range(20)
        for mode in modes
    ]


def test_complete_ablation_reports_original_method_difference_and_causal_drop():
    summary = ablation.summarize_comparison(rows(0.8), rows(0.25))
    assert summary["n_seeds"] == 20
    assert summary["exploratory_post_hoc"] is True
    assert summary["biased_minus_unbiased"]["mean"] == pytest.approx(0.55)
    assert summary["unbiased"]["intact_minus_permuted"]["mean"] == pytest.approx(0.05)


def test_missing_seed_cannot_be_presented_as_a_complete_experiment():
    with pytest.raises(ValueError, match="seed"):
        ablation.summarize_comparison(rows(0.8), rows(0.25)[:-2])


def test_duplicate_and_invalid_results_are_rejected():
    with pytest.raises(ValueError, match="duplicate"):
        ablation.summarize_comparison(rows(0.8), rows(0.25) + rows(0.25)[:1])
    bad = rows(0.25)
    bad[0]["success"] = "nan"
    with pytest.raises(ValueError, match="success"):
        ablation.summarize_comparison(rows(0.8), bad)
