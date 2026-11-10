import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    "course_presentation", Path(__file__).parents[1] / "scripts/build_presentation.py"
)
presentation = importlib.util.module_from_spec(spec)
spec.loader.exec_module(presentation)


@pytest.mark.parametrize(
    ("lo", "hi", "interpretation"),
    [
        (0.1, 0.3, "favors the cue-directed method"),
        (-0.3, -0.1, "favors unbiased IPPO"),
        (-0.1, 0.1, "does not establish an advantage"),
    ],
)
def test_original_method_slide_interprets_the_interval_without_hiding_null_results(lo, hi, interpretation):
    summary = {
        "n_seeds": 20,
        "exploratory_post_hoc": True,
        "unbiased": {"success": {"mean": 0.6}},
        "biased": {"success": {"mean": 0.6 + (lo + hi) / 2}},
        "biased_minus_unbiased": {"mean": (lo + hi) / 2, "lo": lo, "hi": hi},
    }
    slide = presentation.ablation_slide(summary)
    assert "60.0%" in slide["notes"]
    assert presentation.mean(summary["biased"]["success"]) in slide["notes"]
    assert interpretation in slide["notes"]
    assert "exploratory" in slide["notes"]
    assert "95%" in slide["bullets"][2]


def test_incomplete_ablation_cannot_enter_the_presentation():
    with pytest.raises(ValueError, match="complete"):
        presentation.ablation_slide({"n_seeds": 6, "exploratory_post_hoc": True})
