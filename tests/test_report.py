import importlib.util
import json
import shutil
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location("build_report", Path(__file__).parents[1] / "report" / "build_report.py")
build_report = importlib.util.module_from_spec(spec)
spec.loader.exec_module(build_report)

DATA = {
    "conditions": {"comm": {"success": {"mean": 0.812, "lo": 0.75, "hi": 0.86}, "mi_first_msg": {"mean": 1.73}}},
    "comparisons": {"comm_minus_no_comm": {"mean": 0.55, "lo": 0.49, "hi": 0.61}},
    "criterion": {"met": True},
    "calibration": {"scripted": {"scripted_comm": 1.0}},
}


def _ci(mean=0.7, lo=0.6, hi=0.8):
    return {"mean": mean, "lo": lo, "hi": hi}


def _claims(a=True, b=True, c=True):
    return {
        "h1_gain_at_least_15pp": a,
        "h1_ci_excludes_zero": a,
        "h2_drop_at_least_10pp": a,
        "h2_ci_excludes_zero": a,
        "h3_gain_at_least_15pp": b,
        "h3_ci_excludes_zero": b,
        "h4_mean_above_reference": c,
        "h4_ci_above_reference": c,
        "claim_a": a,
        "claim_b": a and b,
        "claim_c": a and b and c,
    }


def _synthetic_v2_data(claims=None):
    split = {
        "success": _ci(),
        "color_acc": _ci(0.8, 0.7, 0.9),
        "shape_acc": _ci(0.75, 0.65, 0.85),
        "twin_color": _ci(0.4, 0.3, 0.5),
        "twin_shape": _ci(0.35, 0.25, 0.45),
        "twin_opposite": _ci(0.25, 0.15, 0.35),
        "mi_w1_color": _ci(1.1, 1.0, 1.2),
        "mi_w1_shape": _ci(0.2, 0.1, 0.3),
        "mi_w2_color": _ci(0.3, 0.2, 0.4),
        "mi_w2_shape": _ci(1.0, 0.9, 1.1),
        "topsim": _ci(0.6, 0.5, 0.7),
        "posdis": _ci(0.7, 0.6, 0.8),
    }
    conditions = {
        condition: {"held_out": dict(split), "unseen": dict(split)}
        for condition in ("comm", "no_comm", "random_msg", "oracle")
    }
    comparisons = {
        split_name: {
            name: _ci(0.2, 0.1, 0.3)
            for name in (
                "comm_minus_no_comm",
                "intact_minus_permuted",
                "intact_minus_permuted_w1",
                "intact_minus_permuted_w2",
                "intact_minus_silenced",
            )
        }
        for split_name in ("held_out", "unseen")
    }
    summary = {"conditions": conditions, "comparisons": comparisons, "claims": _claims() if claims is None else claims}
    summary.update(
        calibration={
            "scripted": {
                "held_out": {
                    "scripted_color_only": 0.5,
                    "scripted_shape_only": 0.5,
                    "scripted_no_comm": 0.25,
                    "scripted_comm": 1.0,
                }
            }
        },
        pilot={
            "rung1": {"means": {"oracle": 0.9, "no_comm": 0.25, "comm": 0.7}},
            "rung2": {"means": {"oracle": 0.9, "no_comm": 0.25, "comm": 0.7}},
        },
        meta={
            "env_steps_millions": "20.0",
            "sig_coef": "0.2",
            "held_out_combos": "red sphere, yellow ring, green cylinder, blue cube, purple cone",
        },
        support_reference={
            "fixed_h4_reference": 0.5,
            "support_aware_no_message_identification_rate": 0.875,
            "one_property_plus_matching_identification_rate": 1.0,
            "unseen_prior": {
                "equiprobable_target_square_states": 80,
                "held_out_candidate_count_distribution": {"1": 60, "2": 20},
            },
        },
    )
    summary["report"] = build_report._v2_report_fields(summary)
    return {"v2": summary}


def _between(path, start, end):
    text = path.read_text()
    return text[text.index(start) : text.index(end)]


def test_render_formats():
    text = build_report.render(
        "{{conditions.comm.success|ci}}; {{comparisons.comm_minus_no_comm|pp}}; "
        "{{conditions.comm.mi_first_msg|bits}}; {{criterion.met}}; {{calibration.scripted.scripted_comm|pct}}",
        DATA,
    )
    assert text == "81.2% (95% CI 75.0–86.0); +55.0 pp (95% CI +49.0 to +61.0); 1.73 bits; True; 100.0%"


def test_unknown_placeholder_fails_loudly():
    with pytest.raises(KeyError, match="conditions.oracle"):
        build_report.render("{{conditions.oracle.success|ci}}", DATA)


def test_mean_percentage_format():
    assert build_report.render("{{conditions.comm.success|mpct}}", DATA) == "81.2%"


def test_num2_and_yesno_formats():
    data = {"x": {"mean": 0.873, "lo": 0.8, "hi": 0.9}, "ok": True, "bad": False}
    assert build_report.render("{{x|num2}} {{ok|yesno}} {{bad|yesno}}", data) == "0.87 yes no"


def test_null_and_nonfinite_numbers_are_not_estimable():
    data = {"null_ci": _ci(None, None, None), "nan_ci": _ci(0.7, float("nan"), 0.8), "null": None}
    rendered = build_report.render("{{null_ci|ci}}; {{nan_ci|ci}}; {{null|pct}}", data)
    assert rendered == "not estimable; not estimable; not estimable"
    assert "None" not in rendered and "nan" not in rendered.lower()


def test_yesno_rejects_non_boolean_values():
    with pytest.raises(ValueError, match="requires a boolean"):
        build_report.render("{{value|yesno}}", {"value": 1})


def test_load_data_includes_v2(tmp_path):
    root = Path(__file__).parents[1]
    (tmp_path / "results" / "v2").mkdir(parents=True)
    (tmp_path / "configs").mkdir()
    for name in ("base.toml", "v2.toml"):
        shutil.copy(root / "configs" / name, tmp_path / "configs" / name)
    for rel in ("results/summary.json", "results/calibration.json", "results/v2/calibration.json"):
        (tmp_path / rel).write_text("{}")
    (tmp_path / "results" / "v2" / "summary.json").write_text(json.dumps({"claims": {"claim_a": True}}))
    (tmp_path / "results" / "v2" / "support_reference.json").write_text("{}")
    (tmp_path / "results" / "v2" / "pilot_rung1.json").write_text(json.dumps({"means": {"comm": 0.8}}))
    data = build_report.load_data(tmp_path)
    text = build_report.render(
        "{{v2.claims.claim_a|yesno}} | {{v2.meta.held_out_combos}} | {{v2.pilot.rung1.means.comm|pct}}", data
    )
    assert text == "yes | red sphere, yellow ring, green cylinder, blue cube, purple cone | 80.0%"


@pytest.mark.parametrize(
    ("claims", "statuses"),
    [
        (_claims(), ("supported", "supported", "supported")),
        (_claims(c=False), ("supported", "supported", "not supported")),
        (_claims(b=False, c=False), ("supported", "not supported", "not reached")),
        (_claims(a=False, b=True, c=True), ("not supported", "not reached", "not reached")),
        ({}, ("not estimable", "not estimable", "not estimable")),
    ],
)
def test_fixed_sequence_statuses_and_interpretation_are_safe(claims, statuses):
    fields = build_report._v2_report_fields({"claims": claims})
    assert (fields["claim_a_status"], fields["claim_b_status"], fields["claim_c_status"]) == statuses
    if statuses != ("supported", "supported", "supported"):
        assert "Claims A, B, and C were supported" not in fields["interpretation"]
    assert "proves compositionality" not in fields["interpretation"]
    assert "both properties transfer" not in fields["interpretation"]


def test_v2_report_and_guide_sections_resolve_from_a_synthetic_full_summary():
    root = Path(__file__).parents[1]
    templates = (
        _between(root / "report/report.md", "## 9. v2: words that combine", "## 10. Limitations"),
        _between(root / "docs/guide/how-it-works.md", "## 7. v2: words that combine", "## 8. Try it yourself"),
    )
    for template in templates:
        rendered = build_report.render(template, _synthetic_v2_data())
        assert "{{" not in rendered
        assert "Claims A, B, and C were supported" in rendered
        assert "does not" in rendered and "compositionality" in rendered


def test_v2_templates_render_null_results_without_overclaiming():
    data = _synthetic_v2_data(claims={})
    for condition in data["v2"]["conditions"].values():
        for split in condition.values():
            for value in split.values():
                value.update(mean=None, lo=None, hi=None)
    for split in data["v2"]["comparisons"].values():
        for value in split.values():
            value.update(mean=None, lo=None, hi=None)
    root = Path(__file__).parents[1]
    templates = (
        _between(root / "report/report.md", "## 9. v2: words that combine", "## 10. Limitations"),
        _between(root / "docs/guide/how-it-works.md", "## 7. v2: words that combine", "## 8. Try it yourself"),
    )
    for template in templates:
        rendered = build_report.render(template, data)
        assert "not estimable" in rendered
        assert "None%" not in rendered and "nan%" not in rendered.lower()
        assert "None (" not in rendered and "nan (" not in rendered.lower()
        assert "No confirmatory claim is made" in rendered
        assert "Claims A, B, and C were supported" not in rendered
