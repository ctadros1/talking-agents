import csv

import numpy as np

from emcomm.analysis_v2 import claims, load_rows, summarize, to_markdown
from emcomm.benchmark_v2 import ROW_FIELDS, RUN_FIELDS_V2, RUN_LEVEL_FIELDS

OTHER = ("mean_return", "steps_success", "path_efficiency", "mi_first_msg", "runner_pick_rate", *ROW_FIELDS)


def _row(condition, seed, split, mode, success):
    return {
        "condition": condition,
        "seed": seed,
        "split": split,
        "intervention": mode,
        "n": 200.0,
        "success": success + max(seed, 0) / 1000,
        **dict.fromkeys(OTHER, 0.5),
        **dict.fromkeys(RUN_LEVEL_FIELDS, 0.7),
    }


def _rows(comm=0.8, no_comm=0.25, permuted=0.3, unseen_comm=0.7, unseen_no_comm=0.25):
    rows = [_row("scripted_color_only", -1, "held_out", "intact", 0.5)]
    for s in range(20):
        for mode, value in (
            ("intact", comm),
            ("permuted", permuted),
            ("permuted_w1", 0.5),
            ("permuted_w2", 0.5),
        ):
            rows.append(_row("comm", s, "held_out", mode, value))
        rows.append(_row("comm", s, "held_out", "silenced", 0.25))
        for mode, value in (
            ("intact", unseen_comm),
            ("permuted", 0.25),
            ("permuted_w1", 0.4),
            ("permuted_w2", 0.4),
        ):
            rows.append(_row("comm", s, "unseen", mode, value))
        rows.append(_row("comm", s, "train", "intact", comm + 0.02))
        for split, value in (("held_out", no_comm), ("unseen", unseen_no_comm), ("train", no_comm)):
            rows.append(_row("no_comm", s, split, "silenced", value))
    return rows


def test_all_claims_hold():
    summary = summarize(_rows())
    assert summary["claims"]["claim_a"] and summary["claims"]["claim_b"] and summary["claims"]["claim_c"]
    assert summary["claims"]["h4_mean_above_reference"] and summary["claims"]["h4_ci_above_reference"]
    assert "h4_mean_above_ceiling" not in summary["claims"]
    assert abs(summary["comparisons"]["unseen"]["comm_minus_no_comm"]["mean"] - 0.45) < 1e-9
    assert abs(summary["conditions"]["comm"]["generalization_gap"]["mean"] - 0.02) < 1e-9
    assert summary["scripted"]["held_out"]["scripted_color_only"] == 0.5
    assert "permuted_w1" in summary["conditions"]["comm"]["interventions"]["unseen"]


def test_small_unseen_gain_fails_b_and_c():
    c = summarize(_rows(unseen_comm=0.3))["claims"]
    assert c["claim_a"] and not c["h3_gain_at_least_15pp"] and not c["claim_b"] and not c["claim_c"]


def test_unseen_success_below_reference_passes_b_not_c():
    c = summarize(_rows(unseen_comm=0.45))["claims"]
    assert c["claim_b"] and not c["h4_mean_above_reference"] and not c["claim_c"]


def test_fixed_sequence_blocks_b_when_a_fails():
    c = summarize(_rows(permuted=0.78))["claims"]
    assert not c["claim_a"] and c["h3_gain_at_least_15pp"] and c["h3_ci_excludes_zero"] and not c["claim_b"]


def test_claims_empty_without_components():
    assert claims({"comparisons": {"held_out": {}, "unseen": {}}, "conditions": {}}) == {}


def test_markdown_and_csv_round_trip(tmp_path):
    rows = _rows()
    path = tmp_path / "per_run.csv"
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=RUN_FIELDS_V2)
        writer.writeheader()
        writer.writerows([{**r, "topsim": ""} if r["seed"] == -1 else r for r in rows])
    loaded = load_rows(path)
    assert np.isnan(loaded[0]["topsim"]) and loaded[1]["topsim"] == 0.7
    md = to_markdown(summarize(loaded))
    assert "Claim A" in md and "Never-seen" in md
    assert "Claim C (unseen success above 50% reference)" in md
    assert "both properties transfer" not in md
