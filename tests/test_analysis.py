import csv

from emcomm.analysis import load_rows, summarize, to_markdown
from emcomm.benchmark import RUN_FIELDS


def _row(condition, seed, split, intervention, success):
    return {
        "condition": condition,
        "seed": seed,
        "split": split,
        "intervention": intervention,
        "n": 200,
        "success": success,
        "mean_return": success,
        "steps_success": 15.0,
        "path_efficiency": 0.8,
        "mi_first_msg": 1.5,
        "runner_pick_rate": 1.0,
    }


def _write(tmp_path, comm, no_comm, permuted):
    rows = [_row("scripted_comm", -1, "held_out", "intact", 1.0)]
    for s in range(20):
        rows += [
            _row("comm", s, "held_out", "intact", comm),
            _row("comm", s, "held_out", "permuted", permuted),
            _row("comm", s, "held_out", "silenced", 0.25),
            _row("comm", s, "train", "intact", comm + 0.02),
            _row("no_comm", s, "held_out", "silenced", no_comm),
            _row("no_comm", s, "train", "silenced", no_comm),
        ]
    path = tmp_path / "per_run.csv"
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=RUN_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    return load_rows(path)


def test_criterion_met(tmp_path):
    summary = summarize(_write(tmp_path, comm=0.8, no_comm=0.25, permuted=0.3))
    assert summary["criterion"]["met"] is True
    assert abs(summary["comparisons"]["comm_minus_no_comm"]["mean"] - 0.55) < 1e-9
    assert abs(summary["conditions"]["comm"]["generalization_gap"]["mean"] - 0.02) < 1e-9
    assert "Pre-registered success criterion met: True" in to_markdown(summary)


def test_criterion_fails_without_permutation_drop(tmp_path):
    summary = summarize(_write(tmp_path, comm=0.8, no_comm=0.25, permuted=0.78))
    assert summary["criterion"]["gain_at_least_15pp"] and not summary["criterion"]["permutation_drop_at_least_10pp"]
    assert summary["criterion"]["met"] is False
