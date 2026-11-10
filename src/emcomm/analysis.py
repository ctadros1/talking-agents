"""Condition-level statistics and the pre-registered success criterion.

Usage: python -m emcomm.analysis --results results
Reads results/per_run.csv, writes results/summary.json and results/summary.md.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from emcomm.channel import TRAIN_CHANNEL
from emcomm.config import CONDITIONS
from emcomm.stats import bootstrap_diff_ci, bootstrap_mean_ci, paired_diff_ci

MIN_GAIN = 0.15  # comm - no_comm held-out success, pre-registered
MIN_PERMUTATION_DROP = 0.10  # comm intact - comm permuted, pre-registered
METRICS = ("success", "mean_return", "steps_success", "path_efficiency", "mi_first_msg")


def load_rows(path: str | Path) -> list[dict]:
    with open(path) as fh:
        rows = list(csv.DictReader(fh))
    for row in rows:
        row["seed"] = int(row["seed"])
        for key in ("n", *METRICS, "runner_pick_rate"):
            row[key] = float(row[key])
    return rows


def _by_seed(rows: list[dict], condition: str, split: str, intervention: str, metric: str) -> dict[int, float]:
    return {
        r["seed"]: r[metric]
        for r in rows
        if r["condition"] == condition and r["split"] == split and r["intervention"] == intervention
    }


EMPTY = {"mean": None, "lo": None, "hi": None, "n": 0}


def _ci(values) -> dict:
    values = [v for v in values if not np.isnan(v)]
    if not values:
        return dict(EMPTY)
    mean, lo, hi = bootstrap_mean_ci(values)
    return {"mean": mean, "lo": lo, "hi": hi, "n": len(values)}


def _independent(a: dict[int, float], b: dict[int, float]) -> dict:
    """Difference of condition means; seeds in a and b are different training runs."""
    if not a or not b:
        return dict(EMPTY)
    mean, lo, hi = bootstrap_diff_ci(list(a.values()), list(b.values()))
    return {"mean": mean, "lo": lo, "hi": hi, "n": min(len(a), len(b))}


def _paired(a: dict[int, float], b: dict[int, float]) -> dict:
    """Mean per-seed difference; a[s] and b[s] come from the same training run."""
    seeds = sorted(set(a) & set(b))
    if not seeds:
        return dict(EMPTY)
    mean, lo, hi = paired_diff_ci([a[s] for s in seeds], [b[s] for s in seeds])
    return {"mean": mean, "lo": lo, "hi": hi, "n": len(seeds)}


def summarize(rows: list[dict]) -> dict:
    learned = [c for c in CONDITIONS if any(r["condition"] == c for r in rows)]
    out: dict = {"conditions": {}, "scripted": {}, "comparisons": {}, "criterion": {}}

    def held_out(c: str, metric: str = "success", mode: str | None = None) -> dict[int, float]:
        return _by_seed(rows, c, "held_out", mode or TRAIN_CHANNEL[c], metric)

    for c in learned:
        entry = {m: _ci(held_out(c, m).values()) for m in METRICS}
        train = _by_seed(rows, c, "train", TRAIN_CHANNEL[c], "success")
        entry["train_success"] = _ci(train.values())
        entry["generalization_gap"] = _paired(train, held_out(c))
        out["conditions"][c] = entry
    for r in rows:
        if r["condition"].startswith("scripted_") and r["split"] == "held_out":
            out["scripted"][r["condition"]] = r["success"]

    if "comm" in learned:
        comm = held_out("comm")
        for other in ("no_comm", "random_msg"):
            if other in learned:
                out["comparisons"][f"comm_minus_{other}"] = _independent(comm, held_out(other))
        for mode in ("permuted", "shuffled", "silenced"):  # shuffled exists only on the vocal channel
            drop = _paired(comm, held_out("comm", mode=mode))
            if drop["n"]:
                out["comparisons"][f"intact_minus_{mode}"] = drop

    gain = out["comparisons"].get("comm_minus_no_comm", EMPTY)
    drop = out["comparisons"].get("intact_minus_permuted", EMPTY)
    if gain["mean"] is not None and drop["mean"] is not None:
        out["criterion"] = {
            "gain_at_least_15pp": gain["mean"] >= MIN_GAIN,
            "gain_ci_excludes_zero": gain["lo"] > 0,
            "permutation_drop_at_least_10pp": drop["mean"] >= MIN_PERMUTATION_DROP,
            "permutation_drop_ci_excludes_zero": drop["lo"] > 0,
        }
        out["criterion"]["met"] = all(out["criterion"].values())
    return out


def _fmt(ci: dict, kind: str = "pct") -> str:
    """kind: 'pct' for a success level, 'pp' for a difference, 'num' for a raw number."""
    if ci["mean"] is None:
        return "n/a"
    scale = 1 if kind == "num" else 100
    unit = {"pct": "%", "pp": " pp", "num": ""}[kind]
    return f"{ci['mean'] * scale:.1f}{unit} [{ci['lo'] * scale:.1f}, {ci['hi'] * scale:.1f}]"


def to_markdown(summary: dict) -> str:
    lines = [
        "| Condition | Held-out success | Train-layout success | Gen. gap "
        "| Steps / success | MI(target; first msg) bits |",
        "|---|---|---|---|---|---|",
    ]
    for c, e in summary["conditions"].items():
        lines.append(
            f"| {c} | {_fmt(e['success'])} | {_fmt(e['train_success'])} | {_fmt(e['generalization_gap'], 'pp')} "
            f"| {_fmt(e['steps_success'], 'num')} | {_fmt(e['mi_first_msg'], 'num')} |"
        )
    for name, value in summary["scripted"].items():
        lines.append(f"| {name} (reference) | {value * 100:.1f}% | | | | |")
    lines += ["", "| Comparison | Mean [95% CI] |", "|---|---|"]
    lines += [f"| {k} | {_fmt(v, 'pp')} |" for k, v in summary["comparisons"].items()]
    if summary["criterion"]:
        lines += ["", f"**Pre-registered success criterion met: {summary['criterion']['met']}**", ""]
        lines += [f"- {k}: {v}" for k, v in summary["criterion"].items() if k != "met"]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", default="results")
    args = parser.parse_args()
    results = Path(args.results)
    summary = summarize(load_rows(results / "per_run.csv"))
    (results / "summary.json").write_text(json.dumps(summary, indent=2))
    (results / "summary.md").write_text(to_markdown(summary))
    print(to_markdown(summary))


if __name__ == "__main__":
    main()
