"""v2 condition statistics and pre-registered claims (v2 spec sections 1, 5, and 8).

Usage: python -m emcomm.analysis_v2 --results results/v2
Reads results/v2/per_run.csv, writes results/v2/summary.json and summary.md.

Claims are tested in the fixed sequence A -> B -> C, each at alpha = 0.05
(95% CI). A later claim holds only if every earlier one does.

The 50% H4 reference is not an information-theoretic bound on the unseen split.
Its held-out matching lets a support-aware strategy identify a target from one
property alone, so exceeding 50% does not establish transfer of both properties.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from emcomm.analysis import EMPTY, MIN_GAIN, MIN_PERMUTATION_DROP, _by_seed, _ci, _independent, _paired
from emcomm.benchmark_v2 import ROW_FIELDS, RUN_LEVEL_FIELDS
from emcomm.channel import TRAIN_CHANNEL
from emcomm.config import CONDITIONS

MIN_UNSEEN_GAIN = 0.15
UNSEEN_SUCCESS_REFERENCE = 0.50
SPLIT_METRICS = ("success", "mean_return", "steps_success", "path_efficiency", "mi_first_msg", *ROW_FIELDS)
SPLIT_METRICS += RUN_LEVEL_FIELDS
FLOAT_FIELDS = ("n", "runner_pick_rate", *SPLIT_METRICS)
INTERVENTION_METRICS = ("success", "color_acc", "shape_acc", "twin_color", "twin_shape", "twin_opposite")
COMM_MODES = {
    "held_out": ("intact", "permuted", "permuted_w1", "permuted_w2", "silenced"),
    "unseen": ("intact", "permuted", "permuted_w1", "permuted_w2"),
}


def load_rows(path: str | Path) -> list[dict]:
    with open(path) as fh:
        rows = list(csv.DictReader(fh))
    for row in rows:
        row["seed"] = int(row["seed"])
        for key in FLOAT_FIELDS:
            row[key] = float(row[key]) if row.get(key) not in (None, "") else float("nan")
    return rows


def claims(summary: dict) -> dict:
    """Pre-registered H1-H4 and fixed-sequence claims A, B, C."""
    held, unseen = summary["comparisons"]["held_out"], summary["comparisons"]["unseen"]
    h1, h2 = held.get("comm_minus_no_comm", EMPTY), held.get("intact_minus_permuted", EMPTY)
    h3 = unseen.get("comm_minus_no_comm", EMPTY)
    h4 = summary["conditions"].get("comm", {}).get("unseen", {}).get("success", EMPTY)
    if any(x["mean"] is None for x in (h1, h2, h3, h4)):
        return {}
    tests = {
        "h1_gain_at_least_15pp": h1["mean"] >= MIN_GAIN,
        "h1_ci_excludes_zero": h1["lo"] > 0,
        "h2_drop_at_least_10pp": h2["mean"] >= MIN_PERMUTATION_DROP,
        "h2_ci_excludes_zero": h2["lo"] > 0,
        "h3_gain_at_least_15pp": h3["mean"] >= MIN_UNSEEN_GAIN,
        "h3_ci_excludes_zero": h3["lo"] > 0,
        "h4_mean_above_reference": h4["mean"] > UNSEEN_SUCCESS_REFERENCE,
        "h4_ci_above_reference": h4["lo"] > UNSEEN_SUCCESS_REFERENCE,
    }
    claim_a = all(v for k, v in tests.items() if k.startswith(("h1", "h2")))
    claim_b = claim_a and all(v for k, v in tests.items() if k.startswith("h3"))
    claim_c = claim_b and all(v for k, v in tests.items() if k.startswith("h4"))
    return {**tests, "claim_a": claim_a, "claim_b": claim_b, "claim_c": claim_c}


def summarize(rows: list[dict]) -> dict:
    learned = [c for c in CONDITIONS if any(r["condition"] == c for r in rows)]
    out: dict = {"conditions": {}, "scripted": {}, "comparisons": {"held_out": {}, "unseen": {}}, "claims": {}}

    def by_seed(c: str, split: str, metric: str = "success", mode: str | None = None) -> dict[int, float]:
        return _by_seed(rows, c, split, mode or TRAIN_CHANNEL[c], metric)

    for c in learned:
        entry = {split: {m: _ci(by_seed(c, split, m).values()) for m in SPLIT_METRICS} for split in COMM_MODES}
        train = by_seed(c, "train")
        entry["train_success"] = _ci(train.values())
        entry["generalization_gap"] = _paired(train, by_seed(c, "held_out"))
        out["conditions"][c] = entry
    for r in rows:
        if r["condition"].startswith("scripted_"):
            out["scripted"].setdefault(r["split"], {})[r["condition"]] = r["success"]
    if "comm" in learned:
        out["conditions"]["comm"]["interventions"] = {
            split: {
                mode: {m: _ci(by_seed("comm", split, m, mode).values()) for m in INTERVENTION_METRICS} for mode in modes
            }
            for split, modes in COMM_MODES.items()
        }
        for split, modes in COMM_MODES.items():
            comm, comps = by_seed("comm", split), out["comparisons"][split]
            for other in ("no_comm", "random_msg"):
                if other in learned:
                    comps[f"comm_minus_{other}"] = _independent(comm, by_seed(other, split))
            for mode in modes[1:]:
                comps[f"intact_minus_{mode}"] = _paired(comm, by_seed("comm", split, mode=mode))
        out["claims"] = claims(out)
    return out


def _pct(ci: dict) -> str:
    return "n/a" if ci["mean"] is None else f"{ci['mean'] * 100:.1f}% [{ci['lo'] * 100:.1f}, {ci['hi'] * 100:.1f}]"


def _pp(ci: dict) -> str:
    return "n/a" if ci["mean"] is None else f"{ci['mean'] * 100:+.1f} pp [{ci['lo'] * 100:+.1f}, {ci['hi'] * 100:+.1f}]"


def _num(ci: dict) -> str:
    return "n/a" if ci["mean"] is None else f"{ci['mean']:.2f} [{ci['lo']:.2f}, {ci['hi']:.2f}]"


def to_markdown(summary: dict) -> str:
    lines = [
        "| Condition | Seen (held_out) | Never-seen (unseen) | Train layouts | Topsim | Posdis |",
        "|---|---|---|---|---|---|",
    ]
    for c, e in summary["conditions"].items():
        lines.append(
            f"| {c} | {_pct(e['held_out']['success'])} | {_pct(e['unseen']['success'])} | {_pct(e['train_success'])} "
            f"| {_num(e['held_out']['topsim'])} | {_num(e['held_out']['posdis'])} |"
        )
    lines += ["", "| Scripted reference | held_out | unseen | train |", "|---|---|---|---|"]
    names = sorted({n for split in summary["scripted"].values() for n in split})
    for name in names:
        cells = [summary["scripted"].get(s, {}).get(name) for s in ("held_out", "unseen", "train")]
        lines.append(f"| {name} | " + " | ".join("" if v is None else f"{v * 100:.1f}%" for v in cells) + " |")
    for split, comps in summary["comparisons"].items():
        lines += ["", f"| Comparison ({split}) | Mean [95% CI] |", "|---|---|"]
        lines += [f"| {k} | {_pp(v)} |" for k, v in comps.items()]
    if summary["claims"]:
        c = summary["claims"]
        lines += [
            "",
            f"**Claim A (communication works): {c['claim_a']}** · **Claim B (generalizes): {c['claim_b']}** · "
            f"**Claim C (unseen success above 50% reference): {c['claim_c']}**",
            "",
        ]
        lines += [f"- {k}: {v}" for k, v in c.items() if not k.startswith("claim")]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", default="results/v2")
    args = parser.parse_args()
    results = Path(args.results)
    summary = summarize(load_rows(results / "per_run.csv"))
    (results / "summary.json").write_text(json.dumps(summary, indent=2))
    (results / "summary.md").write_text(to_markdown(summary))
    print(to_markdown(summary))


if __name__ == "__main__":
    main()
