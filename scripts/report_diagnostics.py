"""Retrospective descriptive outcome audit; never selects or changes a policy.

Generate counts from preserved evaluations:
  uv run python scripts/report_diagnostics.py --runs runs_v2
Reproduce the summary from the packaged, hashed counts:
  uv run python scripts/report_diagnostics.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from emcomm.stats import bootstrap_mean_ci

ROOT = Path(__file__).resolve().parents[1]
OUTCOMES = ("no_pick", "wrong_pick", "correct_pick_no_delivery", "success")


def outcome(record: dict) -> str:
    pick, target = record["runner_pick"], record["target"]
    if record["success"]:
        if pick != target:
            raise ValueError("successful delivery must have picked the target")
        return "success"
    if pick < 0:
        return "no_pick"
    return "correct_pick_no_delivery" if pick == target else "wrong_pick"


def summarize_counts(rows: list[dict]) -> dict:
    import numpy as np

    groups = {}
    identities = set()
    for row in rows:
        identity = (row["condition"], row["split"], row["seed"])
        if (
            identity in identities
            or row["n"] != 200
            or sum(row[k] for k in OUTCOMES) != row["n"]
            or any(not isinstance(row[k], int) or row[k] < 0 for k in OUTCOMES)
        ):
            raise ValueError("counts must partition 200 episodes with unique seed/split identities")
        identities.add(identity)
        groups.setdefault((row["condition"], row["split"]), []).append(row)
    summary = {
        "scope": "Post-hoc descriptive analysis declared after benchmark outcomes were known; no tuning.",
        "uncertainty": "Percentile bootstrap of 20 training-seed means, conditional on fixed 200 scenarios per split.",
        "conditions": {},
    }
    for (condition, split), values in sorted(groups.items()):
        if len(values) != 20 or {v["seed"] for v in values} != set(range(20)):
            raise ValueError("each group must contain seeds 0–19")
        summary["conditions"].setdefault(condition, {})[split] = {
            k: dict(zip(("mean", "lo", "hi"), bootstrap_mean_ci(np.array([v[k] / v["n"] for v in values]))))
            for k in OUTCOMES
        }
    return summary


def collect(runs: Path) -> tuple[list[dict], dict]:
    rows, sources = [], {}
    for condition, intervention in (("comm", "intact"), ("oracle", "silenced")):
        for seed in range(20):
            path = runs / condition / f"seed_{seed:04d}" / "eval.jsonl"
            sources[str(path.relative_to(runs))] = hashlib.sha256(path.read_bytes()).hexdigest()
            by_split = {s: [] for s in ("held_out", "unseen")}
            for line in path.open():
                record = json.loads(line)
                if record["intervention"] == intervention and record["split"] in by_split:
                    if record["condition"] != condition or record["seed"] != seed:
                        raise ValueError("evaluation identity mismatch")
                    by_split[record["split"]].append(record)
            for split, records in by_split.items():
                if len(records) != 200 or {r["episode"] for r in records} != set(range(200)):
                    raise ValueError("incomplete fixed evaluation set")
                rows.append(
                    {
                        "condition": condition,
                        "split": split,
                        "seed": seed,
                        "n": len(records),
                        **{k: sum(outcome(r) == k for r in records) for k in OUTCOMES},
                    }
                )
    return rows, sources


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path)
    parser.add_argument("--out", type=Path, default=ROOT / "results/report_diagnostics")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    counts_path = args.out / "outcome_counts.json"
    if args.runs:
        rows, sources = collect(args.runs)
        counts_path.write_text(
            json.dumps(
                {"declared_on": "2026-11-09", "post_hoc": True, "source_sha256": sources, "rows": rows}, indent=2
            )
            + "\n"
        )
    counts = json.loads(counts_path.read_text())
    summary = summarize_counts(counts["rows"])
    summary["counts_sha256"] = hashlib.sha256(counts_path.read_bytes()).hexdigest()
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"Reproduced outcome summary from {len(counts['rows'])} seed/split count rows")


if __name__ == "__main__":
    main()
