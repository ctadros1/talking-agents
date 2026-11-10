"""Gates G2 / V2-G2: evaluate pilot runs on TRAINING layouts only and print the checks.

Usage: uv run python scripts/pilot_report.py --runs runs/pilot [--out results/pilot_rung0.json]
Held-out layouts are never touched here: pilots must not see the confirmatory split.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from emcomm.evaluate import evaluate_checkpoint


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", default="runs/pilot")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    by_condition: dict[str, list[float]] = {}
    for path in sorted(Path(args.runs).glob("*/seed_*/final.pt")):
        _, rows = evaluate_checkpoint(path, splits=("train",))
        (row,) = rows  # the train split is evaluated once, under the condition's own channel
        by_condition.setdefault(row["condition"], []).append(row["success"])
        print(f"{path.parent}: train-layout success {row['success']:.3f}")
    means = {c: float(np.mean(v)) for c, v in by_condition.items()}
    gain = means.get("comm", 0.0) - means.get("no_comm", 0.0)
    checks = {
        "oracle >= 0.80": means.get("oracle", 0.0) >= 0.80,
        "no_comm <= 0.32": means.get("no_comm", 1.0) <= 0.32,
        "comm - no_comm >= 0.15": gain >= 0.15,
    }
    report = {"per_seed": by_condition, "means": means, "checks": checks, "comm_minus_no_comm": gain}
    print(json.dumps(report, indent=2))
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(report, indent=2))
    print("Pilot training checks:", "PASSED" if all(checks.values()) else "FAILED")


if __name__ == "__main__":
    main()
