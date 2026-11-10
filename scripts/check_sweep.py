"""Gate G3: every (condition, seed) run finished, with equal budgets and finite metrics.

Usage: uv run python scripts/check_sweep.py --runs runs --seeds 0-19 [--conditions comm,no_comm]
"""

import argparse
import csv
import math
import sys
from pathlib import Path

from emcomm.config import CONDITIONS
from emcomm.sweep import parse_seeds, run_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", default="runs")
    parser.add_argument("--seeds", default="0-19")
    parser.add_argument("--conditions", default=",".join(CONDITIONS))
    args = parser.parse_args()
    conditions = args.conditions.split(",")
    problems, budgets = [], set()
    for condition in conditions:
        for seed in parse_seeds(args.seeds):
            out = run_dir(Path(args.runs), condition, seed)
            if not (out / "final.pt").exists():
                problems.append(f"missing final.pt: {out}")
                continue
            with open(out / "metrics.csv") as fh:
                rows = list(csv.DictReader(fh))
            budgets.add(rows[-1]["env_steps"])
            bad = [
                r["update"]
                for r in rows
                for k in ("policy_loss_0", "value_loss_0", "policy_loss_1", "value_loss_1")
                if not math.isfinite(float(r[k]))
            ]
            if bad:
                problems.append(f"non-finite losses in {out} at updates {bad[:5]}")
    if len(budgets) > 1:
        problems.append(f"unequal budgets: {sorted(budgets)}")
    if problems:
        sys.exit("Gate G3 FAILED\n" + "\n".join(problems))
    print(f"Gate G3 PASSED: {len(conditions) * len(parse_seeds(args.seeds))} runs, env_steps={budgets.pop()}")


if __name__ == "__main__":
    main()
