"""Gate V2-G4: the v2 code reproduces a frozen v1 result exactly.

Re-evaluates runs/comm/seed_0000/final.pt (a v1 benchmark run) and compares every field of every row
with its row in results/per_run.csv, as written to CSV.

Usage: uv run python scripts/check_v1_reproduces.py
"""

import csv
import sys

from emcomm.benchmark import RUN_FIELDS
from emcomm.evaluate import evaluate_checkpoint


def main() -> None:
    with open("results/per_run.csv") as fh:
        frozen = {
            (r["split"], r["intervention"]): r
            for r in csv.DictReader(fh)
            if r["condition"] == "comm" and r["seed"] == "0"
        }
    _, rows = evaluate_checkpoint("runs/comm/seed_0000/final.pt")
    problems = [
        f"{row['split']}/{row['intervention']} {key}: {frozen[(row['split'], row['intervention'])][key]} != {row[key]}"
        for row in rows
        for key in RUN_FIELDS
        if str(row[key]) != frozen[(row["split"], row["intervention"])][key]
    ]
    if len(rows) != len(frozen) or problems:
        sys.exit("v1 reproduction FAILED\n" + "\n".join(problems or [f"{len(rows)} rows vs {len(frozen)} frozen"]))
    print(f"v1 reproduction PASSED: {len(rows)} rows x {len(RUN_FIELDS)} fields identical")


if __name__ == "__main__":
    main()
