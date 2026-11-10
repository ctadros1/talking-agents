"""Launch the (condition x seed) training sweep in parallel, resumably.

Usage: caffeinate -i uv run python -m emcomm.sweep --config configs/base.toml --seeds 0-19 --workers 6
Jobs are ordered seed-major so an interrupted sweep has equal seeds per condition.
A run is skipped if its final.pt exists; a partial run is restarted from scratch.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from emcomm.config import CONDITIONS


def parse_seeds(spec: str) -> list[int]:
    seeds: list[int] = []
    for part in spec.split(","):
        lo, _, hi = part.partition("-")
        seeds += list(range(int(lo), int(hi or lo) + 1))
    return seeds


def run_dir(root: Path, condition: str, seed: int) -> Path:
    return root / condition / f"seed_{seed:04d}"


def plan_jobs(root: Path, conditions: list[str], seeds: list[int]) -> list[tuple[str, int]]:
    return [(c, s) for s in seeds for c in conditions if not (run_dir(root, c, s) / "final.pt").exists()]


def _launch(config: str, root: Path, condition: str, seed: int, extra: list[str]) -> tuple[str, int, int]:
    out = run_dir(root, condition, seed)
    out.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        "-m",
        "emcomm.train",
        "--config",
        config,
        "--condition",
        condition,
        "--seed",
        str(seed),
        "--out",
        str(out),
        *extra,
    ]
    with open(out / "train.log", "w") as log:
        code = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT).returncode
    return condition, seed, code


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--root", default="runs")
    parser.add_argument("--seeds", required=True, help="e.g. 0-19 or 1000-1002")
    parser.add_argument("--conditions", default=",".join(CONDITIONS))
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--total-env-steps", type=int, default=None)
    args = parser.parse_args()
    root = Path(args.root)
    extra = ["--total-env-steps", str(args.total_env_steps)] if args.total_env_steps else []
    jobs = plan_jobs(root, args.conditions.split(","), parse_seeds(args.seeds))
    print(f"{len(jobs)} runs to train")
    failed = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for condition, seed, code in pool.map(lambda job: _launch(args.config, root, *job, extra), jobs):
            print(f"{condition} seed {seed}: {'ok' if code == 0 else f'FAILED ({code})'}", flush=True)
            if code:
                failed.append((condition, seed))
    if failed:
        sys.exit(f"{len(failed)} runs failed: {failed}")


if __name__ == "__main__":
    main()
