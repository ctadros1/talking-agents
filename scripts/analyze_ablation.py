"""Analyze the declared post-hoc original-method ablation without changing frozen results."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from dataclasses import asdict, replace
from pathlib import Path

from emcomm.config import load_config
from emcomm.stats import bootstrap_diff_ci, bootstrap_mean_ci, paired_diff_ci
from emcomm.train import load_checkpoint

SEEDS = set(range(20))


def _values(rows: list[dict], mode: str) -> list[float]:
    values = {}
    for row in rows:
        if (row["condition"], row["split"], row["intervention"]) != ("comm", "held_out", mode):
            continue
        seed, success = int(row["seed"]), float(row["success"])
        if seed in values:
            raise ValueError(f"duplicate seed {seed} for {mode}")
        if not math.isfinite(success) or not 0 <= success <= 1:
            raise ValueError(f"invalid success for seed {seed}: {success}")
        values[seed] = success
    if set(values) != SEEDS:
        raise ValueError(f"expected exactly seeds 0–19 for {mode}, got {sorted(values)}")
    return [values[seed] for seed in sorted(SEEDS)]


def _estimate(value: tuple[float, float, float]) -> dict:
    return dict(zip(("mean", "lo", "hi"), value))


def summarize_comparison(biased_rows: list[dict], unbiased_rows: list[dict]) -> dict:
    biased, unbiased = _values(biased_rows, "intact"), _values(unbiased_rows, "intact")
    permuted = _values(unbiased_rows, "permuted")
    return {
        "n_seeds": len(SEEDS),
        "exploratory_post_hoc": True,
        "biased": {"success": _estimate(bootstrap_mean_ci(biased))},
        "unbiased": {
            "success": _estimate(bootstrap_mean_ci(unbiased)),
            "intact_minus_permuted": _estimate(paired_diff_ci(unbiased, permuted)),
        },
        "biased_minus_unbiased": _estimate(bootstrap_diff_ci(biased, unbiased)),
    }


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    root = args.root.resolve()
    run_root = root / "runs/ablation_unbiased"
    launch = json.loads((run_root / "launch.json").read_text())
    for name, digest in launch["files"].items():
        if _hash(root / name) != digest:
            raise ValueError(f"declared experimental source changed: {name}")
    env, original_ppo = load_config(root / "configs/base.toml")
    ablation_env, ppo = load_config(root / "configs/ablation_unbiased.toml")
    if ablation_env != env or ppo != replace(original_ppo, sig_coef=0.0):
        raise ValueError("the declared comparison changes more than the signalling weight")
    checkpoints = {}
    for seed in sorted(SEEDS):
        path = run_root / "comm" / f"seed_{seed:04d}" / "final.pt"
        _, saved_env, saved_ppo, saved = load_checkpoint(path)
        if (
            saved_env != env
            or saved_ppo != ppo
            or saved["seed"] != seed
            or saved["condition"] != "comm"
            or saved["update"] != ppo.n_updates
            or saved["env_steps"] != ppo.n_updates * ppo.steps_per_update
        ):
            raise ValueError(f"checkpoint config, identity or final budget mismatch: {path}")
        checkpoints[str(seed)] = _hash(path)
    biased_path = root / "results/per_run.csv"
    unbiased_path = root / "results/ablation_unbiased/per_run.csv"
    with biased_path.open() as fh:
        biased_rows = list(csv.DictReader(fh))
    with unbiased_path.open() as fh:
        unbiased_rows = list(csv.DictReader(fh))
    summary = summarize_comparison(biased_rows, unbiased_rows)
    summary["provenance"] = {
        "launch": launch,
        "biased_table_sha256": _hash(biased_path),
        "unbiased_table_sha256": _hash(unbiased_path),
        "checkpoint_sha256": checkpoints,
        "env": asdict(env),
        "ppo": asdict(ppo),
    }
    out = root / "results/ablation_unbiased/summary.json"
    out.write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    print(f"Complete exploratory ablation: {summary['n_seeds']} seeds -> {out}")


if __name__ == "__main__":
    main()
