"""Gate G1: acceptance rates and scripted headroom on held-out layouts.

Usage: uv run python scripts/calibrate.py [--config configs/base.toml] [--out results/calibration.json]
       (v2: --config configs/v2.toml --out results/v2/calibration.json)
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from emcomm import benchmark_v2
from emcomm.benchmark import evaluate_scripted
from emcomm.config import load_config
from emcomm.layouts import make_layout_splits, try_geometry

CHECKS = {
    "min_layout_acceptance >= 0.20": lambda c: c["min_layout_acceptance"] >= 0.20,
    "scripted_comm >= 0.98": lambda c: c["scripted"]["scripted_comm"] >= 0.98,
    "scripted_oracle >= 0.98": lambda c: c["scripted"]["scripted_oracle"] >= 0.98,
    "0.18 <= scripted_no_comm <= 0.32": lambda c: 0.18 <= c["scripted"]["scripted_no_comm"] <= 0.32,
    "0.15 <= scripted_random_msg <= 0.35": lambda c: 0.15 <= c["scripted"]["scripted_random_msg"] <= 0.35,
    "headroom (comm - no_comm) >= 0.50": lambda c: (
        c["scripted"]["scripted_comm"] - c["scripted"]["scripted_no_comm"] >= 0.50
    ),
}

SEEN_AND_UNSEEN = ("held_out", "unseen")
CHECKS_V2 = {  # v2 spec section 6
    "min_layout_acceptance >= 0.20": lambda c: c["min_layout_acceptance"] >= 0.20,
    "scripted_comm >= 0.98 on held_out and unseen": lambda c: (
        min(c["scripted"][s]["scripted_comm"] for s in SEEN_AND_UNSEEN) >= 0.98
    ),
    "scripted_oracle >= 0.98 on held_out and unseen": lambda c: (
        min(c["scripted"][s]["scripted_oracle"] for s in SEEN_AND_UNSEEN) >= 0.98
    ),
    "0.18 <= scripted_no_comm <= 0.32": lambda c: 0.18 <= c["scripted"]["held_out"]["scripted_no_comm"] <= 0.32,
    "0.40 <= scripted_color_only <= 0.60": lambda c: 0.40 <= c["scripted"]["held_out"]["scripted_color_only"] <= 0.60,
    "0.40 <= scripted_shape_only <= 0.60": lambda c: 0.40 <= c["scripted"]["held_out"]["scripted_shape_only"] <= 0.60,
    "headroom (comm - no_comm) >= 0.50": lambda c: (
        c["scripted"]["held_out"]["scripted_comm"] - c["scripted"]["held_out"]["scripted_no_comm"] >= 0.50
    ),
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/base.toml")
    parser.add_argument("--out", default="results/calibration.json")
    args = parser.parse_args()
    cfg, _ = load_config(args.config)
    train, held_out = make_layout_splits(cfg)
    rng = np.random.default_rng(0)
    acceptance = [
        np.mean([try_geometry(layout, rng, cfg) is not None for _ in range(300)]) for layout in train + held_out
    ]
    v2 = cfg.task == "compositional"
    if v2:
        scripted: dict = {}
        for r in benchmark_v2.evaluate_scripted(cfg):
            scripted.setdefault(r["split"], {})[r["condition"]] = r["success"]
    else:
        scripted = {r["condition"]: r["success"] for r in evaluate_scripted(cfg) if r["split"] == "held_out"}
    calib = {
        "mean_layout_acceptance": float(np.mean(acceptance)),
        "min_layout_acceptance": float(np.min(acceptance)),
        "scripted": scripted,
    }
    calib["checks"] = {name: bool(check(calib)) for name, check in (CHECKS_V2 if v2 else CHECKS).items()}
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(calib, indent=2))
    print(json.dumps(calib, indent=2))
    if not all(calib["checks"].values()):
        sys.exit("Gate G1 FAILED")
    print("Gate G1 PASSED")


if __name__ == "__main__":
    main()
