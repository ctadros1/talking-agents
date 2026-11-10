"""Run fresh final-checkpoint inference and export a separate offline demonstration.

Usage: uv run python scripts/demo_inference.py --phase both --out demo_output/rehearsal
This does not train or change the committed scientific results or viewers.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from datetime import UTC, datetime
from pathlib import Path

import torch

from emcomm.evaluate import evaluate_checkpoint
from emcomm.replays import build_bundle as build_v1
from emcomm.replays_v2 import build_bundle as build_v2

ROOT = Path(__file__).resolve().parents[1]


def new_output(root: Path, out: Path) -> Path:
    out = out.resolve()
    try:
        relative = out.relative_to(root.resolve())
    except ValueError as exc:
        raise ValueError("demo output must be inside the project") from exc
    if not relative.parts or relative.parts[0] != "demo_output":
        raise ValueError("demo output must be a new directory under demo_output/")
    if out.exists():
        raise ValueError(f"demo output already exists; choose a fresh name: {out}")
    out.mkdir(parents=True)
    return out


def checkpoint_root(root: Path, phase: str) -> tuple[Path, int]:
    seed = 19 if phase == "v1" else 9
    packaged = root / "demo_checkpoints" / phase
    original = root / ("runs" if phase == "v1" else "runs_v2")
    return (packaged if packaged.is_dir() else original), seed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("v1", "v2", "both"), default="both")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    out = new_output(ROOT, args.out or ROOT / "demo_output" / stamp)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    receipt = {"executed_at_utc": datetime.now(UTC).isoformat(), "fresh_inference": True, "phases": {}}
    viewer = out / "viewer"
    shutil.copytree(ROOT / "viewer", viewer)
    for phase in ("v1", "v2") if args.phase == "both" else (args.phase,):
        runs, seed = checkpoint_root(ROOT, phase)
        path = runs / "comm" / f"seed_{seed:04d}" / "final.pt"
        if not path.is_file():
            raise FileNotFoundError(f"missing final checkpoint: {path}")
        print(f"\n{phase}: loading final.pt, comm seed {seed}; running fresh policy inference", flush=True)
        episodes, rows = evaluate_checkpoint(path)
        (out / f"{phase}-episodes.jsonl").write_text("".join(json.dumps(row) + "\n" for row in episodes))
        (out / f"{phase}-metrics.json").write_text(json.dumps(rows, indent=2) + "\n")
        bundle = (build_v1 if phase == "v1" else build_v2)(runs, seed)
        filename = "replays.js" if phase == "v1" else "replays_v2.js"
        global_name = "REPLAYS" if phase == "v1" else "REPLAYS_V2"
        (viewer / "data" / filename).write_text(f"window.{global_name} = " + json.dumps(bundle) + ";\n")
        for row in rows:
            print(
                f"  {row['split']:8s} {row['intervention']:12s} success={row['success']:.3f}  episodes={row['n']}",
                flush=True,
            )
        receipt["phases"][phase] = {
            "seed": seed,
            "checkpoint_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "evaluated_episode_records": len(episodes),
            "summary_rows": len(rows),
            "replay_episodes": len(bundle["episodes"]),
        }
    (out / "EXECUTION_RECEIPT.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(f"\nFresh execution complete. Output: {out}")
    print(f"Open {viewer / 'index.html'} or {viewer / 'v2.html'} for the new replays.")
    print("These are representative-seed demonstrations; aggregate claims come from the full seed summaries.")


if __name__ == "__main__":
    main()
