"""Export replay episodes for the static viewer (viewer/data/replays.js).

The same held-out episodes are replayed for the representative comm seed with
intact and permuted messages (shuffled on the vocal channel), and for the same-seed no-comm run.

Usage: python -m emcomm.replays --runs runs --results results --out viewer/data/replays.js
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from emcomm.analysis import load_rows
from emcomm.benchmark import EVAL_SEED, N_EVAL_EPISODES, episode_sets
from emcomm.evaluate import LearnedTeam
from emcomm.layouts import EpisodeSpec
from emcomm.plots import representative_seed
from emcomm.rollout import run_episodes
from emcomm.train import load_checkpoint

VARIANTS = {
    "discrete": (("comm", "intact"), ("comm", "permuted"), ("no_comm", "silenced")),
    "vocal": (("comm", "intact"), ("comm", "shuffled"), ("no_comm", "silenced")),
}


def pick_episodes(records: list[dict], n_targets: int, per_target: int = 1, failures: int = 2) -> list[int]:
    """One success per target (shows the whole codebook) plus a couple of failures."""
    chosen: list[int] = []
    for k in range(n_targets):
        chosen += [r["episode"] for r in records if r["success"] and r["target"] == k][:per_target]
    chosen += [r["episode"] for r in records if not r["success"]][:failures]
    return sorted(chosen)


def spec_json(spec: EpisodeSpec) -> dict:
    return {
        "layout_id": spec.layout.layout_id,
        "walls": spec.layout.walls.astype(int).tolist(),
        "target": spec.target,
        "items": spec.item_pos.tolist(),
        "depot": list(spec.depot),
        "runner_opt": spec.runner_opt,
    }


def build_bundle(runs: Path, seed: int) -> dict:
    specs = None
    variants, chosen = {}, None
    channel = load_checkpoint(runs / "comm" / f"seed_{seed:04d}" / "final.pt")[1].channel
    for condition, mode in VARIANTS[channel]:
        nets, cfg, _, _ = load_checkpoint(runs / condition / f"seed_{seed:04d}" / "final.pt")
        specs = specs or episode_sets(cfg)["held_out"]
        team = LearnedTeam(nets, N_EVAL_EPISODES, EVAL_SEED)
        if chosen is None:  # choose episodes from the intact comm run, then replay exactly those everywhere
            records, _ = run_episodes(cfg, specs, team, condition, mode, seed=EVAL_SEED)
            chosen = pick_episodes(records, cfg.n_item_types)
            team = LearnedTeam(nets, N_EVAL_EPISODES, EVAL_SEED)
        records, traces = run_episodes(cfg, specs, team, condition, mode, seed=EVAL_SEED, record=tuple(chosen))
        variants[f"{condition}/{mode}"] = {str(i): {"record": records[i], "steps": traces[i]} for i in chosen}
    return {
        "seed": seed,
        "config": {
            "grid_size": cfg.grid_size,
            "wall_col": cfg.wall_col,
            "view_radius": cfg.view_radius,
            "max_steps": cfg.max_steps,
            "vocab_size": cfg.vocab_size,
            "channel": cfg.channel,
        },
        "episodes": {str(i): spec_json(specs[i]) for i in chosen},
        "variants": variants,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", default="runs")
    parser.add_argument("--results", default="results")
    parser.add_argument("--seed", type=int, default=None, help="default: representative (median) comm seed")
    parser.add_argument("--out", default="viewer/data/replays.js")
    args = parser.parse_args()
    seed = args.seed if args.seed is not None else representative_seed(load_rows(Path(args.results) / "per_run.csv"))
    bundle = build_bundle(Path(args.runs), seed)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("window.REPLAYS = " + json.dumps(bundle) + ";\n")
    print(f"seed {seed}: {len(bundle['episodes'])} episodes x {len(bundle['variants'])} variants -> {out}")


if __name__ == "__main__":
    main()
