"""Export representative v2 episodes for the static 3D-look viewer.

The representative comm seed's intact records select episodes with broad color
and shape coverage, two held-out failures, and up to two unseen examples. Each
selected episode is replayed under every viewer variant on the same split.

Usage: python -m emcomm.replays_v2 --runs runs_v2 --results results/v2 --out viewer/data/replays_v2.js
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from emcomm import benchmark_v2
from emcomm.analysis_v2 import load_rows
from emcomm.benchmark import EVAL_SEED, N_EVAL_EPISODES
from emcomm.combos import COLOR_NAMES, SHAPE_NAMES
from emcomm.evaluate import LearnedTeam
from emcomm.lexicon import word_tables
from emcomm.plots import representative_seed
from emcomm.replays import spec_json
from emcomm.rollout import run_episodes
from emcomm.train import load_checkpoint

VARIANTS = (("comm", "intact"), ("comm", "permuted"), ("comm", "permuted_w1"), ("no_comm", "silenced"))


def pick_v2(held_out_records: list[dict], unseen_records: list[dict]) -> dict[str, list[int]]:
    """Choose representative indices, preserving honest coverage when sparse."""
    chosen: list[int] = []
    uncovered_colors = {r["target_attrs"][0] for r in held_out_records if r["success"]}
    uncovered_shapes = {r["target_attrs"][1] for r in held_out_records if r["success"]}
    candidates = [r for r in held_out_records if r["success"]]

    while uncovered_colors or uncovered_shapes:
        best = None
        best_gain = 0
        for record in candidates:
            color, shape = record["target_attrs"]
            gain = int(color in uncovered_colors) + int(shape in uncovered_shapes)
            if gain > best_gain:
                best, best_gain = record, gain
        if best is None:
            break
        chosen.append(best["episode"])
        color, shape = best["target_attrs"]
        uncovered_colors.discard(color)
        uncovered_shapes.discard(shape)
        candidates.remove(best)

    failures = [r["episode"] for r in held_out_records if not r["success"] and r["episode"] not in chosen]
    chosen.extend(failures[:2])

    wins = [r["episode"] for r in unseen_records if r["success"]]
    losses = [r["episode"] for r in unseen_records if not r["success"]]
    unseen = wins[:1] + losses[:1]
    remaining = [i for i in wins[1:] + losses[1:] if i not in unseen]
    unseen.extend(remaining[: max(0, 2 - len(unseen))])
    return {"held_out": sorted(set(chosen)), "unseen": sorted(set(unseen))}


def build_bundle(runs: Path, seed: int) -> dict:
    """Build the viewer payload for one seed's checkpoint pair."""
    nets, cfg, _, _ = load_checkpoint(runs / "comm" / f"seed_{seed:04d}" / "final.pt")
    sets = benchmark_v2.episode_sets(cfg)
    first_records = {
        split: run_episodes(
            cfg,
            sets[split],
            LearnedTeam(nets, N_EVAL_EPISODES, EVAL_SEED),
            "comm",
            "intact",
            seed=EVAL_SEED,
        )[0]
        for split in ("held_out", "unseen")
    }
    chosen = pick_v2(first_records["held_out"], first_records["unseen"])

    variants: dict[str, dict] = {}
    checkpoints: dict[str, tuple] = {"comm": (nets, cfg)}
    for condition in ("no_comm",):
        nets, condition_cfg, _, _ = load_checkpoint(runs / condition / f"seed_{seed:04d}" / "final.pt")
        checkpoints[condition] = (nets, condition_cfg)

    for condition, mode in VARIANTS:
        nets, condition_cfg = checkpoints[condition]
        replay_rows: dict[str, dict] = {}
        for split, episode_ids in chosen.items():
            team = LearnedTeam(nets, N_EVAL_EPISODES, EVAL_SEED)
            records, traces = run_episodes(
                condition_cfg,
                sets[split],
                team,
                condition,
                mode,
                seed=EVAL_SEED,
                record=tuple(episode_ids),
            )
            replay_rows.update({f"{split}/{i}": {"record": records[i], "steps": traces[i]} for i in episode_ids})
        variants[f"{condition}/{mode}"] = replay_rows

    episodes = {
        f"{split}/{i}": {
            **spec_json(sets[split][i]),
            "split": split,
            "attrs": sets[split][i].item_attrs.tolist(),
        }
        for split, episode_ids in chosen.items()
        for i in episode_ids
    }
    return {
        "seed": seed,
        "config": {
            "grid_size": cfg.grid_size,
            "wall_col": cfg.wall_col,
            "view_radius": cfg.view_radius,
            "max_steps": cfg.max_steps,
            "vocab_size": cfg.vocab_size,
            "n_values": cfg.n_values,
            "msg_words": cfg.msg_words,
            "colors": list(COLOR_NAMES),
            "shapes": list(SHAPE_NAMES),
        },
        "episodes": episodes,
        "variants": variants,
        "lexicon": word_tables(first_records["held_out"] + first_records["unseen"], cfg.n_values, cfg.vocab_size),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", default="runs_v2")
    parser.add_argument("--results", default="results/v2")
    parser.add_argument("--seed", type=int, default=None, help="default: representative (median) comm seed")
    parser.add_argument("--out", default="viewer/data/replays_v2.js")
    args = parser.parse_args()
    seed = args.seed if args.seed is not None else representative_seed(load_rows(Path(args.results) / "per_run.csv"))
    bundle = build_bundle(Path(args.runs), seed)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("window.REPLAYS_V2 = " + json.dumps(bundle) + ";\n")
    print(f"seed {seed}: {len(bundle['episodes'])} episodes x {len(bundle['variants'])} variants -> {out}")


if __name__ == "__main__":
    main()
