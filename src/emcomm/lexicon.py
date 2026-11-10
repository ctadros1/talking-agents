"""The learned dictionary and message log for the v2 compositional task.

Usage: python -m emcomm.lexicon --runs runs_v2 --results results/v2
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from emcomm.analysis_v2 import load_rows
from emcomm.combos import COLOR_NAMES, PROPERTIES, SHAPE_NAMES
from emcomm.config import load_config
from emcomm.plots import representative_seed

LOG_FIELDS = [
    "seed",
    "episode",
    "split",
    "target_color",
    "target_shape",
    "first_word1",
    "first_word2",
    "all_words",
    "picked_color",
    "picked_shape",
    "success",
]


def intact_records(eval_jsonl: Path) -> list[dict]:
    """Read intact held-out and unseen episodes in evaluation file order."""
    with open(eval_jsonl) as fh:
        records = [json.loads(line) for line in fh]
    return [r for r in records if r["intervention"] == "intact" and r["split"] in ("held_out", "unseen")]


def word_tables(records: list[dict], n_values: int, vocab_size: int) -> dict[str, list[list[float]]]:
    """Estimate first-message symbol probabilities conditioned on each target property."""
    counts = np.zeros((2, 2, n_values, vocab_size))  # word position, property, value, symbol
    for r in records:
        for w, symbol in enumerate(r["scout_msgs"][0]):
            for p, value in enumerate(r["target_attrs"]):
                counts[w, p, value, symbol] += 1
    probs = counts / np.maximum(counts.sum(axis=-1, keepdims=True), 1)
    return {
        f"word{w + 1}_given_{name}": probs[w, p].round(4).tolist()
        for w in range(2)
        for p, name in enumerate(PROPERTIES)
    }


def message_log(records: list[dict], seed: int) -> list[dict]:
    """Make one human-readable row per intact evaluation episode."""
    rows = []
    for r in records:
        picked = r["picked_attrs"]
        rows.append(
            {
                "seed": seed,
                "episode": r["episode"],
                "split": r["split"],
                "target_color": COLOR_NAMES[r["target_attrs"][0]],
                "target_shape": SHAPE_NAMES[r["target_attrs"][1]],
                "first_word1": r["scout_msgs"][0][0],
                "first_word2": r["scout_msgs"][0][1],
                "all_words": " ".join(f"{a}-{b}" for a, b in r["scout_msgs"]),
                "picked_color": COLOR_NAMES[picked[0]] if picked else "",
                "picked_shape": SHAPE_NAMES[picked[1]] if picked else "",
                "success": r["success"],
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", default="runs_v2")
    parser.add_argument("--results", default="results/v2")
    parser.add_argument("--config", default="configs/v2.toml")
    args = parser.parse_args()
    cfg, _ = load_config(args.config)
    runs, results = Path(args.runs), Path(args.results)
    seed = representative_seed(load_rows(results / "per_run.csv"))
    lexicon: dict = {
        "representative_seed": seed,
        "source": "scout's first message, intact held_out and unseen evaluation episodes",
        "seeds": {},
    }
    log: list[dict] = []
    for path in sorted(runs.glob("comm/seed_*/eval.jsonl")):
        s = int(path.parent.name.removeprefix("seed_"))
        records = intact_records(path)
        lexicon["seeds"][str(s)] = word_tables(records, cfg.n_values, cfg.vocab_size)
        if s == seed:
            log = message_log(records, s)
    (results / "lexicon.json").write_text(json.dumps(lexicon, indent=1))
    with open(results / "message_log.csv", "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=LOG_FIELDS)
        writer.writeheader()
        writer.writerows(log)
    print(f"lexicon for {len(lexicon['seeds'])} seeds; message log: seed {seed}, {len(log)} episodes")


if __name__ == "__main__":
    main()
