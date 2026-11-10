"""Evaluate trained checkpoints on the fixed benchmark under the pre-registered interventions.

Usage: python -m emcomm.evaluate --config configs/base.toml --runs runs --out results --workers 6
       python -m emcomm.evaluate --config configs/v2.toml --runs runs_v2 --out results/v2 --workers 8
Writes runs/<condition>/seed_XXXX/eval.jsonl (per episode) and results/per_run.csv (per run x split x intervention).
"""

from __future__ import annotations

import argparse
import csv
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch

from emcomm import benchmark_v2
from emcomm.benchmark import (
    EVAL_SEED,
    N_EVAL_EPISODES,
    RUN_FIELDS,
    episode_sets,
    eval_plan,
    evaluate_scripted,
    summarize,
)
from emcomm.channel import TRAIN_CHANNEL
from emcomm.config import load_config
from emcomm.env import GridEnv
from emcomm.policy import AgentNet
from emcomm.rollout import run_episodes
from emcomm.train import load_checkpoint


class LearnedTeam:
    """Two trained agents acting stochastically with a private, seeded torch generator."""

    def __init__(self, nets: list[AgentNet], n_envs: int, seed: int):
        self.nets = nets
        self.h = [net.initial_state(n_envs) for net in nets]
        self.first = torch.ones(n_envs, dtype=torch.bool)
        self.gen = torch.Generator().manual_seed(seed)

    def reset(self, i: int) -> None:
        self.first[i] = True

    def act(self, env: GridEnv, obs: np.ndarray, received: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        obs_t = torch.from_numpy(obs)
        vocal = self.nets[0].vocal
        moves = np.zeros((env.n, 2), dtype=np.int64)
        shape = (env.n, 2) if env.cfg.msg_words == 1 else (env.n, 2, env.cfg.msg_words)
        sent = np.zeros(shape, dtype=np.float64 if vocal else np.int64)
        with torch.no_grad():
            for a, net in enumerate(self.nets):
                move_logits, msg_logits, _, self.h[a] = net.forward_sequence(
                    obs_t[None, :, a], self.h[a], self.first[None]
                )
                move, msg, _ = net.sample(move_logits[0], msg_logits[0], self.gen)
                moves[:, a], sent[:, a] = move.numpy(), np.clip(msg.numpy(), -1.0, 1.0) if vocal else msg.numpy()
        self.first[:] = False
        return moves, sent


def evaluate_checkpoint(path: str | Path, splits: tuple[str, ...] | None = None) -> tuple[list[dict], list[dict]]:
    """Return episode records and summary rows for one run.

    ``splits`` limits evaluation; pilot callers pass ``("train",)``. V2 rows
    carry run-level metrics when both held-out and unseen were evaluated.
    """
    torch.set_num_threads(1)
    nets, cfg, _, ckpt = load_checkpoint(path)
    condition, seed = ckpt["condition"], ckpt["seed"]
    v2 = cfg.task == "compositional"
    sets = benchmark_v2.episode_sets(cfg) if v2 else episode_sets(cfg)
    plan = benchmark_v2.eval_plan(condition) if v2 else eval_plan(condition, cfg.channel)
    episodes, rows, pooled, pooled_splits = [], [], [], set()
    for split, mode in plan:
        if splits is not None and split not in splits:
            continue
        team = LearnedTeam(nets, N_EVAL_EPISODES, EVAL_SEED)
        records, _ = run_episodes(cfg, sets[split], team, condition, mode, seed=EVAL_SEED)
        tag = {"condition": condition, "seed": seed, "split": split, "intervention": mode}
        episodes += [{**tag, **r} for r in records]
        rows.append({**tag, **(benchmark_v2.summarize(records, cfg.n_values) if v2 else summarize(records))})
        if v2 and split in ("held_out", "unseen") and mode == TRAIN_CHANNEL[condition]:
            pooled += records
            pooled_splits.add(split)
    if pooled_splits == {"held_out", "unseen"}:
        extra = benchmark_v2.run_level(pooled)
        rows = [{**row, **extra} for row in rows]
    return episodes, rows


def _evaluate_one(path: str) -> list[dict]:
    episodes, rows = evaluate_checkpoint(path)
    with open(Path(path).parent / "eval.jsonl", "w") as fh:
        fh.writelines(json.dumps(record) + "\n" for record in episodes)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--runs", default="runs")
    parser.add_argument("--out", default="results")
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()
    cfg, _ = load_config(args.config)
    paths = sorted(str(p) for p in Path(args.runs).glob("*/seed_*/final.pt"))
    v2 = cfg.task == "compositional"
    scripted = benchmark_v2.evaluate_scripted if v2 else evaluate_scripted
    rows = scripted(cfg) if cfg.channel == "discrete" else []  # scripted codebooks are discrete
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for run_rows in pool.map(_evaluate_one, paths):
            rows += run_rows
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "per_run.csv", "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=benchmark_v2.RUN_FIELDS_V2 if v2 else RUN_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"evaluated {len(paths)} runs -> {out / 'per_run.csv'}")


if __name__ == "__main__":
    main()
