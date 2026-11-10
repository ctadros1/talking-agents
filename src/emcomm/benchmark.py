"""The fixed evaluation protocol (torch-free): episode sets, per-condition
evaluation plan, per-run summary metrics, and the scripted reference rows."""

from __future__ import annotations

import numpy as np

from emcomm.channel import TRAIN_CHANNEL
from emcomm.config import CONDITIONS, EnvConfig
from emcomm.heuristics import ScriptedTeam
from emcomm.layouts import EpisodeSpec, make_episode_set, make_layout_splits
from emcomm.rollout import run_episodes
from emcomm.stats import mutual_information_bits

N_EVAL_EPISODES = 200
EVAL_SEED = 424_242  # episode sets, action sampling, random channel and permutations
RUN_FIELDS = [
    "condition",
    "seed",
    "split",
    "intervention",
    "n",
    "success",
    "mean_return",
    "steps_success",
    "path_efficiency",
    "mi_first_msg",
    "runner_pick_rate",
]


def eval_plan(condition: str, channel: str = "discrete") -> list[tuple[str, str]]:
    """(split, channel mode) pairs evaluated for a condition. Interventions apply to learned messages only.
    On the vocal channel the causal test is "shuffled" (another episode's pitch) instead of "permuted"."""
    trained = TRAIN_CHANNEL[condition]
    if condition == "comm" and channel == "vocal":
        return [("held_out", "intact"), ("held_out", "shuffled"), ("held_out", "silenced"), ("train", "intact")]
    if condition == "comm":
        return [("held_out", "intact"), ("held_out", "permuted"), ("held_out", "silenced"), ("train", "intact")]
    return [("held_out", trained), ("train", trained)]


def episode_sets(cfg: EnvConfig) -> dict[str, list[EpisodeSpec]]:
    train_layouts, held_out_layouts = make_layout_splits(cfg)
    return {
        "held_out": make_episode_set(held_out_layouts, N_EVAL_EPISODES, EVAL_SEED, cfg),
        "train": make_episode_set(train_layouts, N_EVAL_EPISODES, EVAL_SEED + 1, cfg),
    }


def pitch_bins(pitch: np.ndarray, n_bins: int) -> np.ndarray:
    """Equal-width bins over [-1, 1] so a continuous pitch can enter the plug-in MI estimate."""
    return np.clip(((pitch + 1.0) / 2.0 * n_bins).astype(np.int64), 0, n_bins - 1)


def summarize(records: list[dict], n_pitch_bins: int = 8) -> dict:
    success = np.array([r["success"] for r in records])
    first_msgs = np.array([r["scout_msgs"][0] for r in records])
    if np.issubdtype(first_msgs.dtype, np.floating):  # vocal channel
        first_msgs = pitch_bins(first_msgs, n_pitch_bins)
    targets = np.array([r["target"] for r in records])
    return {
        "n": len(records),
        "success": float(success.mean()),
        "mean_return": float(np.mean([r["return"] for r in records])),
        "steps_success": float(np.mean([r["steps"] for r in records if r["success"]]))
        if success.any()
        else float("nan"),
        "path_efficiency": float(np.nanmean([r["path_efficiency"] for r in records]))
        if success.any()
        else float("nan"),
        "mi_first_msg": mutual_information_bits(targets, first_msgs),
        "runner_pick_rate": float(np.mean([r["runner_pick"] >= 0 for r in records])),
    }


def evaluate_scripted(cfg: EnvConfig) -> list[dict]:
    """Authored reference rows (privileged navigation), condition names prefixed 'scripted_'."""
    rows = []
    sets = episode_sets(cfg)
    for condition in CONDITIONS:
        mode = TRAIN_CHANNEL[condition]
        for split in ("held_out", "train"):
            team = ScriptedTeam(N_EVAL_EPISODES, runner_sees_cue=condition == "oracle")
            records, _ = run_episodes(cfg, sets[split], team, condition, mode, seed=EVAL_SEED)
            rows.append(
                {
                    "condition": f"scripted_{condition}",
                    "seed": -1,
                    "split": split,
                    "intervention": mode,
                    **summarize(records),
                }
            )
    return rows
