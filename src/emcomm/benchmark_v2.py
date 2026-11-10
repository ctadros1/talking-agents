"""The v2 evaluation protocol for the compositional task.

The v1 protocol remains in ``emcomm.benchmark``. This module adds the unseen
split, word interventions, property metrics, and scripted references.
"""

from __future__ import annotations

import numpy as np

from emcomm.benchmark import EVAL_SEED, N_EVAL_EPISODES, RUN_FIELDS
from emcomm.channel import TRAIN_CHANNEL
from emcomm.combos import PROPERTIES, combo_index
from emcomm.config import EnvConfig
from emcomm.heuristics import ScriptedWordsTeam
from emcomm.layouts import EpisodeSpec, make_episode_set, make_layout_splits
from emcomm.rollout import run_episodes
from emcomm.stats import mutual_information_bits, positional_disentanglement, topographic_similarity

SPLITS = ("held_out", "unseen", "train")
WORD_INTERVENTIONS = ("permuted", "permuted_w1", "permuted_w2")
ROW_FIELDS = (
    "color_acc",
    "shape_acc",
    "twin_color",
    "twin_shape",
    "twin_opposite",
    "mi_w1_color",
    "mi_w1_shape",
    "mi_w2_color",
    "mi_w2_shape",
)
RUN_LEVEL_FIELDS = ("topsim", "posdis")
RUN_FIELDS_V2 = [*RUN_FIELDS, *ROW_FIELDS, *RUN_LEVEL_FIELDS]
SCRIPTED = {
    "comm": ("comm", PROPERTIES),
    "oracle": ("oracle", PROPERTIES),
    "no_comm": ("no_comm", PROPERTIES),
    "random_msg": ("random_msg", PROPERTIES),
    "color_only": ("comm", ("color",)),
    "shape_only": ("comm", ("shape",)),
}


def eval_plan(condition: str) -> list[tuple[str, str]]:
    """Split and channel mode pairs evaluated for a condition."""
    trained = TRAIN_CHANNEL[condition]
    if condition == "comm":
        return [
            *(("held_out", mode) for mode in ("intact", *WORD_INTERVENTIONS, "silenced")),
            *(("unseen", mode) for mode in ("intact", *WORD_INTERVENTIONS)),
            ("train", "intact"),
        ]
    return [(split, trained) for split in SPLITS]


def episode_sets(cfg: EnvConfig) -> dict[str, list[EpisodeSpec]]:
    train_layouts, held_out_layouts = make_layout_splits(cfg)
    return {
        "held_out": make_episode_set(held_out_layouts, N_EVAL_EPISODES, EVAL_SEED, cfg),
        "unseen": make_episode_set(held_out_layouts, N_EVAL_EPISODES, EVAL_SEED + 2, cfg, pool="unseen"),
        "train": make_episode_set(train_layouts, N_EVAL_EPISODES, EVAL_SEED + 1, cfg),
    }


def _rate(hits: np.ndarray, among: np.ndarray) -> float:
    return float(hits[among].mean()) if among.any() else float("nan")


def _codes(rows: np.ndarray) -> np.ndarray:
    """Injectively label whole messages for plug-in mutual information."""
    return np.unique(rows, axis=0, return_inverse=True)[1].reshape(-1)


def summarize(records: list[dict], n_values: int) -> dict:
    """Per-split, per-intervention metrics over a run's episodes."""
    success = np.array([r["success"] for r in records])
    target = np.array([r["target_attrs"] for r in records], dtype=np.int64)
    picked = np.array([r["picked_attrs"] or [-1, -1] for r in records], dtype=np.int64)
    first = np.array([r["scout_msgs"][0] for r in records], dtype=np.int64)
    has_pick = picked[:, 0] >= 0
    same = picked == target
    wrong = has_pick & ~same.all(axis=1)
    row = {
        "n": len(records),
        "success": float(success.mean()),
        "mean_return": float(np.mean([r["return"] for r in records])),
        "steps_success": float(np.mean([r["steps"] for r in records if r["success"]]))
        if success.any()
        else float("nan"),
        "path_efficiency": float(np.nanmean([r["path_efficiency"] for r in records]))
        if success.any()
        else float("nan"),
        "mi_first_msg": mutual_information_bits(combo_index(target, n_values), _codes(first)),
        "runner_pick_rate": float(has_pick.mean()),
        "color_acc": _rate(same[:, 0], has_pick),
        "shape_acc": _rate(same[:, 1], has_pick),
        "twin_color": _rate(same[:, 0] & ~same[:, 1], wrong),
        "twin_shape": _rate(~same[:, 0] & same[:, 1], wrong),
        "twin_opposite": _rate(~same[:, 0] & ~same[:, 1], wrong),
    }
    for w in range(2):
        for p, name in enumerate(PROPERTIES):
            row[f"mi_w{w + 1}_{name}"] = mutual_information_bits(first[:, w], target[:, p])
    return row


def run_level(records: list[dict]) -> dict:
    """Measure first messages pooled over intact held-out and unseen episodes."""
    meanings = np.array([r["target_attrs"] for r in records], dtype=np.int64)
    messages = np.array([r["scout_msgs"][0] for r in records], dtype=np.int64)
    return {
        "topsim": topographic_similarity(meanings, messages),
        "posdis": positional_disentanglement(meanings, messages),
    }


def evaluate_scripted(cfg: EnvConfig) -> list[dict]:
    """Privileged-navigation reference rows on every split."""
    rows, sets = [], episode_sets(cfg)
    for name, (condition, states) in SCRIPTED.items():
        mode = TRAIN_CHANNEL[condition]
        for split in SPLITS:
            team = ScriptedWordsTeam(N_EVAL_EPISODES, runner_sees_cue=condition == "oracle", states=states)
            records, _ = run_episodes(cfg, sets[split], team, condition, mode, seed=EVAL_SEED)
            rows.append(
                {
                    "condition": f"scripted_{name}",
                    "seed": -1,
                    "split": split,
                    "intervention": mode,
                    **summarize(records, cfg.n_values),
                }
            )
    return rows
