"""Pre-registered statistics: percentile bootstrap CIs over seeds, plug-in mutual information, and (v2) \
compositionality measures."""

from __future__ import annotations

from collections import Counter
from itertools import combinations

import numpy as np

N_BOOT = 10_000


def bootstrap_mean_ci(x, n_boot: int = N_BOOT, seed: int = 0, alpha: float = 0.05) -> tuple[float, float, float]:
    x = np.asarray(x, dtype=np.float64)
    rng = np.random.default_rng(seed)
    means = x[rng.integers(len(x), size=(n_boot, len(x)))].mean(axis=1)
    lo, hi = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return float(x.mean()), float(lo), float(hi)


def bootstrap_diff_ci(a, b, n_boot: int = N_BOOT, seed: int = 0, alpha: float = 0.05) -> tuple[float, float, float]:
    """mean(a) - mean(b) for independent groups (different training runs)."""
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    rng = np.random.default_rng(seed)
    da = a[rng.integers(len(a), size=(n_boot, len(a)))].mean(axis=1)
    db = b[rng.integers(len(b), size=(n_boot, len(b)))].mean(axis=1)
    lo, hi = np.quantile(da - db, [alpha / 2, 1 - alpha / 2])
    return float(a.mean() - b.mean()), float(lo), float(hi)


def paired_diff_ci(a, b, n_boot: int = N_BOOT, seed: int = 0, alpha: float = 0.05) -> tuple[float, float, float]:
    """mean(a - b) where a[i], b[i] come from the same run (e.g. intact vs permuted)."""
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError("paired samples must have equal length")
    return bootstrap_mean_ci(a - b, n_boot=n_boot, seed=seed, alpha=alpha)


def mutual_information_bits(x, y) -> float:
    """Plug-in I(X;Y) in bits for two equal-length arrays of non-negative ints."""
    x, y = np.asarray(x, dtype=np.int64), np.asarray(y, dtype=np.int64)
    joint = np.zeros((x.max() + 1, y.max() + 1))
    np.add.at(joint, (x, y), 1.0)
    joint /= joint.sum()
    px, py = joint.sum(axis=1, keepdims=True), joint.sum(axis=0, keepdims=True)
    nz = joint > 0
    return float((joint[nz] * np.log2(joint[nz] / (px @ py)[nz])).sum())


# ---- v2 (compositional task): frozen at tag prereg-v2 ----


def entropy_bits(x) -> float:
    """Plug-in entropy in bits of an array of non-negative ints."""
    p = np.bincount(np.asarray(x, dtype=np.int64)) / len(x)
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())


def _ranks(x: np.ndarray) -> np.ndarray:
    """Ranks 0..n-1 with ties given their average rank (as Spearman's rho requires)."""
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x))
    ranks[order] = np.arange(len(x), dtype=np.float64)
    _, inverse, counts = np.unique(x, return_inverse=True, return_counts=True)
    inverse = inverse.reshape(-1)
    return (np.bincount(inverse, weights=ranks) / counts)[inverse]


def spearman(x, y) -> float:
    """Spearman rank correlation; nan when either variable is constant."""
    rx, ry = _ranks(np.asarray(x, dtype=np.float64)), _ranks(np.asarray(y, dtype=np.float64))
    if rx.std() == 0 or ry.std() == 0:
        return float("nan")
    return float(np.corrcoef(rx, ry)[0, 1])


def topographic_similarity(meanings, messages) -> float:
    """Topographic similarity: Spearman correlation between pairwise meaning and message Hamming distances.

    Each distinct meaning is represented by its most frequent message (ties: the smallest).
    Returns nan if fewer than two meanings or a distance is constant.
    """
    groups: dict[tuple, Counter] = {}
    for meaning, message in zip(np.asarray(meanings).tolist(), np.asarray(messages).tolist()):
        groups.setdefault(tuple(meaning), Counter())[tuple(message)] += 1
    keys = sorted(groups)
    modal = [min(groups[k].items(), key=lambda kv: (-kv[1], kv[0]))[0] for k in keys]
    pairs = list(combinations(range(len(keys)), 2))
    if not pairs:
        return float("nan")
    d_meaning = [sum(a != b for a, b in zip(keys[i], keys[j])) for i, j in pairs]
    d_message = [sum(a != b for a, b in zip(modal[i], modal[j])) for i, j in pairs]
    return spearman(d_meaning, d_message)


def positional_disentanglement(meanings, messages) -> float:
    """Mean positional disentanglement over non-constant message positions; nan if all are constant."""
    meanings, messages = np.asarray(meanings), np.asarray(messages)
    scores = []
    for w in range(messages.shape[1]):
        h = entropy_bits(messages[:, w])
        if h <= 0:
            continue
        mi = sorted(
            (mutual_information_bits(messages[:, w], meanings[:, a]) for a in range(meanings.shape[1])), reverse=True
        )
        scores.append((mi[0] - mi[1]) / h)
    return float(np.mean(scores)) if scores else float("nan")
