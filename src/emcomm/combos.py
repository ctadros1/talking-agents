"""Color-shape combinations for the compositional task (v2 spec sections 2.1-2.4).

An item is a (color, shape) pair of indices in range(n_values). Every episode holds the four items of
a 2x2 square {c1, c2} x {s1, s2}. A fixed permutation pi of the shapes (seeded by `combo_seed`) holds
out the n_values pairs (c, pi[c]): exactly one per color and one per shape. Training squares contain
no held-out pair; with n_values = 5 there are 30 of them and every seen pair lies in exactly 6.
"""

from __future__ import annotations

from functools import cache
from itertools import combinations

import numpy as np

from emcomm.config import EnvConfig

COLOR_NAMES = ("red", "yellow", "green", "blue", "purple")
SHAPE_NAMES = ("cube", "sphere", "cone", "cylinder", "ring")
PROPERTIES = ("color", "shape")  # attribute order in item_attrs; also word order in the scripted codebook

Combo = tuple[int, int]
Square = tuple[tuple[int, int], tuple[int, int]]  # (two colors, two shapes), each ascending


@cache
def held_out_combos(cfg: EnvConfig) -> tuple[Combo, ...]:
    perm = np.random.default_rng(cfg.combo_seed).permutation(cfg.n_values)
    return tuple((c, int(perm[c])) for c in range(cfg.n_values))


def square_cells(square: Square) -> list[Combo]:
    colors, shapes = square
    return [(c, s) for c in colors for s in shapes]


@cache
def all_squares(n_values: int) -> tuple[Square, ...]:
    pairs = list(combinations(range(n_values), 2))
    return tuple((colors, shapes) for colors in pairs for shapes in pairs)


@cache
def training_squares(cfg: EnvConfig) -> tuple[Square, ...]:
    held = set(held_out_combos(cfg))
    return tuple(sq for sq in all_squares(cfg.n_values) if not held & set(square_cells(sq)))


@cache
def squares_containing(cfg: EnvConfig, combo: Combo) -> tuple[Square, ...]:
    return tuple(sq for sq in all_squares(cfg.n_values) if combo in square_cells(sq))


def sample_items(cfg: EnvConfig, rng: np.random.Generator, pool: str) -> tuple[np.ndarray, int]:
    """Attributes (4, 2) of an episode's items and the target's index, drawn after the geometry.

    pool "seen": a uniform training square, its cells in random item order, then a uniform target.
    pool "unseen": a uniform held-out pair as the target, a uniform square containing it, random order.
    """
    if pool == "seen":
        squares = training_squares(cfg)
        cells = square_cells(squares[int(rng.integers(len(squares)))])
        attrs = np.array(cells, dtype=np.int64)[rng.permutation(4)]
        return attrs, int(rng.integers(4))
    if pool == "unseen":
        held = held_out_combos(cfg)
        combo = held[int(rng.integers(len(held)))]
        squares = squares_containing(cfg, combo)
        cells = square_cells(squares[int(rng.integers(len(squares)))])
        attrs = np.array(cells, dtype=np.int64)[rng.permutation(4)]
        return attrs, int(np.flatnonzero((attrs == combo).all(axis=1))[0])
    raise ValueError(f"unknown item pool {pool!r}")


def combo_index(attrs: np.ndarray, n_values: int) -> np.ndarray:
    """(..., 2) color-shape pairs -> (...) indices color * n_values + shape."""
    attrs = np.asarray(attrs)
    return attrs[..., 0] * n_values + attrs[..., 1]
