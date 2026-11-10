"""Map layouts, shortest paths, and split-information episode sampling.

Coordinates are (row, col). A solid partition wall at column `wall_col`
separates the control room (west, the scout) from the warehouse (east, the
runner, every item, and the depot). The scout can never reach an item, so the
message channel is the only path by which target information can reach the
runner.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np

from emcomm.combos import sample_items
from emcomm.config import EnvConfig

SCOUT, RUNNER = 0, 1
UNREACHABLE = 10_000


@dataclass(frozen=True)
class Layout:
    layout_id: int
    walls: np.ndarray  # (G, G) bool, True = blocked


@dataclass(frozen=True)
class EpisodeSpec:
    layout: Layout
    target: int
    item_pos: np.ndarray  # (n_item_types, 2) int64; row k = position of item type k
    depot: tuple[int, int]
    agent_pos: np.ndarray  # (2, 2) int64; row SCOUT then row RUNNER
    runner_opt: int  # fewest steps for a fully informed runner to deliver the target
    item_attrs: np.ndarray | None = None  # (n_item_types, 2) color, shape of each item (compositional task)


def bfs_distances(walls: np.ndarray, start: tuple[int, int]) -> np.ndarray:
    g = walls.shape[0]
    dist = np.full(walls.shape, UNREACHABLE, dtype=np.int64)
    dist[start] = 0
    queue = deque([start])
    while queue:
        r, c = queue.popleft()
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < g and 0 <= nc < g and not walls[nr, nc] and dist[nr, nc] == UNREACHABLE:
                dist[nr, nc] = dist[r, c] + 1
                queue.append((nr, nc))
    return dist


def east_cells(walls: np.ndarray, wall_col: int) -> np.ndarray:
    g = walls.shape[0]
    return np.array([(r, c) for r in range(g) for c in range(wall_col + 1, g) if not walls[r, c]], dtype=np.int64)


def west_cells(walls: np.ndarray, wall_col: int) -> np.ndarray:
    g = walls.shape[0]
    return np.array([(r, c) for r in range(g) for c in range(wall_col) if not walls[r, c]], dtype=np.int64)


def _generate_layout(layout_id: int, rng: np.random.Generator, cfg: EnvConfig) -> Layout:
    g, wc = cfg.grid_size, cfg.wall_col
    while True:
        walls = np.zeros((g, g), dtype=bool)
        walls[:, wc] = True
        east = [(r, c) for r in range(g) for c in range(wc + 1, g)]
        for idx in rng.choice(len(east), size=cfg.n_obstacles, replace=False):
            walls[east[idx]] = True
        free = east_cells(walls, wc)
        dist = bfs_distances(walls, tuple(int(v) for v in free[0]))
        if all(dist[r, c] < UNREACHABLE for r, c in free):
            return Layout(layout_id=layout_id, walls=walls)


def make_layout_splits(cfg: EnvConfig) -> tuple[list[Layout], list[Layout]]:
    """Deterministic, disjoint train and held-out layout sets."""
    rng = np.random.default_rng(cfg.layout_seed)
    layouts: list[Layout] = []
    seen: set[bytes] = set()
    while len(layouts) < cfg.n_train_layouts + cfg.n_eval_layouts:
        layout = _generate_layout(len(layouts), rng, cfg)
        key = layout.walls.tobytes()
        if key not in seen:
            seen.add(key)
            layouts.append(layout)
    return layouts[: cfg.n_train_layouts], layouts[cfg.n_train_layouts :]


def try_geometry(layout: Layout, rng: np.random.Generator, cfg: EnvConfig, pool: str = "seen") -> EpisodeSpec | None:
    """One rejection-sampling attempt; None if some item cannot be delivered in time.

    The constraint covers EVERY item and the target (and, in the compositional task, the items'
    colors and shapes) is drawn only after the geometry is accepted, so the target is independent of
    the geometry: a runner without target information can do no better than 1 / n_item_types.
    `pool` chooses the compositional task's item set: "seen" (training squares) or "unseen".
    """
    if cfg.task == "types" and pool != "seen":
        raise ValueError("only the compositional task has an unseen item pool")
    n_items = cfg.n_item_types
    east, west = east_cells(layout.walls, cfg.wall_col), west_cells(layout.walls, cfg.wall_col)
    picks = east[rng.choice(len(east), size=n_items + 2, replace=False)]
    items = [tuple(int(v) for v in p) for p in picks[:n_items]]
    depot = tuple(int(v) for v in picks[n_items])
    runner = tuple(int(v) for v in picks[n_items + 1])
    d_runner, d_depot = bfs_distances(layout.walls, runner), bfs_distances(layout.walls, depot)
    runner_opt = [int(d_runner[p] + d_depot[p] + 2) for p in items]  # +2: pick-up and delivery actions
    if max(runner_opt) > cfg.max_steps - cfg.runner_slack:
        return None
    scout = tuple(int(v) for v in west[rng.integers(len(west))])
    if cfg.task == "types":
        target, item_attrs = int(rng.integers(n_items)), None
    else:
        item_attrs, target = sample_items(cfg, rng, pool)
    return EpisodeSpec(
        layout=layout,
        target=target,
        item_pos=picks[:n_items].copy(),
        depot=depot,
        agent_pos=np.array([scout, runner], dtype=np.int64),
        runner_opt=runner_opt[target],
        item_attrs=item_attrs,
    )


def sample_episode(
    layout: Layout, rng: np.random.Generator, cfg: EnvConfig, max_tries: int = 1000, pool: str = "seen"
) -> EpisodeSpec:
    for _ in range(max_tries):
        spec = try_geometry(layout, rng, cfg, pool)
        if spec is not None:
            return spec
    raise RuntimeError(f"layout {layout.layout_id}: no valid episode in {max_tries} tries")


def make_episode_set(layouts: list[Layout], n: int, seed: int, cfg: EnvConfig, pool: str = "seen") -> list[EpisodeSpec]:
    """Fixed evaluation episodes: identical for every run that uses the same seed."""
    rng = np.random.default_rng(seed)
    return [sample_episode(layouts[int(rng.integers(len(layouts)))], rng, cfg, pool=pool) for _ in range(n)]


class EpisodeSampler:
    """Infinite, seeded stream of training episodes over a layout set."""

    def __init__(self, layouts: list[Layout], cfg: EnvConfig, seed: int):
        self.layouts, self.cfg = layouts, cfg
        self.rng = np.random.default_rng(seed)

    def sample(self) -> EpisodeSpec:
        return sample_episode(self.layouts[int(self.rng.integers(len(self.layouts)))], self.rng, self.cfg)
