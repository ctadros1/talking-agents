from collections import Counter

import numpy as np
import pytest

from emcomm.combos import held_out_combos
from emcomm.config import EnvConfig
from emcomm.layouts import (
    RUNNER,
    SCOUT,
    UNREACHABLE,
    EpisodeSampler,
    bfs_distances,
    east_cells,
    make_episode_set,
    make_layout_splits,
)

V2 = EnvConfig(task="compositional", msg_words=2)


def test_splits_are_deterministic_and_disjoint(cfg, splits):
    train, held_out = splits
    assert len(train) == cfg.n_train_layouts and len(held_out) == cfg.n_eval_layouts
    again_train, again_held = make_layout_splits(cfg)
    assert all(np.array_equal(a.walls, b.walls) for a, b in zip(train + held_out, again_train + again_held))
    train_keys = {layout.walls.tobytes() for layout in train}
    assert not train_keys & {layout.walls.tobytes() for layout in held_out}


def test_partition_wall_is_solid_and_warehouse_connected(cfg, splits):
    for layout in splits[0] + splits[1]:
        assert layout.walls[:, cfg.wall_col].all()
        free = east_cells(layout.walls, cfg.wall_col)
        dist = bfs_distances(layout.walls, tuple(free[0]))
        assert all(dist[tuple(p)] < UNREACHABLE for p in free)
        assert layout.walls.sum() == cfg.grid_size + cfg.n_obstacles


def test_bfs_simple_corridor():
    walls = np.zeros((3, 3), dtype=bool)
    walls[1, :2] = True
    dist = bfs_distances(walls, (0, 0))
    assert dist[2, 0] == 6 and dist[1, 0] == UNREACHABLE


def test_episode_invariants(cfg, splits):
    specs = make_episode_set(splits[1], 200, seed=1, cfg=cfg)
    budget = cfg.max_steps - cfg.runner_slack
    for spec in specs:
        walls = spec.layout.walls
        assert spec.agent_pos[SCOUT, 1] < cfg.wall_col < spec.agent_pos[RUNNER, 1]
        assert (spec.item_pos[:, 1] > cfg.wall_col).all() and spec.depot[1] > cfg.wall_col
        cells = [tuple(p) for p in spec.item_pos] + [spec.depot, tuple(spec.agent_pos[RUNNER])]
        assert len(set(cells)) == len(cells)
        assert not any(walls[c] for c in cells)
        d_runner, d_depot = bfs_distances(walls, tuple(spec.agent_pos[RUNNER])), bfs_distances(walls, spec.depot)
        for k, p in enumerate(spec.item_pos):
            opt = d_runner[tuple(p)] + d_depot[tuple(p)] + 2
            assert opt <= budget  # EVERY item deliverable in time: geometry does not reveal the target
            if k == spec.target:
                assert opt == spec.runner_opt


def test_targets_roughly_uniform(cfg, splits):
    specs = make_episode_set(splits[0], 2000, seed=2, cfg=cfg)
    counts = np.bincount([s.target for s in specs], minlength=cfg.n_item_types)
    assert counts.min() > 400  # expected 500 each


def test_episode_sets_are_reproducible(cfg, splits):
    a = make_episode_set(splits[1], 50, seed=9, cfg=cfg)
    b = make_episode_set(splits[1], 50, seed=9, cfg=cfg)
    assert all(x.target == y.target and np.array_equal(x.item_pos, y.item_pos) for x, y in zip(a, b))
    s1, s2 = EpisodeSampler(splits[0], cfg, 3), EpisodeSampler(splits[0], cfg, 3)
    assert [s1.sample().target for _ in range(20)] == [s2.sample().target for _ in range(20)]


def test_v2_training_episodes_use_seen_squares(splits):
    held = set(held_out_combos(V2))
    sampler = EpisodeSampler(splits[0], V2, seed=0)
    targets = Counter()
    for _ in range(600):
        spec = sampler.sample()
        cells = {tuple(int(v) for v in a) for a in spec.item_attrs}
        assert len(cells) == 4 and not cells & held
        targets[tuple(int(v) for v in spec.item_attrs[spec.target])] += 1
    assert len(targets) == 20


def test_v2_unseen_episodes_target_held_out_pairs(splits):
    specs = make_episode_set(splits[1], 100, seed=3, cfg=V2, pool="unseen")
    assert {tuple(int(v) for v in s.item_attrs[s.target]) for s in specs} == set(held_out_combos(V2))


def test_v2_target_position_is_uniform(splits):
    specs = make_episode_set(splits[0], 2000, seed=2, cfg=V2)
    assert np.bincount([s.target for s in specs], minlength=4).min() > 400


def test_v2_every_item_is_deliverable(splits):
    budget = V2.max_steps - V2.runner_slack
    for spec in make_episode_set(splits[1], 100, seed=4, cfg=V2, pool="unseen"):
        walls = spec.layout.walls
        d_runner = bfs_distances(walls, tuple(spec.agent_pos[RUNNER]))
        d_depot = bfs_distances(walls, spec.depot)
        assert all(d_runner[tuple(p)] + d_depot[tuple(p)] + 2 <= budget for p in spec.item_pos)


def test_types_task_has_no_unseen_pool(cfg, splits):
    assert make_episode_set(splits[1], 3, seed=0, cfg=cfg)[0].item_attrs is None
    with pytest.raises(ValueError, match="unseen"):
        make_episode_set(splits[1], 1, seed=0, cfg=cfg, pool="unseen")
