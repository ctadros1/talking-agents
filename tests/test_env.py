import dataclasses

import numpy as np
import pytest

from emcomm.channel import initial_received
from emcomm.config import EnvConfig
from emcomm.env import DOWN, INTERACT, LEFT, RIGHT, STAY, UP, GridEnv
from emcomm.layouts import RUNNER, SCOUT, EpisodeSpec, Layout


@pytest.fixture
def open_spec(cfg):
    walls = np.zeros((cfg.grid_size, cfg.grid_size), dtype=bool)
    walls[:, cfg.wall_col] = True
    items = np.array([[0, 6], [0, 8], [0, 10], [11, 11]], dtype=np.int64)
    return EpisodeSpec(
        layout=Layout(layout_id=-1, walls=walls),
        target=1,
        item_pos=items,
        depot=(4, 8),
        agent_pos=np.array([[5, 1], [1, 8]], dtype=np.int64),
        runner_opt=7,
    )


def _env(cfg, spec):
    env = GridEnv(cfg, 1)
    env.load(0, spec)
    return env


def _act(env, runner_move, scout_move=STAY):
    return env.step(np.array([[scout_move, runner_move]]))


def test_obs_shape_and_cue(cfg, open_spec):
    env = _env(cfg, open_spec)
    received = initial_received(1, cfg.vocab_size)
    obs = env.observe(received, np.array([True, False]))
    assert obs.shape == (1, 2, cfg.obs_dim) and obs.dtype == np.float32
    cue = slice(cfg.obs_dim - 1 - (cfg.vocab_size + 1) - cfg.n_item_types, cfg.obs_dim - 1 - (cfg.vocab_size + 1))
    assert obs[0, SCOUT, cue].tolist() == [0, 1, 0, 0]
    assert obs[0, RUNNER, cue].tolist() == [0, 0, 0, 0]
    recv = slice(cfg.obs_dim - 1 - (cfg.vocab_size + 1), cfg.obs_dim - 1)
    assert obs[0, RUNNER, recv].argmax() == cfg.silence


def test_view_is_egocentric_and_pads_with_walls(cfg, open_spec):
    env = _env(cfg, open_spec)
    views = env._views().reshape(1, 2, cfg.n_view_channels, cfg.view_size, cfg.view_size)
    runner = views[0, RUNNER]
    centre = cfg.view_radius
    assert runner[0, 0].all()  # runner at row 1: view rows -2..-1 are out of bounds -> wall
    assert runner[2, centre - 1, centre] == 1.0  # item type 1 is directly above the runner
    assert runner[1 + cfg.n_item_types, centre + 3, centre] == 1.0  # depot three cells below
    scout = views[0, SCOUT]
    assert scout[0, :, centre + 3].all()  # partition wall three columns east of the scout


def test_walls_block_movement(cfg, open_spec):
    env = _env(cfg, open_spec)
    env.pos[0, RUNNER] = (3, cfg.wall_col + 1)
    _act(env, LEFT)
    assert env.pos[0, RUNNER].tolist() == [3, cfg.wall_col + 1]
    env.pos[0, RUNNER] = (0, 7)
    _act(env, UP)
    assert env.pos[0, RUNNER].tolist() == [0, 7]


def test_correct_delivery_succeeds(cfg, open_spec):
    env = _env(cfg, open_spec)
    r = _act(env, UP)
    assert r.reward[0] == pytest.approx(-cfg.step_penalty)
    r = _act(env, INTERACT)
    assert env.carrying[0, RUNNER] == 1 and not env.item_present[0, 1]
    assert r.picked[0, RUNNER] == 1 and r.reward[0] == pytest.approx(cfg.pickup_bonus - cfg.step_penalty)
    for _ in range(4):
        _act(env, DOWN)
    r = _act(env, INTERACT)
    assert r.done[0] and r.success[0] and r.reward[0] == pytest.approx(1.0 - cfg.step_penalty)


def test_wrong_delivery_fails_and_ends(cfg, open_spec):
    env = _env(cfg, open_spec)
    env.pos[0, RUNNER] = (0, 6)
    _act(env, INTERACT)  # picks item type 0, target is 1
    env.pos[0, RUNNER] = open_spec.depot
    r = _act(env, INTERACT)
    assert r.done[0] and not r.success[0] and r.reward[0] == pytest.approx(-cfg.step_penalty)


def test_cannot_carry_two_items(cfg, open_spec):
    env = _env(cfg, open_spec)
    env.pos[0, RUNNER] = (0, 6)
    _act(env, INTERACT)
    _act(env, RIGHT)
    _act(env, RIGHT)
    r = _act(env, INTERACT)  # on item 1 while carrying item 0
    assert env.carrying[0, RUNNER] == 0 and env.item_present[0, 1] and r.picked[0, RUNNER] == -1


def test_timeout(cfg, open_spec):
    env = _env(cfg, open_spec)
    for t in range(cfg.max_steps):
        r = _act(env, STAY)
        assert r.done[0] == (t == cfg.max_steps - 1)
    assert not r.success[0]


def test_replaying_actions_is_deterministic(cfg, splits):
    from emcomm.layouts import make_episode_set

    specs = make_episode_set(splits[1], 16, seed=4, cfg=cfg)
    rng = np.random.default_rng(0)
    actions = rng.integers(6, size=(cfg.max_steps, 16, 2))

    def play():
        env = GridEnv(cfg, 16)
        for i, s in enumerate(specs):
            env.load(i, s)
        out = []
        for t in range(cfg.max_steps):
            r = env.step(actions[t])
            out.append((env.pos.copy(), env.carrying.copy(), r.reward.copy(), r.done.copy()))
        return out

    for (p1, c1, r1, d1), (p2, c2, r2, d2) in zip(play(), play()):
        assert np.array_equal(p1, p2) and np.array_equal(c1, c2) and np.array_equal(r1, r2) and np.array_equal(d1, d2)


V2 = EnvConfig(task="compositional", msg_words=2)


@pytest.fixture
def v2_env(open_spec):
    attrs = np.array([[0, 1], [0, 3], [2, 1], [2, 3]], dtype=np.int64)  # colors {0, 2} x shapes {1, 3}
    env = GridEnv(V2, 1)
    env.load(0, dataclasses.replace(open_spec, item_attrs=attrs))  # target 1 = (color 0, shape 3)
    return env


def test_v2_obs_blocks(v2_env):
    obs = v2_env.observe(initial_received(1, V2.vocab_size, n_words=2), np.array([True, False]))
    assert obs.shape == (1, 2, 630) and obs.dtype == np.float32
    assert obs[0, SCOUT, V2.cue_slice].tolist() == [1, 0, 0, 0, 0, 0, 0, 0, 1, 0]  # color 0, shape 3
    assert not obs[0, RUNNER, V2.cue_slice].any()
    heard = obs[0, RUNNER, -19:-1]
    assert heard[8] == 1 and heard[17] == 1 and heard.sum() == 2  # both words silent


def test_v2_view_shows_color_and_shape_channels(v2_env):
    views = v2_env._views().reshape(1, 2, V2.n_view_channels, V2.view_size, V2.view_size)
    centre = V2.view_radius
    above = views[0, RUNNER, :, centre - 1, centre]  # item 1 sits directly above the runner
    assert above[1 + 0] == 1 and above[1 + V2.n_values + 3] == 1 and above.sum() == 2
    assert views[0, RUNNER, 1 + V2.n_features, centre + 3, centre] == 1  # depot three cells below


def test_v2_scout_view_hides_items_and_depot_across_partition(v2_env):
    v2_env.pos[0, SCOUT] = (1, V2.wall_col - 1)
    v2_env.depot[0] = (3, 5)  # inside both crops, so hiding the depot is exercised directly
    views = v2_env._views().reshape(1, 2, V2.n_view_channels, V2.view_size, V2.view_size)
    assert views[0, SCOUT, 0].any()  # wall and out-of-bounds information remains visible
    assert views[0, SCOUT, 0, 0].all()  # top rows are out of bounds
    assert views[0, SCOUT, 0, V2.view_radius, V2.wall_col] == 1  # partition wall remains visible
    assert not views[0, SCOUT, 1:].any()
    assert views[0, RUNNER, 1 : 1 + V2.n_features].any()
    assert views[0, RUNNER, 1 + V2.n_features].any()


def test_v2_scout_view_hides_warehouse_obstacle_wall_changes(v2_env):
    v2_env.pos[0, SCOUT] = (3, V2.wall_col - 1)
    v2_env.pos[0, RUNNER] = (3, V2.wall_col + 3)
    obstacle = (3, V2.wall_col + 2)
    v2_env.walls[0, obstacle[0], obstacle[1]] = False
    before = v2_env._views().reshape(1, 2, V2.n_view_channels, V2.view_size, V2.view_size).copy()
    v2_env.walls[0, obstacle[0], obstacle[1]] = True
    after = v2_env._views().reshape(1, 2, V2.n_view_channels, V2.view_size, V2.view_size)
    assert np.array_equal(before[0, SCOUT], after[0, SCOUT])
    assert not np.array_equal(before[0, RUNNER, 0], after[0, RUNNER, 0])


def test_v2_carrying_block(v2_env):
    received, mask = initial_received(1, V2.vocab_size, n_words=2), np.array([True, False])
    start = V2.cue_slice.start - (V2.n_features + 1)
    assert v2_env.observe(received, mask)[0, RUNNER, start : start + 11].tolist() == [1] + [0] * 10
    _act(v2_env, UP)
    _act(v2_env, INTERACT)  # picks item 1: color 0, shape 3
    block = v2_env.observe(received, mask)[0, RUNNER, start : start + 11]
    assert block.tolist() == [0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0]


def test_v2_runner_hears_both_words(v2_env):
    received = np.array([[[8, 8], [3, 6]]])  # the runner (agent 1) hears words 3 and 6
    heard = v2_env.observe(received, np.array([True, False]))[0, RUNNER, -19:-1]
    assert heard[:9].argmax() == 3 and heard[9:].argmax() == 6 and heard.sum() == 2


def test_v2_delivery_is_judged_by_item(v2_env):
    _act(v2_env, UP)
    _act(v2_env, INTERACT)
    for _ in range(4):
        _act(v2_env, DOWN)
    r = _act(v2_env, INTERACT)
    assert r.done[0] and r.success[0]
