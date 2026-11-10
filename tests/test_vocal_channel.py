from dataclasses import replace

import numpy as np
import pytest

from emcomm.channel import initial_received, route
from emcomm.config import EnvConfig
from emcomm.env import GridEnv
from emcomm.layouts import RUNNER, SCOUT, make_episode_set

SENT = np.array([[0.5, -0.2], [-0.9, 0.3], [0.1, 0.8]])


def test_vocal_intact_is_swapped_noisy_clipped_and_seeded():
    a = route(SENT, "intact", 8, np.random.default_rng(0), noise=0.1)
    b = route(SENT, "intact", 8, np.random.default_rng(0), noise=0.1)
    assert np.array_equal(a, b) and a.min() >= -1 and a.max() <= 1
    assert np.allclose(route(SENT, "intact", 8, np.random.default_rng(0), noise=0.0), SENT[:, ::-1])
    assert np.abs(a - SENT[:, ::-1]).max() < 0.5 and not np.allclose(a, SENT[:, ::-1])


def test_vocal_silence_random_and_permuted():
    assert np.isnan(route(SENT, "silenced", 8, np.random.default_rng(0))).all()
    assert np.isnan(initial_received(4, 8, vocal=True)).all() and initial_received(4, 8).dtype == np.int64
    rand = route(np.zeros((500, 2)), "random", 8, np.random.default_rng(1))
    assert rand.min() >= -1 and rand.max() <= 1 and rand.std() > 0.5
    with pytest.raises(ValueError, match="vocal"):
        route(SENT, "permuted", 8, np.random.default_rng(0), perm=np.zeros((3, 8), dtype=np.int64))


def test_shuffled_hears_another_episode():
    heard = route(SENT, "shuffled", 8, np.random.default_rng(3), noise=0.0)
    rows = {tuple(r) for r in SENT[:, ::-1]}
    assert {tuple(r) for r in heard} == rows  # same speakers overall ...
    assert not any(np.array_equal(h, s) for h, s in zip(heard, SENT[:, ::-1]))  # ... but never your own
    tokens = np.array([[1, 2], [3, 4], [5, 6]])
    assert sorted(map(tuple, route(tokens, "shuffled", 8, np.random.default_rng(0)))) == [(2, 1), (4, 3), (6, 5)]
    with pytest.raises(ValueError):
        route(SENT[:1], "shuffled", 8, np.random.default_rng(0))


def test_vocal_observation_block(cfg, splits):
    vcfg = replace(cfg, channel="vocal")
    env = GridEnv(vcfg, 2)
    for i, spec in enumerate(make_episode_set(splits[1], 2, seed=0, cfg=vcfg)):
        env.load(i, spec)
    received = np.array([[np.nan, 0.25], [-0.5, np.nan]])
    obs = env.observe(received, np.array([True, False]))
    assert obs.shape == (2, 2, cfg.obs_dim)
    block = obs[..., cfg.obs_dim - 1 - (cfg.vocab_size + 1) : cfg.obs_dim - 1]
    silent_flag, tuning = block[..., -1], block[..., :-1]
    assert silent_flag.tolist() == [[1.0, 0.0], [0.0, 1.0]]
    assert not tuning[0, SCOUT].any() and not tuning[1, RUNNER].any()  # silence: no channel responds
    centres = np.linspace(-1, 1, cfg.vocab_size)
    assert np.argmax(tuning[0, RUNNER]) == np.argmin(np.abs(centres - 0.25))  # the ear's best channel
    assert np.argmax(tuning[1, SCOUT]) == np.argmin(np.abs(centres + 0.5))
    assert tuning.max() <= 1.0 and (tuning[0, RUNNER] > 0.05).sum() >= 2  # overlapping tuning curves


def test_defaults_keep_the_discrete_channel():
    assert EnvConfig().channel == "discrete" and EnvConfig().vocal_noise == 0.1
