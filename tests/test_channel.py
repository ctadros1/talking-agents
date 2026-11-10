import numpy as np
import pytest

from emcomm.channel import (
    CHANNEL_MODES,
    PERMUTED_WORDS,
    TRAIN_CHANNEL,
    cue_mask,
    episode_permutations,
    initial_received,
    route,
)
from emcomm.config import CONDITIONS

V = 8
SENT = np.array([[3, 5], [0, 7]])


def test_every_condition_has_a_channel_and_cue():
    assert set(TRAIN_CHANNEL) == set(CONDITIONS)
    assert cue_mask("comm").tolist() == [True, False]
    assert cue_mask("oracle").tolist() == [True, True]
    with pytest.raises(ValueError):
        cue_mask("telepathy")


def test_intact_swaps_speakers():
    assert route(SENT, "intact", V, np.random.default_rng(0)).tolist() == [[5, 3], [7, 0]]


def test_silenced_and_initial_are_silence():
    assert (route(SENT, "silenced", V, np.random.default_rng(0)) == V).all()
    assert (initial_received(4, V) == V).all()


def test_random_is_in_vocab_and_seeded():
    a = route(np.zeros((500, 2), dtype=np.int64), "random", V, np.random.default_rng(1))
    b = route(np.zeros((500, 2), dtype=np.int64), "random", V, np.random.default_rng(1))
    assert np.array_equal(a, b) and a.min() >= 0 and a.max() < V and len(np.unique(a)) == V


def test_permuted_applies_per_episode_permutation():
    perm = episode_permutations(2, V, seed=0)
    out = route(SENT, "permuted", V, np.random.default_rng(0), perm)
    assert out.tolist() == [[perm[0, 5], perm[0, 3]], [perm[1, 7], perm[1, 0]]]
    assert all(sorted(p) == list(range(V)) for p in perm)


SENT2 = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 0]]])  # (episodes, agents, words)


def test_two_word_routing():
    rng = np.random.default_rng(0)
    assert route(SENT2, "intact", V, rng).tolist() == [[[3, 4], [1, 2]], [[7, 0], [5, 6]]]
    assert (route(SENT2, "silenced", V, rng) == V).all()
    assert initial_received(3, V, n_words=2).shape == (3, 2, 2) and (initial_received(3, V, n_words=2) == V).all()
    rnd = route(np.zeros((400, 2, 2), dtype=np.int64), "random", V, np.random.default_rng(1))
    assert rnd.shape == (400, 2, 2) and rnd.max() < V and len(np.unique(rnd[:, :, 1])) == V


def test_per_word_permutations():
    assert {"permuted_w1", "permuted_w2"} <= set(CHANNEL_MODES) and PERMUTED_WORDS["permuted_w2"] == (1,)
    full = episode_permutations(2, V, seed=0, n_words=2)
    w1 = episode_permutations(2, V, seed=0, n_words=2, words=(0,))
    assert full.shape == (2, 2, V) and all(sorted(p) == list(range(V)) for p in full.reshape(-1, V))
    assert np.array_equal(w1[:, 0], full[:, 0]) and (w1[:, 1] == np.arange(V)).all()
    out = route(SENT2, "permuted_w1", V, np.random.default_rng(0), w1)
    assert out[0, 0].tolist() == [w1[0, 0, 3], 4] and out[1, 1].tolist() == [w1[1, 0, 5], 6]
    both = route(SENT2, "permuted", V, np.random.default_rng(0), full)
    assert both[0, 0].tolist() == [full[0, 0, 3], full[0, 1, 4]]
    assert episode_permutations(2, V, seed=0).shape == (2, V)  # the one-word (v1) layout is unchanged
