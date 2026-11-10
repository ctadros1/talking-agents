import numpy as np
import pytest

from emcomm.stats import (
    bootstrap_diff_ci,
    bootstrap_mean_ci,
    entropy_bits,
    mutual_information_bits,
    paired_diff_ci,
    positional_disentanglement,
    spearman,
    topographic_similarity,
)


def test_bootstrap_mean_ci_brackets_mean_and_is_seeded():
    x = np.random.default_rng(0).normal(0.5, 0.1, size=20)
    mean, lo, hi = bootstrap_mean_ci(x)
    assert lo < mean < hi and mean == pytest.approx(x.mean())
    assert bootstrap_mean_ci(x) == bootstrap_mean_ci(x)
    assert bootstrap_mean_ci([0.3] * 20) == pytest.approx((0.3, 0.3, 0.3))


def test_diff_ci_detects_clear_gap_and_not_noise():
    rng = np.random.default_rng(1)
    a, b = rng.normal(0.8, 0.05, 20), rng.normal(0.25, 0.05, 20)
    mean, lo, hi = bootstrap_diff_ci(a, b)
    assert lo > 0.45 and mean == pytest.approx(a.mean() - b.mean())
    c = rng.normal(0.25, 0.05, 20)
    _, lo2, hi2 = bootstrap_diff_ci(b, c)
    assert lo2 < 0 < hi2


def test_paired_diff_uses_pairing():
    base = np.linspace(0.2, 0.9, 20)  # large between-run spread
    mean, lo, hi = paired_diff_ci(base + 0.05, base)
    assert mean == pytest.approx(0.05) and lo == pytest.approx(0.05) and hi == pytest.approx(0.05)
    with pytest.raises(ValueError):
        paired_diff_ci([1, 2], [1])


def test_mutual_information():
    targets = np.repeat(np.arange(4), 50)
    assert mutual_information_bits(targets, targets) == pytest.approx(2.0)
    assert mutual_information_bits(targets, (targets + 1) % 4) == pytest.approx(2.0)  # relabelled code
    assert mutual_information_bits(targets, np.zeros_like(targets)) == pytest.approx(0.0)


MEANINGS = np.array([(c, s) for c in range(5) for s in range(5)])


def test_entropy_and_spearman():
    assert entropy_bits([0, 1, 2, 3]) == pytest.approx(2.0) and entropy_bits([5, 5]) == 0
    assert spearman([1, 2, 3, 4], [10, 20, 30, 40]) == pytest.approx(1.0)
    assert spearman([1, 2, 3, 4], [4, 3, 2, 1]) == pytest.approx(-1.0)
    assert spearman([1, 1, 2, 2], [1, 2, 3, 4]) == pytest.approx(0.894427191)
    assert np.isnan(spearman([1, 1], [1, 2]))


def test_compositional_code_scores_one():
    messages = MEANINGS.copy()
    assert topographic_similarity(MEANINGS, messages) == pytest.approx(1.0)
    assert positional_disentanglement(MEANINGS, messages) == pytest.approx(1.0)


def test_holistic_code_scores_low():
    names = np.random.default_rng(0).permutation(64)[:25]
    messages = np.stack([names // 8, names % 8], axis=1)
    assert abs(topographic_similarity(MEANINGS, messages)) < 0.3
    assert positional_disentanglement(MEANINGS, messages) < 0.5


def test_topsim_uses_each_meanings_modal_message():
    meanings = np.array([[0, 0], [0, 0], [0, 0], [0, 1], [1, 0], [1, 1]])
    messages = np.array([[0, 0], [0, 0], [7, 7], [0, 1], [1, 0], [1, 1]])
    assert topographic_similarity(meanings, messages) == pytest.approx(1.0)
    assert np.isnan(topographic_similarity(meanings[:1], messages[:1]))
