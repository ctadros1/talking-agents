from collections import Counter

import numpy as np
import pytest

from emcomm.combos import (
    COLOR_NAMES,
    PROPERTIES,
    SHAPE_NAMES,
    combo_index,
    held_out_combos,
    sample_items,
    square_cells,
    squares_containing,
    training_squares,
)
from emcomm.config import EnvConfig

V2 = EnvConfig(task="compositional", msg_words=2)


def test_names():
    assert COLOR_NAMES == ("red", "yellow", "green", "blue", "purple")
    assert SHAPE_NAMES == ("cube", "sphere", "cone", "cylinder", "ring")
    assert PROPERTIES == ("color", "shape")


def test_held_out_is_one_per_color_and_one_per_shape():
    held = held_out_combos(V2)
    assert len(held) == 5
    assert sorted(c for c, _ in held) == list(range(5)) and sorted(s for _, s in held) == list(range(5))
    assert held == held_out_combos(EnvConfig(task="compositional", msg_words=2))  # fixed by combo_seed


def test_training_squares_avoid_held_out_and_cover_seen_pairs_evenly():
    squares = training_squares(V2)
    held = set(held_out_combos(V2))
    counts = Counter(cell for square in squares for cell in square_cells(square))
    assert len(squares) == 30
    assert not held & set(counts) and len(counts) == 20 and set(counts.values()) == {6}


def test_every_held_out_pair_lies_in_16_squares():
    assert all(len(squares_containing(V2, combo)) == 16 for combo in held_out_combos(V2))


def test_sample_items_seen_and_unseen():
    rng = np.random.default_rng(0)
    held = set(held_out_combos(V2))
    for _ in range(200):
        attrs, target = sample_items(V2, rng, "seen")
        cells = {tuple(int(v) for v in a) for a in attrs}
        assert attrs.shape == (4, 2) and len(cells) == 4
        assert len({c for c, _ in cells}) == 2 and len({s for _, s in cells}) == 2  # a full 2x2 square
        assert not cells & held and 0 <= target < 4
        attrs, target = sample_items(V2, rng, "unseen")
        assert tuple(int(v) for v in attrs[target]) in held
    with pytest.raises(ValueError, match="pool"):
        sample_items(V2, rng, "everything")


def test_combo_index():
    assert combo_index(np.array([[0, 0], [2, 3], [4, 4]]), 5).tolist() == [0, 13, 24]
