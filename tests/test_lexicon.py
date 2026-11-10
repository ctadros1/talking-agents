import csv
import json
import sys

import numpy as np

from emcomm.lexicon import LOG_FIELDS, intact_records, main, message_log, word_tables


def _rec(c, s, first, split="held_out", success=True, picked=None, intervention="intact"):
    return {
        "episode": 7,
        "split": split,
        "intervention": intervention,
        "target_attrs": [c, s],
        "picked_attrs": picked,
        "scout_msgs": [first, first],
        "success": success,
    }


def test_word_tables_of_a_compositional_code():
    records = [_rec(c, s, [c, s + 3]) for c in range(5) for s in range(5)]
    tables = word_tables(records, n_values=5, vocab_size=8)
    assert np.array_equal(np.array(tables["word1_given_color"]), np.eye(5, 8))
    assert np.allclose(np.array(tables["word1_given_shape"])[:, :5], 0.2)
    assert np.array(tables["word2_given_shape"])[2, 5] == 1.0


def test_message_log_rows():
    (row,) = message_log([_rec(3, 2, [1, 6], split="unseen", success=False, picked=[3, 4])], seed=11)
    assert row == {
        "seed": 11,
        "episode": 7,
        "split": "unseen",
        "target_color": "blue",
        "target_shape": "cone",
        "first_word1": 1,
        "first_word2": 6,
        "all_words": "1-6 1-6",
        "picked_color": "blue",
        "picked_shape": "ring",
        "success": False,
    }
    assert list(row) == LOG_FIELDS


def test_intact_records_keep_intact_held_out_and_unseen(tmp_path):
    path = tmp_path / "eval.jsonl"
    lines = [
        _rec(0, 0, [0, 0]),
        _rec(0, 0, [0, 0], split="unseen"),
        _rec(0, 0, [0, 0], intervention="permuted"),
        _rec(0, 0, [0, 0], split="train"),
    ]
    path.write_text("".join(json.dumps(r) + "\n" for r in lines))
    assert [(r["split"], r["intervention"]) for r in intact_records(path)] == [
        ("held_out", "intact"),
        ("unseen", "intact"),
    ]


def test_cli_exports_each_seed_and_representative_log(tmp_path, monkeypatch):
    runs, results = tmp_path / "runs", tmp_path / "results"
    results.mkdir()
    with open(results / "per_run.csv", "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["condition", "seed", "split", "intervention", "success"])
        writer.writeheader()
        for seed, success in ((3, 0.7), (4, 0.8), (5, 0.9)):
            writer.writerow(
                {"condition": "comm", "seed": seed, "split": "held_out", "intervention": "intact", "success": success}
            )
            path = runs / "comm" / f"seed_{seed:04d}" / "eval.jsonl"
            path.parent.mkdir(parents=True)
            path.write_text("".join(json.dumps(_rec(0, 1, [1, 2], split=s)) + "\n" for s in ("held_out", "unseen")))
    monkeypatch.setattr(
        sys,
        "argv",
        ["lexicon", "--runs", str(runs), "--results", str(results), "--config", "configs/v2.toml"],
    )
    main()
    lexicon = json.loads((results / "lexicon.json").read_text())
    assert lexicon["representative_seed"] == 4
    assert set(lexicon["seeds"]) == {"3", "4", "5"}
    with open(results / "message_log.csv", newline="") as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == 2
    assert all(row["seed"] == "4" for row in rows)
    assert set(row["split"] for row in rows) == {"held_out", "unseen"}
