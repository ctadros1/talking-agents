import numpy as np

from emcomm.benchmark import N_EVAL_EPISODES, episode_sets, eval_plan, evaluate_scripted, summarize


def test_eval_plan():
    assert eval_plan("comm") == [
        ("held_out", "intact"),
        ("held_out", "permuted"),
        ("held_out", "silenced"),
        ("train", "intact"),
    ]
    assert eval_plan("oracle") == [("held_out", "silenced"), ("train", "silenced")]
    assert eval_plan("random_msg") == [("held_out", "random"), ("train", "random")]


def test_episode_sets_are_fixed_and_split_correctly(cfg, splits):
    a, b = episode_sets(cfg), episode_sets(cfg)
    assert len(a["held_out"]) == len(a["train"]) == N_EVAL_EPISODES
    assert [s.target for s in a["held_out"]] == [s.target for s in b["held_out"]]
    held_ids = {layout.layout_id for layout in splits[1]}
    assert all(s.layout.layout_id in held_ids for s in a["held_out"])
    assert not any(s.layout.layout_id in held_ids for s in a["train"])


def test_summarize():
    records = [
        {
            "success": True,
            "return": 1.0,
            "steps": 10,
            "path_efficiency": 0.8,
            "scout_msgs": [0],
            "target": 0,
            "runner_pick": 0,
        },
        {
            "success": False,
            "return": 0.0,
            "steps": 32,
            "path_efficiency": float("nan"),
            "scout_msgs": [1],
            "target": 1,
            "runner_pick": -1,
        },
    ]
    row = summarize(records)
    assert row["n"] == 2 and row["success"] == 0.5 and row["steps_success"] == 10.0
    assert row["path_efficiency"] == 0.8 and row["runner_pick_rate"] == 0.5 and np.isclose(row["mi_first_msg"], 1.0)


def test_scripted_reference_rows(cfg):
    rows = {(r["condition"], r["split"]): r["success"] for r in evaluate_scripted(cfg)}
    assert rows[("scripted_comm", "held_out")] >= 0.98 and rows[("scripted_oracle", "held_out")] >= 0.98
    assert 0.15 <= rows[("scripted_no_comm", "held_out")] <= 0.35
