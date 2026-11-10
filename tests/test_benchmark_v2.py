import pytest

from emcomm.benchmark_v2 import (
    RUN_FIELDS_V2,
    episode_sets,
    eval_plan,
    evaluate_scripted,
    run_level,
    summarize,
)
from emcomm.combos import held_out_combos
from emcomm.config import EnvConfig

V2 = EnvConfig(task="compositional", msg_words=2)


def test_eval_plan_v2():
    plan = eval_plan("comm")
    assert len(plan) == 10 and plan[0] == ("held_out", "intact") and plan[-1] == ("train", "intact")
    assert ("unseen", "permuted_w2") in plan and ("held_out", "silenced") in plan
    assert ("unseen", "silenced") not in plan
    assert eval_plan("no_comm") == [("held_out", "silenced"), ("unseen", "silenced"), ("train", "silenced")]
    assert eval_plan("random_msg")[1] == ("unseen", "random")


def test_episode_sets_v2():
    sets = episode_sets(V2)
    held = set(held_out_combos(V2))
    assert {len(v) for v in sets.values()} == {200}
    assert all(tuple(s.item_attrs[s.target].tolist()) in held for s in sets["unseen"])
    assert not any(held & {tuple(a) for a in s.item_attrs.tolist()} for s in sets["held_out"] + sets["train"])


def _rec(target, picked, first, success):
    return {
        "success": success,
        "return": 1.0 if success else 0.0,
        "steps": 10,
        "path_efficiency": 0.5 if success else float("nan"),
        "target_attrs": target,
        "picked_attrs": picked,
        "scout_msgs": [first],
    }


def test_summarize_property_metrics():
    records = [
        _rec([0, 1], [0, 1], [0, 1], True),
        _rec([0, 1], [0, 2], [0, 2], False),
        _rec([2, 3], [4, 3], [4, 3], False),
        _rec([2, 3], None, [2, 3], False),
    ]
    row = summarize(records, n_values=5)
    assert row["n"] == 4 and row["success"] == 0.25 and row["runner_pick_rate"] == 0.75
    assert row["color_acc"] == pytest.approx(2 / 3) and row["shape_acc"] == pytest.approx(2 / 3)
    assert row["twin_color"] == 0.5 and row["twin_shape"] == 0.5 and row["twin_opposite"] == 0.0
    assert row["mi_w1_color"] > 0 and set(row) <= set(RUN_FIELDS_V2)


def test_run_level_compositional_code():
    records = [_rec([c, s], [c, s], [c, s], True) for c in range(5) for s in range(5)]
    assert run_level(records) == pytest.approx({"topsim": 1.0, "posdis": 1.0})


def test_v2_scripted_reference_rows():
    rows = {(r["condition"], r["split"]): r["success"] for r in evaluate_scripted(V2)}
    assert len(rows) == 6 * 3
    assert rows[("scripted_comm", "unseen")] >= 0.98 and rows[("scripted_oracle", "held_out")] >= 0.98
    assert 0.40 <= rows[("scripted_color_only", "held_out")] <= 0.60
    assert 0.18 <= rows[("scripted_no_comm", "held_out")] <= 0.32
