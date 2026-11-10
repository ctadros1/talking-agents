import json
from dataclasses import replace

from emcomm.config import EnvConfig
from emcomm.heuristics import ScriptedTeam, ScriptedWordsTeam
from emcomm.layouts import make_episode_set
from emcomm.rollout import run_episodes


def test_records_and_traces(cfg, splits):
    specs = make_episode_set(splits[1], 20, seed=5, cfg=cfg)
    records, traces = run_episodes(cfg, specs, ScriptedTeam(20, False), "comm", "intact", seed=0, record=(0, 3))
    assert len(records) == 20 and set(traces) == {0, 3}
    for r in records:
        assert 1 <= r["steps"] <= cfg.max_steps and len(r["scout_msgs"]) == r["steps"]
        assert r["scout_msgs"][0] == r["target"]  # scripted scout names the target
        if r["success"]:
            assert 0 < r["path_efficiency"] <= 1.0 and r["runner_pick"] == r["target"]
    assert len(traces[0]) == records[0]["steps"] + 1 and traces[0][-1]["moves"] is None
    assert traces[0][0]["received"] == [cfg.silence, cfg.silence]


def test_rollout_is_reproducible(cfg, splits):
    specs = make_episode_set(splits[1], 30, seed=6, cfg=cfg)
    a, _ = run_episodes(cfg, specs, ScriptedTeam(30, False), "random_msg", "random", seed=2)
    b, _ = run_episodes(cfg, specs, ScriptedTeam(30, False), "random_msg", "random", seed=2)
    assert json.dumps(a) == json.dumps(b)  # json: NaN path_efficiency compares equal


V2 = EnvConfig(task="compositional", msg_words=2)


def test_v2_records_hold_attributes_and_two_word_messages(cfg, splits):
    specs = make_episode_set(splits[1], 20, seed=5, cfg=V2)
    records, traces = run_episodes(V2, specs, ScriptedWordsTeam(20, False), "comm", "intact", seed=0, record=(0,))
    for r, s in zip(records, specs):
        assert r["target_attrs"] == s.item_attrs[s.target].tolist()
        assert r["scout_msgs"][0] == r["target_attrs"]
        assert len(r["scout_msgs"]) == r["steps"]
        if r["success"]:
            assert r["picked_attrs"] == r["target_attrs"]
    assert traces[0][0]["received"] == [[V2.silence, V2.silence], [V2.silence, V2.silence]]
    assert traces[0][0]["sent"] == [records[0]["target_attrs"], [0, 0]]
    assert traces[0][1]["received"][1] == records[0]["target_attrs"]
    v1_specs = make_episode_set(splits[1], 5, seed=5, cfg=cfg)
    v1_records, _ = run_episodes(cfg, v1_specs, ScriptedTeam(5, False), "comm", "intact", seed=0)
    assert "target_attrs" not in v1_records[0]


def test_v2_picked_attrs_is_none_before_first_pick(splits):
    specs = make_episode_set(splits[1], 1, seed=5, cfg=V2)
    records, _ = run_episodes(replace(V2, max_steps=1), specs, ScriptedWordsTeam(1, False), "comm", "intact", seed=0)
    assert records[0]["runner_pick"] == -1
    assert records[0]["picked_attrs"] is None
