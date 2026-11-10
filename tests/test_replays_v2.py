from pathlib import Path
from types import SimpleNamespace

import numpy as np

import emcomm.replays_v2 as replays_v2
from emcomm.replays_v2 import VARIANTS, pick_v2


def _r(i, color, shape, ok):
    return {"episode": i, "target_attrs": [color, shape], "success": ok}


def test_pick_covers_available_colors_shapes_and_failures():
    held = [_r(i, i % 5, (2 * i) % 5, i % 4 != 0) for i in range(40)]
    unseen = [_r(i, 0, 1, i % 2 == 0) for i in range(6)]

    chosen = pick_v2(held, unseen)
    picked = [held[i] for i in chosen["held_out"]]
    wins = [r for r in picked if r["success"]]

    assert {r["target_attrs"][0] for r in wins} == set(range(5))
    assert {r["target_attrs"][1] for r in wins} == set(range(5))
    assert sum(not r["success"] for r in picked) == 2
    assert chosen["held_out"] == sorted(set(chosen["held_out"]))
    assert chosen["unseen"] == [0, 1]  # one success and one failure


def test_pick_reports_sparse_coverage_without_inventing_successes():
    held = [_r(0, 2, 4, False), _r(1, 2, 4, False), _r(2, 2, 4, False)]
    unseen = [_r(0, 0, 1, False), _r(1, 1, 2, False), _r(2, 2, 3, False)]

    chosen = pick_v2(held, unseen)

    assert chosen["held_out"] == [0, 1]
    assert chosen["unseen"] == [0, 1]


def test_pick_keeps_sparse_success_and_reports_only_its_attributes():
    held = [_r(0, 2, 4, True), _r(1, 1, 3, False), _r(2, 0, 2, False)]

    chosen = pick_v2(held, [])

    picked = [held[i] for i in chosen["held_out"]]
    wins = [r for r in picked if r["success"]]
    assert [r["episode"] for r in wins] == [0]
    assert {r["target_attrs"][0] for r in wins} == {2}
    assert {r["target_attrs"][1] for r in wins} == {4}
    assert chosen["unseen"] == []


def test_variants():
    assert VARIANTS == (
        ("comm", "intact"),
        ("comm", "permuted"),
        ("comm", "permuted_w1"),
        ("no_comm", "silenced"),
    )


def test_build_bundle_uses_fresh_teams_and_replays_selected_base_outcomes(monkeypatch, tmp_path):
    cfg = SimpleNamespace(
        grid_size=12,
        wall_col=4,
        view_radius=3,
        max_steps=32,
        vocab_size=8,
        n_values=5,
        msg_words=2,
    )
    sets = {
        "held_out": [
            SimpleNamespace(split="held_out", item_attrs=np.array([[0, 0], [1, 1], [2, 2], [3, 3]])) for _ in range(4)
        ],
        "unseen": [
            SimpleNamespace(split="unseen", item_attrs=np.array([[0, 0], [1, 1], [2, 2], [3, 3]])) for _ in range(4)
        ],
    }

    class FakeTeam:
        instances = []

        def __init__(self, nets, n_envs, seed):
            self.seed = seed
            self.instances.append(self)

    baselines = {
        "held_out": [_r(0, 0, 0, True), _r(1, 1, 1, True), _r(2, 2, 2, False), _r(3, 3, 3, False)],
        "unseen": [_r(0, 4, 4, True), _r(1, 0, 0, False), _r(2, 1, 1, True), _r(3, 2, 2, False)],
    }
    run_calls = []

    def fake_run_episodes(cfg, specs, team, condition, mode, seed, record=()):
        split = specs[0].split
        run_calls.append((split, team, condition, mode, tuple(record)))
        records = [dict(row) for row in baselines[split]]
        traces = {i: [{"episode": i}] for i in record}
        return records, traces

    monkeypatch.setattr(replays_v2.benchmark_v2, "episode_sets", lambda cfg: sets)
    monkeypatch.setattr(replays_v2, "load_checkpoint", lambda path: ([object()], cfg, None, {}))
    monkeypatch.setattr(replays_v2, "LearnedTeam", FakeTeam)
    monkeypatch.setattr(replays_v2, "run_episodes", fake_run_episodes)
    monkeypatch.setattr(replays_v2, "spec_json", lambda spec: {"layout_id": spec.split})
    monkeypatch.setattr(replays_v2, "word_tables", lambda records, n_values, vocab_size: {"rows": len(records)})

    bundle = replays_v2.build_bundle(Path(tmp_path), 12)

    first_pass_teams = [team for split, team, condition, mode, ids in run_calls if not ids]
    assert len(first_pass_teams) == 2
    assert first_pass_teams[0] is not first_pass_teams[1]
    assert all(team.seed == replays_v2.EVAL_SEED for team in first_pass_teams)
    for split, ids in replays_v2.pick_v2(baselines["held_out"], baselines["unseen"]).items():
        for i in ids:
            assert bundle["variants"]["comm/intact"][f"{split}/{i}"]["record"] == baselines[split][i]
