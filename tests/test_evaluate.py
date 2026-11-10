import pytest

from emcomm import benchmark_v2
from emcomm.benchmark import N_EVAL_EPISODES, eval_plan
from emcomm.config import EnvConfig, PPOConfig
from emcomm.evaluate import evaluate_checkpoint
from emcomm.train import train

TINY = PPOConfig(n_envs=8, rollout_len=16, total_env_steps=8 * 16, hidden=32, n_minibatches=2, checkpoint_every=10)


@pytest.fixture(scope="module")
def comm_ckpt(tmp_path_factory):
    return train(EnvConfig(), TINY, "comm", seed=0, out_dir=tmp_path_factory.mktemp("comm"))


def test_evaluate_checkpoint_rows_and_determinism(comm_ckpt):
    episodes, rows = evaluate_checkpoint(comm_ckpt)
    assert len(rows) == 4 and len(episodes) == 4 * N_EVAL_EPISODES
    assert {(r["split"], r["intervention"]) for r in rows} == set(eval_plan("comm"))
    assert all(0.0 <= r["success"] <= 1.0 and r["n"] == N_EVAL_EPISODES for r in rows)
    _, again = evaluate_checkpoint(comm_ckpt)
    assert [r["success"] for r in rows] == [r["success"] for r in again]
    assert [r["mean_return"] for r in rows] == [r["mean_return"] for r in again]


def test_split_filter_never_touches_held_out(comm_ckpt):
    episodes, rows = evaluate_checkpoint(comm_ckpt, splits=("train",))
    assert {r["split"] for r in rows} == {"train"} and all(e["split"] == "train" for e in episodes)


V2 = EnvConfig(task="compositional", msg_words=2)


@pytest.fixture(scope="module")
def v2_ckpt(tmp_path_factory):
    return train(V2, TINY, "comm", seed=0, out_dir=tmp_path_factory.mktemp("v2"))


def test_v2_evaluation_rows(v2_ckpt, monkeypatch):
    # Validate every split with a scripted stand-in, never learned policies before prereg-v2.
    from emcomm.heuristics import ScriptedWordsTeam

    monkeypatch.setattr(
        "emcomm.evaluate.LearnedTeam", lambda nets, n_envs, seed: ScriptedWordsTeam(n_envs, runner_sees_cue=False)
    )
    episodes, rows = evaluate_checkpoint(v2_ckpt)
    assert {(r["split"], r["intervention"]) for r in rows} == set(benchmark_v2.eval_plan("comm"))
    assert all(set(r) == set(benchmark_v2.RUN_FIELDS_V2) for r in rows)
    assert len({str(r["topsim"]) for r in rows}) == 1
    assert len(episodes) == 10 * N_EVAL_EPISODES and "target_attrs" in episodes[0]


def test_v2_split_filter_never_touches_unseen(v2_ckpt):
    episodes, rows = evaluate_checkpoint(v2_ckpt, splits=("train",))
    assert {r["split"] for r in rows} == {"train"} and all(e["split"] == "train" for e in episodes)
    assert "topsim" not in rows[0]
