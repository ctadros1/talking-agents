import numpy as np

from emcomm.channel import TRAIN_CHANNEL
from emcomm.config import EnvConfig
from emcomm.heuristics import ScriptedTeam, ScriptedWordsTeam
from emcomm.layouts import make_episode_set
from emcomm.rollout import run_episodes


def _success(cfg, specs, condition, mode):
    team = ScriptedTeam(len(specs), runner_sees_cue=condition == "oracle")
    records, _ = run_episodes(cfg, specs, team, condition, mode, seed=0)
    return np.mean([r["success"] for r in records])


def test_scripted_headroom(cfg, splits):
    """Gate G1 in miniature: communication is necessary and sufficient."""
    specs = make_episode_set(splits[1], 400, seed=11, cfg=cfg)
    assert _success(cfg, specs, "oracle", "silenced") >= 0.98
    assert _success(cfg, specs, "comm", "intact") >= 0.98
    no_comm = _success(cfg, specs, "no_comm", TRAIN_CHANNEL["no_comm"])
    assert 0.18 <= no_comm <= 0.32  # chance is 1/4: geometry must not leak the target
    assert _success(cfg, specs, "comm", "permuted") <= 0.40


V2 = EnvConfig(task="compositional", msg_words=2)


def _v2_success(specs, condition, mode, states=("color", "shape")):
    team = ScriptedWordsTeam(len(specs), runner_sees_cue=condition == "oracle", states=states)
    records, _ = run_episodes(V2, specs, team, condition, mode, seed=0)
    return np.mean([r["success"] for r in records])


def test_v2_scripted_headroom(splits):
    """Gate V2-G1 in miniature: both words are needed, one word gets half way."""
    seen = make_episode_set(splits[1], 400, seed=11, cfg=V2)
    unseen = make_episode_set(splits[1], 400, seed=12, cfg=V2, pool="unseen")
    for specs in (seen, unseen):
        assert _v2_success(specs, "comm", "intact") >= 0.98
        assert _v2_success(specs, "oracle", "silenced") >= 0.98
    assert 0.18 <= _v2_success(seen, "no_comm", "silenced") <= 0.32
    for states in (("color",), ("shape",)):
        assert 0.40 <= _v2_success(seen, "comm", "intact", states) <= 0.60
    assert _v2_success(seen, "comm", "permuted") <= 0.40
