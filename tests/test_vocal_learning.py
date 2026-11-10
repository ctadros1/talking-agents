import math
from dataclasses import replace

import numpy as np
import pytest
import torch
from torch.distributions import Categorical, Normal

from emcomm.benchmark import N_EVAL_EPISODES, eval_plan, pitch_bins, summarize
from emcomm.config import EnvConfig, PPOConfig
from emcomm.evaluate import evaluate_checkpoint
from emcomm.policy import AgentNet
from emcomm.ppo import pitch_cue_information
from emcomm.train import load_checkpoint, train

VOCAL = replace(EnvConfig(), channel="vocal")
TINY = PPOConfig(n_envs=8, rollout_len=16, total_env_steps=8 * 16 * 2, hidden=32, n_minibatches=2, sig_coef=0.2)


def test_vocal_net_has_a_gaussian_pitch_head():
    net = AgentNet(obs_dim=10, n_moves=6, vocab_size=8, hidden=16, vocal=True)
    move, msg, value, _ = net.forward_sequence(
        torch.randn(3, 4, 10), net.initial_state(4), torch.zeros(3, 4, dtype=torch.bool)
    )
    assert msg.shape == (3, 4, 1) and isinstance(net.msg_distribution(msg), Normal)
    m, u, logp = net.sample(move[0], msg[0], torch.Generator().manual_seed(0))
    expected = torch.log_softmax(move[0], -1)[torch.arange(4), m] + net.msg_distribution(msg[0]).log_prob(u)
    assert u.dtype == torch.float32 and torch.allclose(logp, expected)
    discrete = AgentNet(obs_dim=10, n_moves=6, vocab_size=8, hidden=16)
    assert not hasattr(discrete, "pitch_log_std") and isinstance(
        discrete.msg_distribution(torch.zeros(4, 8)), Categorical
    )


def test_pitch_cue_information():
    cue = torch.eye(4).repeat(26, 1)  # 104 samples, 26 per cue
    notes = torch.tensor([-0.75, -0.25, 0.25, 0.75]).repeat(26)
    tight = torch.tensor([math.log(0.01)])
    assert torch.isclose(pitch_cue_information(notes, tight, cue, 0.01), torch.tensor(math.log(4)), atol=1e-3)
    assert torch.isclose(pitch_cue_information(torch.zeros(104), tight, cue, 0.1), torch.tensor(0.0), atol=1e-6)
    two_notes = torch.tensor([-0.75, -0.75, 0.75, 0.75]).repeat(26)
    assert pitch_cue_information(two_notes, tight, cue, 0.1) < pitch_cue_information(notes, tight, cue, 0.1)
    flip = notes.clone()  # cues C and D alternate between the extremes across situations: inconsistent notes
    flip[2::4] = torch.tensor([-1.0, 1.0]).repeat(13)
    flip[3::4] = torch.tensor([1.0, -1.0]).repeat(13)
    assert pitch_cue_information(flip, tight, cue, 0.1) < pitch_cue_information(notes, tight, cue, 0.1) - 0.3
    assert pitch_cue_information(notes, tight, torch.zeros(104, 4), 0.1) == 0


def test_vocal_eval_plan_and_binned_summary():
    assert eval_plan("comm", "vocal") == [
        ("held_out", "intact"),
        ("held_out", "shuffled"),
        ("held_out", "silenced"),
        ("train", "intact"),
    ]
    assert eval_plan("no_comm", "vocal") == eval_plan("no_comm")
    assert pitch_bins(np.array([-1.0, -0.76, 0.0, 0.99, 1.0]), 8).tolist() == [0, 0, 4, 7, 7]
    records = [
        {
            "success": True,
            "return": 1.0,
            "steps": 9,
            "path_efficiency": 1.0,
            "target": k % 4,
            "scout_msgs": [(-0.75, -0.25, 0.25, 0.75)[k % 4]],
            "runner_pick": k % 4,
        }
        for k in range(40)
    ]
    assert np.isclose(summarize(records)["mi_first_msg"], 2.0)


@pytest.fixture(scope="module")
def vocal_ckpt(tmp_path_factory):
    return train(VOCAL, TINY, "comm", seed=0, out_dir=tmp_path_factory.mktemp("vocal"))


def test_vocal_training_is_reproducible_and_loads(vocal_ckpt, tmp_path):
    nets, env_cfg, _, ckpt = load_checkpoint(vocal_ckpt)
    assert env_cfg.channel == "vocal" and all(net.vocal for net in nets)
    again, *_ = load_checkpoint(train(VOCAL, TINY, "comm", seed=0, out_dir=tmp_path))
    assert torch.equal(nets[0].pitch_log_std, again[0].pitch_log_std)
    assert torch.equal(nets[1].msg_head.weight, again[1].msg_head.weight)


def test_vocal_evaluation(vocal_ckpt):
    episodes, rows = evaluate_checkpoint(vocal_ckpt)
    assert {(r["split"], r["intervention"]) for r in rows} == set(eval_plan("comm", "vocal"))
    assert len(episodes) == 4 * N_EVAL_EPISODES
    pitches = [e["scout_msgs"][0] for e in episodes]
    assert all(-1 <= p <= 1 for p in pitches) and all(np.isfinite(r["mi_first_msg"]) for r in rows)
