import csv
import math

import numpy as np
import torch

from emcomm.channel import initial_received
from emcomm.config import EnvConfig, PPOConfig
from emcomm.env import GridEnv
from emcomm.evaluate import LearnedTeam
from emcomm.train import load_checkpoint, make_nets, train

TINY = PPOConfig(n_envs=8, rollout_len=16, total_env_steps=8 * 16 * 3, hidden=32, n_minibatches=2, checkpoint_every=2)
V2 = EnvConfig(task="compositional", msg_words=2)


def test_train_writes_metrics_and_checkpoints(tmp_path):
    final = train(EnvConfig(), TINY, "comm", seed=0, out_dir=tmp_path)
    rows = list(csv.DictReader(open(tmp_path / "metrics.csv")))
    assert len(rows) == 3 and rows[-1]["env_steps"] == str(8 * 16 * 3)
    assert (tmp_path / "update_00002.pt").exists() and final.exists()
    nets, env_cfg, ppo_cfg, ckpt = load_checkpoint(final)
    assert ckpt["condition"] == "comm" and ppo_cfg == TINY and env_cfg == EnvConfig() and len(nets) == 2


def test_training_is_bitwise_reproducible(tmp_path):
    a = load_checkpoint(train(EnvConfig(), TINY, "random_msg", seed=5, out_dir=tmp_path / "a"))[0]
    b = load_checkpoint(train(EnvConfig(), TINY, "random_msg", seed=5, out_dir=tmp_path / "b"))[0]
    for net_a, net_b in zip(a, b):
        for (_, pa), (_, pb) in zip(net_a.state_dict().items(), net_b.state_dict().items()):
            assert torch.equal(pa, pb)
    c = load_checkpoint(train(EnvConfig(), TINY, "random_msg", seed=6, out_dir=tmp_path / "c"))[0]
    assert not torch.equal(a[0].msg_head.weight, c[0].msg_head.weight)


def test_v2_training_runs_reloads_and_reproduces(tmp_path):
    final = train(V2, TINY, "comm", seed=0, out_dir=tmp_path / "a")
    nets, env_cfg, _, _ = load_checkpoint(final)
    assert env_cfg == V2 and nets[0].n_words == 2 and nets[0].msg_head.out_features == 16
    rows = list(csv.DictReader(open(tmp_path / "a" / "metrics.csv")))
    assert all(math.isfinite(float(r["msg_signal_0"])) for r in rows)
    again = load_checkpoint(train(V2, TINY, "comm", seed=0, out_dir=tmp_path / "b"))[0]
    for net_a, net_b in zip(nets, again):
        for (_, pa), (_, pb) in zip(net_a.state_dict().items(), net_b.state_dict().items()):
            assert torch.equal(pa, pb)


def test_v2_learned_team_emits_two_words():
    env = GridEnv(V2, n_envs=2)
    team = LearnedTeam(make_nets(V2, TINY), n_envs=2, seed=0)
    obs = np.zeros((2, 2, V2.obs_dim), dtype=np.float32)
    received = initial_received(2, V2.vocab_size, vocal=False, n_words=2)
    moves, sent = team.act(env, obs, received)
    assert moves.shape == (2, 2)
    assert sent.shape == (2, 2, 2)
