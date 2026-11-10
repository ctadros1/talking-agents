import math

import torch
from torch.distributions import Categorical

from emcomm.config import PPOConfig
from emcomm.policy import AgentNet
from emcomm.ppo import Rollout, compute_gae, cue_information, joint_cue, ppo_update, word_cue_information


def test_gae_matches_hand_computation():
    rewards = torch.tensor([[1.0], [0.0], [2.0]])
    values = torch.tensor([[0.5], [0.4], [0.3]])
    dones = torch.tensor([[0.0], [1.0], [0.0]])
    last = torch.tensor([0.2])
    adv, ret = compute_gae(rewards, values, dones, last, gamma=0.9, lam=0.8)
    d2 = 2.0 + 0.9 * 0.2 - 0.3
    d1 = 0.0 - 0.4  # episode ends after step 1: no bootstrap
    d0 = 1.0 + 0.9 * 0.4 - 0.5
    assert torch.allclose(adv[:, 0], torch.tensor([d0 + 0.9 * 0.8 * d1, d1, d2]))
    assert torch.allclose(ret, adv + values)


def test_ppo_update_changes_params_and_reports_finite_stats():
    torch.manual_seed(0)
    net = AgentNet(obs_dim=10, n_moves=6, vocab_size=8, hidden=16)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    steps, n = 8, 4
    obs = torch.randn(steps, n, 10)
    first = torch.zeros(steps, n, dtype=torch.bool)
    first[0] = True
    with torch.no_grad():
        move_logits, msg_logits, values, _ = net.forward_sequence(obs, net.initial_state(n), first)
    moves, msgs = torch.randint(6, (steps, n)), torch.randint(8, (steps, n))
    logp = torch.log_softmax(move_logits, -1).gather(-1, moves[..., None]).squeeze(-1)
    logp = logp + torch.log_softmax(msg_logits, -1).gather(-1, msgs[..., None]).squeeze(-1)
    ro = Rollout(
        obs=obs,
        first=first,
        moves=moves,
        msgs=msgs,
        logp=logp,
        values=values,
        rewards=torch.randn(steps, n),
        dones=torch.zeros(steps, n),
        h0=net.initial_state(n),
    )
    adv, ret = compute_gae(ro.rewards, ro.values, ro.dones, torch.zeros(n), 0.99, 0.95)
    before = [p.clone() for p in net.parameters()]
    stats = ppo_update(net, opt, ro, adv, ret, PPOConfig(epochs=2, n_minibatches=2))
    assert set(stats) == {"policy_loss", "value_loss", "entropy_move", "entropy_msg", "msg_signal", "approx_kl"}
    assert all(math.isfinite(v) for v in stats.values())
    assert any(not torch.equal(b, p) for b, p in zip(before, net.parameters()))
    assert abs(stats["entropy_msg"] - math.log(8)) < 0.05  # near-uniform symbol policy at init (gain 0.01)


def test_cue_information():
    cue = torch.eye(4).repeat(10, 1)  # 40 samples, each cue shown 10 times
    informative = torch.eye(8)[:4].repeat(10, 1)  # cue k -> symbol k deterministically
    assert torch.isclose(cue_information(informative, cue), torch.tensor(math.log(4)), atol=1e-5)
    constant = torch.full((40, 8), 1 / 8)
    assert torch.isclose(cue_information(constant, cue), torch.tensor(0.0), atol=1e-6)
    assert cue_information(informative, torch.zeros(40, 4)) == 0  # uncued agent: bias is inert


def test_joint_cue():
    cue = torch.tensor([[0, 1, 0, 0, 0, 0, 0, 0, 1, 0], [0.0] * 10])
    joint = joint_cue(cue, 5)
    assert joint.shape == (2, 25)
    assert joint[0].argmax() == 1 * 5 + 3
    assert joint[0].sum() == 1
    assert joint[1].sum() == 0


def test_word_cue_information_sums_over_words():
    pairs = [(c, s) for c in range(5) for s in range(5)]
    cue = torch.eye(25).repeat(4, 1)
    probs = torch.zeros(100, 2, 8)
    for i in range(100):
        c, s = pairs[i % 25]
        probs[i, 0, c], probs[i, 1, s] = 1.0, 1.0
    assert torch.isclose(word_cue_information(probs, cue), torch.tensor(2 * math.log(5)), atol=1e-5)
    assert word_cue_information(probs, torch.zeros(100, 25)) == 0


def test_two_word_ppo_update():
    torch.manual_seed(0)
    net = AgentNet(obs_dim=12, n_moves=6, vocab_size=8, hidden=16, n_words=2)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    steps, n = 8, 4
    obs = torch.zeros(steps, n, 12)
    for t in range(steps):
        for i in range(n):
            obs[t, i, 2 + (t + i) % 5] = 1.0
            obs[t, i, 7 + (t * i) % 5] = 1.0
    first = torch.zeros(steps, n, dtype=torch.bool)
    first[0] = True
    with torch.no_grad():
        move_logits, msg_logits, values, _ = net.forward_sequence(obs, net.initial_state(n), first)
        moves, msgs = torch.randint(6, (steps, n)), torch.randint(8, (steps, n, 2))
        logp = Categorical(logits=move_logits).log_prob(moves) + net.msg_distribution(msg_logits).log_prob(msgs)
    ro = Rollout(
        obs=obs,
        first=first,
        moves=moves,
        msgs=msgs,
        logp=logp,
        values=values,
        rewards=torch.randn(steps, n),
        dones=torch.zeros(steps, n),
        h0=net.initial_state(n),
    )
    adv, ret = compute_gae(ro.rewards, ro.values, ro.dones, torch.zeros(n), 0.99, 0.95)
    cfg = PPOConfig(epochs=1, n_minibatches=2, sig_coef=0.2)
    stats = ppo_update(net, opt, ro, adv, ret, cfg, cue_slice=slice(2, 12), joint_values=5)
    assert all(math.isfinite(v) for v in stats.values()) and stats["msg_signal"] >= -1e-6
    assert abs(stats["entropy_msg"] - 2 * math.log(8)) < 0.1
