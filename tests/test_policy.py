import math

import torch

from emcomm.policy import AgentNet, sample_actions, sample_words


def test_forward_shapes_and_reset():
    net = AgentNet(obs_dim=10, n_moves=6, vocab_size=8, hidden=16)
    obs = torch.randn(5, 3, 10)
    first = torch.zeros(5, 3, dtype=torch.bool)
    move, msg, value, h = net.forward_sequence(obs, net.initial_state(3), first)
    assert move.shape == (5, 3, 6) and msg.shape == (5, 3, 8) and value.shape == (5, 3) and h.shape == (3, 16)
    # a reset at t=2 makes outputs from t=2 on independent of the prefix
    first[2] = True
    _, _, v_a, _ = net.forward_sequence(obs, torch.randn(3, 16), first)
    other = obs.clone()
    other[:2] = torch.randn(2, 3, 10)
    _, _, v_b, _ = net.forward_sequence(other, torch.randn(3, 16), first)
    assert torch.allclose(v_a[2:], v_b[2:]) and not torch.allclose(v_a[:2], v_b[:2])


def test_sample_actions_logp_and_generator():
    torch.manual_seed(0)
    ml, sl = torch.randn(4, 6), torch.randn(4, 8)
    m1, s1, lp = sample_actions(ml, sl, torch.Generator().manual_seed(3))
    m2, s2, _ = sample_actions(ml, sl, torch.Generator().manual_seed(3))
    assert torch.equal(m1, m2) and torch.equal(s1, s2)
    expected = torch.log_softmax(ml, -1)[torch.arange(4), m1] + torch.log_softmax(sl, -1)[torch.arange(4), s1]
    assert torch.allclose(lp, expected)


def test_two_word_heads_sampling_and_distribution():
    torch.manual_seed(0)
    net = AgentNet(obs_dim=10, n_moves=6, vocab_size=8, hidden=16, n_words=2)
    first = torch.ones(1, 4, dtype=torch.bool)
    move_logits, msg_logits, _, _ = net.forward_sequence(torch.randn(1, 4, 10), net.initial_state(4), first)
    assert net.n_words == 2 and msg_logits.shape == (1, 4, 16)
    move, words, logp = net.sample(move_logits[0], msg_logits[0], torch.Generator().manual_seed(0))
    assert words.shape == (4, 2) and words.max() < 8
    again = sample_words(move_logits[0], msg_logits[0], 2, torch.Generator().manual_seed(0))
    assert torch.equal(again[1], words)
    dist = net.msg_distribution(msg_logits[0])
    expected = torch.log_softmax(move_logits[0], -1)[torch.arange(4), move] + dist.log_prob(words)
    assert torch.allclose(logp, expected)
    assert dist.entropy().shape == (4,)
    assert torch.allclose(dist.entropy(), torch.full((4,), 2 * math.log(8)), atol=0.01)  # near uniform at init
    assert dist.base_dist.probs.shape == (4, 2, 8)
