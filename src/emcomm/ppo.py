"""Recurrent PPO update for one independently learning agent (IPPO)."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.distributions import Categorical

from emcomm.config import PPOConfig
from emcomm.policy import AgentNet


@dataclass
class Rollout:
    obs: torch.Tensor  # (S, n, D) float32
    first: torch.Tensor  # (S, n) bool
    moves: torch.Tensor  # (S, n) int64
    msgs: torch.Tensor  # (S, n) int64, or (S, n, W) with several words
    logp: torch.Tensor  # (S, n) float32, joint log-prob at collection time
    values: torch.Tensor  # (S, n) float32
    rewards: torch.Tensor  # (S, n) float32, shared team reward
    dones: torch.Tensor  # (S, n) float32, 1.0 where the episode ended after step t
    h0: torch.Tensor  # (n, H) recurrent state at rollout start


def compute_gae(
    rewards: torch.Tensor, values: torch.Tensor, dones: torch.Tensor, last_value: torch.Tensor, gamma: float, lam: float
) -> tuple[torch.Tensor, torch.Tensor]:
    adv = torch.zeros_like(rewards)
    running = torch.zeros_like(last_value)
    for t in reversed(range(rewards.shape[0])):
        next_value = last_value if t == rewards.shape[0] - 1 else values[t + 1]
        nonterminal = 1.0 - dones[t]
        delta = rewards[t] + gamma * next_value * nonterminal - values[t]
        running = delta + gamma * lam * nonterminal * running
        adv[t] = running
    return adv, adv + values


def _entropy(q: torch.Tensor) -> torch.Tensor:
    return -(q * q.clamp_min(1e-8).log()).sum(-1)


def cue_information(msg_probs: torch.Tensor, cue: torch.Tensor) -> torch.Tensor:
    """Minibatch estimate of I(symbol; cue) in nats, over samples whose observation shows a cue.

    msg_probs (N, V) symbol distributions; cue (N, K) one-hot, or all-zero where no cue is shown.
    Returns 0 when no sample shows a cue (e.g. the uncued runner), so the bias is inert there.
    """
    seen = cue.sum(-1) > 0
    if not seen.any():
        return msg_probs.new_zeros(())
    p, groups = msg_probs[seen], cue[seen]
    counts = groups.sum(0)
    per_cue = (groups.T @ p) / counts.clamp_min(1)[:, None]  # (K, V) mean symbol distribution per cue
    return _entropy(p.mean(0)) - ((counts / counts.sum()) * _entropy(per_cue)).sum()


def joint_cue(cue: torch.Tensor, n_values: int) -> torch.Tensor:
    """Color then shape one-hots (N, 2V) -> one-hot over (color, shape) pairs (N, V*V).

    An all-zero row for an uncued agent stays all-zero.
    """
    return (cue[:, :n_values, None] * cue[:, None, n_values:]).flatten(1)


def word_cue_information(word_probs: torch.Tensor, cue: torch.Tensor) -> torch.Tensor:
    """Sum I(word; cue) over positions, without assigning a property to any word.

    word_probs is (N, W, V); cue is (N, K), one-hot or all-zero.
    """
    return sum(cue_information(word_probs[:, w], cue) for w in range(word_probs.shape[1]))


def pitch_cue_information(
    pitch_mean: torch.Tensor, log_std: torch.Tensor, cue: torch.Tensor, noise: float
) -> torch.Tensor:
    """Vocal analogue of `cue_information`: a lower bound on I(heard pitch; cue) in nats.

    Treats the heard pitch given cue k as a Gaussian centred on that cue's mean pitch, with variance =
    the spread of the pitch mean across that cue's samples + policy variance + hearing noise^2, and
    applies the pairwise Bhattacharyya bound (Kolchinsky & Tracey 2017). It is 0 when all cues share
    one pitch and log(K) when every pair of cues is distinguishable; an inconsistent "note" (a cue
    whose pitch varies across situations) counts as uninformative. `pitch_mean` is already in [-1, 1].
    0 when no sample shows a cue.
    """
    seen = cue.sum(-1) > 0
    if not seen.any():
        return pitch_mean.new_zeros(())
    means, groups = pitch_mean[seen], cue[seen]
    counts = groups.sum(0)
    present = counts > 0
    weights = counts / counts.sum()
    centres = (groups.T @ means) / counts.clamp_min(1)
    spread = (groups.T @ means.pow(2)) / counts.clamp_min(1) - centres.pow(2)
    var = spread.clamp_min(0) + log_std.exp().pow(2) + noise**2  # (K,)
    pooled = (var[:, None] + var[None, :]) / 2
    overlap = torch.exp(-(centres[:, None] - centres[None, :]).pow(2) / (8 * pooled)) * present[None, :]
    per_cue = torch.log((overlap * weights[None, :]).sum(1).clamp_min(1e-12))
    return -(weights * per_cue)[present].sum()


def ppo_update(
    net: AgentNet,
    opt: torch.optim.Optimizer,
    ro: Rollout,
    adv: torch.Tensor,
    returns: torch.Tensor,
    cfg: PPOConfig,
    cue_slice: slice | None = None,
    pitch_noise: float = 0.0,
    joint_values: int = 0,
) -> dict[str, float]:
    n_envs = ro.obs.shape[1]
    per_batch = n_envs // cfg.n_minibatches
    stats = dict.fromkeys(("policy_loss", "value_loss", "entropy_move", "entropy_msg", "msg_signal", "approx_kl"), 0.0)
    n_steps = 0
    for _ in range(cfg.epochs):
        order = torch.randperm(n_envs)
        for start in range(0, n_envs, per_batch):
            idx = order[start : start + per_batch]  # minibatch = whole env sequences
            move_logits, msg_logits, values, _ = net.forward_sequence(ro.obs[:, idx], ro.h0[idx], ro.first[:, idx])
            move_dist, msg_dist = Categorical(logits=move_logits), net.msg_distribution(msg_logits)
            logp = move_dist.log_prob(ro.moves[:, idx]) + msg_dist.log_prob(ro.msgs[:, idx])
            log_ratio = logp - ro.logp[:, idx]
            ratio = log_ratio.exp()
            a = adv[:, idx]
            a = (a - a.mean()) / (a.std() + 1e-8)
            policy_loss = -torch.min(ratio * a, ratio.clamp(1 - cfg.clip, 1 + cfg.clip) * a).mean()
            value_loss = 0.5 * (values - returns[:, idx]).pow(2).mean()
            ent_move, ent_msg = move_dist.entropy().mean(), msg_dist.entropy().mean()
            # informativeness bias (a positive-signalling variant, cf. Eccles et al. 2019): reward symbols that
            # depend on the agent's private cue. It says nothing about WHICH symbol should mean what.
            if cue_slice is None:
                signal = torch.zeros(())
            else:
                cue = ro.obs[:, idx][..., cue_slice].reshape(-1, cue_slice.stop - cue_slice.start)
                if joint_values:
                    cue = joint_cue(cue, joint_values)
                if net.vocal:
                    signal = pitch_cue_information(msg_dist.loc.reshape(-1), net.pitch_log_std, cue, pitch_noise)
                elif net.n_words > 1:
                    probs = msg_dist.base_dist.probs  # (S, b, W, V)
                    signal = word_cue_information(probs.reshape(-1, *probs.shape[-2:]), cue)
                else:
                    vocab = msg_dist.probs.shape[-1]
                    signal = cue_information(msg_dist.probs.reshape(-1, vocab), cue)
            loss = policy_loss + cfg.vf_coef * value_loss - cfg.ent_move * ent_move - cfg.ent_msg * ent_msg
            loss = loss - cfg.sig_coef * signal
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), cfg.max_grad_norm)
            opt.step()
            with torch.no_grad():
                stats["policy_loss"] += policy_loss.item()
                stats["value_loss"] += value_loss.item()
                stats["entropy_move"] += ent_move.item()
                stats["entropy_msg"] += ent_msg.item()
                stats["msg_signal"] += signal.item()
                stats["approx_kl"] += ((ratio - 1) - log_ratio).mean().item()
            n_steps += 1
    return {k: v / n_steps for k, v in stats.items()}
