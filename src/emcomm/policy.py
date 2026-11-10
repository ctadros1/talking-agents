"""Recurrent actor-critic for one agent: factorized move and message heads (one or more words)."""

from __future__ import annotations

import torch
from torch import nn
from torch.distributions import Categorical, Independent, Normal


def _linear(n_in: int, n_out: int, gain: float) -> nn.Linear:
    layer = nn.Linear(n_in, n_out)
    nn.init.orthogonal_(layer.weight, gain)
    nn.init.zeros_(layer.bias)
    return layer


class AgentNet(nn.Module):
    def __init__(self, obs_dim: int, n_moves: int, vocab_size: int, hidden: int, vocal: bool = False, n_words: int = 1):
        super().__init__()
        self.hidden, self.vocal, self.n_words = hidden, vocal, n_words
        self.encoder = nn.Sequential(
            _linear(obs_dim, hidden, 2**0.5), nn.ReLU(), _linear(hidden, hidden, 2**0.5), nn.ReLU()
        )
        self.gru = nn.GRUCell(hidden, hidden)
        self.move_head = _linear(hidden, n_moves, 0.01)
        self.msg_head = _linear(hidden, 1 if vocal else vocab_size * n_words, 0.01)  # vocal: pitch mean = tanh(output)
        self.value_head = _linear(hidden, 1, 1.0)
        if vocal:
            self.pitch_log_std = nn.Parameter(torch.full((1,), -1.0))  # initial pitch std ~0.37

    def msg_distribution(self, msg_out: torch.Tensor) -> Categorical | Independent | Normal:
        """Symbol distribution (one word), independent per-word symbol distributions whose log-probs and
        entropies sum over word positions (several words), or N(tanh(output), std) (vocal)."""
        if self.vocal:
            return Normal(torch.tanh(msg_out.squeeze(-1)), self.pitch_log_std.exp())
        if self.n_words > 1:
            return Independent(Categorical(logits=msg_out.unflatten(-1, (self.n_words, -1))), 1)
        return Categorical(logits=msg_out)

    def sample(
        self, move_logits: torch.Tensor, msg_out: torch.Tensor, generator: torch.Generator | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """(move, message, joint log-prob) for one step; the message is a symbol, a (n, words) array of
        symbols, or a raw pitch sample (emitted as clip(sample, -1, 1))."""
        if self.vocal:
            return sample_pitch(move_logits, torch.tanh(msg_out.squeeze(-1)), self.pitch_log_std, generator)
        if self.n_words > 1:
            return sample_words(move_logits, msg_out, self.n_words, generator)
        return sample_actions(move_logits, msg_out, generator)

    def initial_state(self, n: int) -> torch.Tensor:
        return torch.zeros(n, self.hidden)

    def forward_sequence(
        self, obs: torch.Tensor, h0: torch.Tensor, first: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """obs (S, n, D), h0 (n, H), first (S, n) bool: True where obs[t] starts an episode.

        Returns move logits (S, n, M), message logits (S, n, V * words), values (S, n), final state (n, H).
        """
        x = self.encoder(obs)
        h, states = h0, []
        for t in range(obs.shape[0]):
            h = h * (~first[t]).float().unsqueeze(1)  # forget across episode boundaries
            h = self.gru(x[t], h)
            states.append(h)
        hs = torch.stack(states)
        return self.move_head(hs), self.msg_head(hs), self.value_head(hs).squeeze(-1), h


def sample_actions(
    move_logits: torch.Tensor, msg_logits: torch.Tensor, generator: torch.Generator | None = None
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Sample (move, message) for a batch; returns both and the joint log-probability."""
    move_lp, msg_lp = torch.log_softmax(move_logits, -1), torch.log_softmax(msg_logits, -1)
    move = torch.multinomial(move_lp.exp(), 1, generator=generator).squeeze(1)
    msg = torch.multinomial(msg_lp.exp(), 1, generator=generator).squeeze(1)
    logp = move_lp.gather(1, move[:, None]).squeeze(1) + msg_lp.gather(1, msg[:, None]).squeeze(1)
    return move, msg, logp


def sample_words(
    move_logits: torch.Tensor, msg_logits: torch.Tensor, n_words: int, generator: torch.Generator | None = None
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Sample a move and one symbol per word position; returns (move (n,), words (n, n_words), joint log-prob)."""
    move_lp = torch.log_softmax(move_logits, -1)
    move = torch.multinomial(move_lp.exp(), 1, generator=generator).squeeze(1)
    word_lp = torch.log_softmax(msg_logits.unflatten(-1, (n_words, -1)), -1)  # (n, W, V)
    words = torch.multinomial(word_lp.exp().flatten(0, 1), 1, generator=generator).view(-1, n_words)
    logp = move_lp.gather(1, move[:, None]).squeeze(1) + word_lp.gather(2, words[..., None]).squeeze(-1).sum(-1)
    return move, words, logp


def sample_pitch(
    move_logits: torch.Tensor, pitch_mean: torch.Tensor, log_std: torch.Tensor, generator: torch.Generator | None = None
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Vocal channel: sample a move and a raw pitch u ~ N(mean, std); the emitted pitch is clip(u, -1, 1).

    Returns (move (n,), u (n,), joint log-probability of (move, u))."""
    move_lp = torch.log_softmax(move_logits, -1)
    move = torch.multinomial(move_lp.exp(), 1, generator=generator).squeeze(1)
    std = log_std.exp().expand_as(pitch_mean)
    u = pitch_mean + std * torch.randn(pitch_mean.shape, generator=generator)
    logp = move_lp.gather(1, move[:, None]).squeeze(1) + Normal(pitch_mean, std).log_prob(u)
    return move, u, logp
