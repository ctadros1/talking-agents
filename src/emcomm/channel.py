"""Message routing for each experimental condition and evaluation intervention.

Every condition has the same observation and action space: each agent always emits one message per step
(one symbol, or one symbol per word position when `msg_words > 1`) and always observes one received message.
"""

from __future__ import annotations

import numpy as np

from emcomm.config import CONDITIONS

# channel mode used during training (and the default at evaluation)
TRAIN_CHANNEL = {"comm": "intact", "no_comm": "silenced", "random_msg": "random", "oracle": "silenced"}
CHANNEL_MODES = ("intact", "permuted", "permuted_w1", "permuted_w2", "silenced", "random", "shuffled")
# word positions each permutation intervention scrambles (None = every position; v2 spec section 3)
PERMUTED_WORDS = {"permuted": None, "permuted_w1": (0,), "permuted_w2": (1,)}


def cue_mask(condition: str) -> np.ndarray:
    """Which agents observe the target cue: the scout always, the runner only for the oracle."""
    if condition not in CONDITIONS:
        raise ValueError(f"unknown condition {condition!r}")
    return np.array([True, condition == "oracle"])


def initial_received(n_envs: int, vocab_size: int, vocal: bool = False, n_words: int = 1) -> np.ndarray:
    """At an episode's first step nobody has spoken yet: everyone hears silence (token `vocab_size`
    on the discrete channel, in every word position when n_words > 1; NaN on the vocal channel)."""
    if vocal:
        return np.full((n_envs, 2), np.nan)
    if n_words > 1:
        return np.full((n_envs, 2, n_words), vocab_size, dtype=np.int64)
    return np.full((n_envs, 2), vocab_size, dtype=np.int64)


def _other_episode(received: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Every listener hears the speaker of a different episode (random cyclic shift, never 0)."""
    if received.shape[0] < 2:
        raise ValueError("shuffled mode needs at least 2 episodes")
    return np.roll(received, int(rng.integers(1, received.shape[0])), axis=0)


def _route_vocal(received: np.ndarray, mode: str, rng: np.random.Generator, noise: float) -> np.ndarray:
    if mode == "silenced":
        return np.full_like(received, np.nan)
    if mode == "random":
        return rng.uniform(-1.0, 1.0, size=received.shape)
    if mode == "shuffled":
        received = _other_episode(received, rng)
    elif mode != "intact":
        raise ValueError(f"channel mode {mode!r} is not defined for the vocal channel")
    heard = received + rng.normal(0.0, noise, size=received.shape)
    return np.where(np.isnan(received), np.nan, np.clip(heard, -1.0, 1.0))


def route(
    sent: np.ndarray,
    mode: str,
    vocab_size: int,
    rng: np.random.Generator,
    perm: np.ndarray | None = None,
    noise: float = 0.0,
) -> np.ndarray:
    """Tokens each agent hears next step, given the symbols `sent` (n, 2) or (n, 2, words) this step.

    `perm` is a per-episode symbol permutation: (n, vocab_size) for one word,
    (n, words, vocab_size) for several; used by the permuted modes.
    A float `sent` array is the vocal channel: pitches in [-1, 1], heard with Gaussian `noise`.
    """
    received = sent[:, ::-1].copy()  # agent 0 hears agent 1 and vice versa
    if np.issubdtype(sent.dtype, np.floating):
        return _route_vocal(received, mode, rng, noise)
    if mode == "shuffled":
        return _other_episode(received, rng)
    if mode == "intact":
        return received
    if mode == "silenced":
        return np.full_like(received, vocab_size)
    if mode == "random":
        return rng.integers(vocab_size, size=received.shape).astype(np.int64)
    if mode in PERMUTED_WORDS:
        if perm is None:
            raise ValueError(f"{mode} mode needs perm")
        if received.ndim == 3:  # several words: one permutation per episode and word position
            n, _, words = received.shape
            return perm[np.arange(n)[:, None, None], np.arange(words)[None, None, :], received]
        return perm[np.arange(received.shape[0])[:, None], received]
    raise ValueError(f"unknown channel mode {mode!r}")


def episode_permutations(
    n_episodes: int, vocab_size: int, seed: int, n_words: int = 1, words: tuple[int, ...] | None = None
) -> np.ndarray:
    """Per-episode symbol permutations: (n, V) for one word, else (n, n_words, V). Every word position
    draws its permutation; positions not listed in `words` are then reset to the identity, so a
    single-word intervention uses exactly the permutation the full one uses for that position."""
    rng = np.random.default_rng(seed)
    if n_words == 1:
        return np.stack([rng.permutation(vocab_size) for _ in range(n_episodes)]).astype(np.int64)
    perm = np.array([[rng.permutation(vocab_size) for _ in range(n_words)] for _ in range(n_episodes)], dtype=np.int64)
    if words is not None:
        keep = [w for w in range(n_words) if w not in words]
        perm[:, keep] = np.arange(vocab_size)
    return perm
