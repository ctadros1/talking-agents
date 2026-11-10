"""Typed experiment configuration loaded from TOML."""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, fields
from pathlib import Path

CONDITIONS = ("comm", "no_comm", "random_msg", "oracle")
INTERVENTIONS = ("intact", "permuted", "silenced")
CHANNEL_KINDS = ("discrete", "vocal")
TASKS = ("types", "compositional")


@dataclass(frozen=True)
class EnvConfig:
    grid_size: int = 12
    wall_col: int = 4
    view_radius: int = 3
    n_item_types: int = 4  # items per episode (and, in the "types" task, item types)
    n_obstacles: int = 8
    max_steps: int = 32
    runner_slack: int = 12
    pickup_bonus: float = 0.1
    step_penalty: float = 0.01
    vocab_size: int = 8
    n_train_layouts: int = 48
    n_eval_layouts: int = 24
    layout_seed: int = 20260930
    channel: str = "discrete"  # "vocal": continuous pitch in [-1, 1] heard through Gaussian noise (exploratory arm)
    vocal_noise: float = 0.1  # std of the listener's hearing noise, in pitch units
    task: str = "types"  # "compositional": color x shape items in a 2x2 square (v2 spec section 2)
    n_values: int = 5  # colors, and shapes, in the compositional task
    msg_words: int = 1  # words per message (v2: 2)
    combo_seed: int = 20261003  # draws the held-out color-shape combinations (v2 spec section 2.3)

    def __post_init__(self) -> None:
        if self.task not in TASKS:
            raise ValueError(f"unknown task {self.task!r}; expected one of {TASKS}")
        if self.task == "compositional" and self.n_item_types != 4:
            raise ValueError("the compositional task places exactly 4 items (a 2x2 square)")
        if self.msg_words < 1 or (self.channel == "vocal" and self.msg_words != 1):
            raise ValueError("msg_words must be >= 1, and the vocal channel carries one pitch per step")

    @property
    def n_moves(self) -> int:
        return 6  # stay, up, down, left, right, interact

    @property
    def silence(self) -> int:
        return self.vocab_size  # reserved "no message" token index

    @property
    def view_size(self) -> int:
        return 2 * self.view_radius + 1

    @property
    def n_features(self) -> int:
        """Item feature channels: one per item type, or one per color followed by one per shape."""
        return self.n_item_types if self.task == "types" else 2 * self.n_values

    @property
    def n_view_channels(self) -> int:
        return 1 + self.n_features + 1  # wall, item features, depot

    @property
    def cue_slice(self) -> slice:
        """Where the target-cue features sit in the observation vector."""
        start = self.view_size**2 * self.n_view_channels + 2 + self.n_features + 1
        return slice(start, start + self.n_features)

    @property
    def obs_dim(self) -> int:
        view = self.view_size**2 * self.n_view_channels
        position, carrying, cue = 2, self.n_features + 1, self.n_features
        received, time_frac = self.msg_words * (self.vocab_size + 1), 1
        return view + position + carrying + cue + received + time_frac


@dataclass(frozen=True)
class PPOConfig:
    n_envs: int = 64
    rollout_len: int = 128
    total_env_steps: int = 10_000_000
    hidden: int = 128
    lr: float = 1e-3
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip: float = 0.2
    epochs: int = 4
    n_minibatches: int = 4
    vf_coef: float = 0.5
    ent_move: float = 0.01
    ent_msg: float = 0.01
    sig_coef: float = 0.2  # cue-informativeness bias weight (G2 froze rung 1; 0 = pure IPPO)
    max_grad_norm: float = 0.5
    checkpoint_every: int = 250  # updates

    @property
    def steps_per_update(self) -> int:
        return self.n_envs * self.rollout_len

    @property
    def n_updates(self) -> int:
        return self.total_env_steps // self.steps_per_update


def _build(cls, table: dict):
    known = {f.name for f in fields(cls)}
    unknown = set(table) - known
    if unknown:
        raise ValueError(f"unknown {cls.__name__} keys: {sorted(unknown)}")
    return cls(**table)


def load_config(path: str | Path) -> tuple[EnvConfig, PPOConfig]:
    with open(path, "rb") as fh:
        raw = tomllib.load(fh)
    return _build(EnvConfig, raw.get("env", {})), _build(PPOConfig, raw.get("ppo", {}))
