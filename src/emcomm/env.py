"""Vectorized, deterministic two-agent split-information grid world."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from emcomm.config import EnvConfig
from emcomm.layouts import EpisodeSpec

STAY, UP, DOWN, LEFT, RIGHT, INTERACT = range(6)
DELTAS = np.array([[0, 0], [-1, 0], [1, 0], [0, -1], [0, 1], [0, 0]], dtype=np.int64)


@dataclass
class StepResult:
    reward: np.ndarray  # (n,) float32 shared team reward
    done: np.ndarray  # (n,) bool, episode ended this step
    success: np.ndarray  # (n,) bool, correct item delivered this step
    picked: np.ndarray  # (n, 2) int64, item type picked by each agent this step, -1 if none


def one_hot(index: np.ndarray, depth: int) -> np.ndarray:
    out = np.zeros((index.shape[0], depth), dtype=np.float32)
    out[np.arange(index.shape[0]), index] = 1.0
    return out


class GridEnv:
    """A batch of `n_envs` independent episodes stepped in lockstep.

    Agents act simultaneously; within a step agent 0 (scout) resolves before
    agent 1 (runner). Agents never observe each other, so the only information
    path between them is the message channel (see `emcomm.channel`).
    """

    def __init__(self, cfg: EnvConfig, n_envs: int):
        self.cfg, self.n = cfg, n_envs
        g, k = cfg.grid_size, cfg.n_item_types
        self.walls = np.zeros((n_envs, g, g), dtype=bool)
        self.item_pos = np.zeros((n_envs, k, 2), dtype=np.int64)
        self.item_present = np.zeros((n_envs, k), dtype=bool)
        self.item_attrs = np.zeros((n_envs, k, 2), dtype=np.int64)  # color, shape (compositional task)
        self.depot = np.zeros((n_envs, 2), dtype=np.int64)
        self.pos = np.zeros((n_envs, 2, 2), dtype=np.int64)
        self.carrying = np.full((n_envs, 2), -1, dtype=np.int64)
        self.target = np.zeros(n_envs, dtype=np.int64)
        self.t = np.zeros(n_envs, dtype=np.int64)
        self.specs: list[EpisodeSpec | None] = [None] * n_envs

    def load(self, i: int, spec: EpisodeSpec) -> None:
        self.walls[i] = spec.layout.walls
        self.item_pos[i] = spec.item_pos
        self.item_present[i] = True
        if spec.item_attrs is not None:
            self.item_attrs[i] = spec.item_attrs
        self.depot[i] = spec.depot
        self.pos[i] = spec.agent_pos
        self.carrying[i] = -1
        self.target[i] = spec.target
        self.t[i] = 0
        self.specs[i] = spec

    def _views(self) -> np.ndarray:
        cfg, n = self.cfg, self.n
        g, rad, k = cfg.grid_size, cfg.view_radius, cfg.n_item_types
        size, chans = cfg.view_size, cfg.n_view_channels
        grid = np.zeros((n, chans, g + 2 * rad, g + 2 * rad), dtype=np.float32)
        grid[:, 0] = 1.0  # out of bounds reads as wall
        grid[:, 0, rad : rad + g, rad : rad + g] = self.walls
        envs = np.arange(n)
        for j in range(k):
            here = self.item_present[:, j]
            rows, cols = self.item_pos[here, j, 0] + rad, self.item_pos[here, j, 1] + rad
            if cfg.task == "types":
                grid[envs[here], 1 + j, rows, cols] = 1.0
            else:  # one channel per color, then one per shape
                grid[envs[here], 1 + self.item_attrs[here, j, 0], rows, cols] = 1.0
                grid[envs[here], 1 + cfg.n_values + self.item_attrs[here, j, 1], rows, cols] = 1.0
        grid[envs, 1 + cfg.n_features, self.depot[:, 0] + rad, self.depot[:, 1] + rad] = 1.0
        offsets = np.arange(size)
        views = np.empty((n, 2, chans * size * size), dtype=np.float32)
        for a in range(2):
            rows = self.pos[:, a, 0][:, None] + offsets  # window start = pos (padded coords)
            cols = self.pos[:, a, 1][:, None] + offsets
            crop = grid[
                envs[:, None, None, None],
                np.arange(chans)[None, :, None, None],
                rows[:, None, :, None],
                cols[:, None, None, :],
            ]
            if cfg.task == "compositional" and a == 0:
                crop[:, 1:] = 0.0  # scout sees control-room walls, never items or depot
                view_rows, view_cols = rows - rad, cols - rad
                warehouse = (
                    (view_rows >= 0)[:, :, None]
                    & (view_rows < g)[:, :, None]
                    & (view_cols > cfg.wall_col)[:, None, :]
                    & (view_cols < g)[:, None, :]
                )
                crop[:, 0] = np.where(warehouse, 0.0, crop[:, 0])
            views[:, a] = crop.reshape(n, -1)
        return views

    def observe(self, received: np.ndarray, cue_mask: np.ndarray) -> np.ndarray:
        """Observations (n, 2, obs_dim). `received` holds what each agent hears this step: (n, 2) tokens,
        or (n, 2, words) with several words; `cue_mask` (2,) says which agents see the target."""
        cfg = self.cfg
        views = self._views()
        target = self._features(self.target)
        time_frac = (self.t / cfg.max_steps).astype(np.float32)[:, None]
        per_agent = []
        for a in range(2):
            per_agent.append(
                np.concatenate(
                    [
                        views[:, a],
                        (self.pos[:, a] / (cfg.grid_size - 1)).astype(np.float32),
                        self._carrying_block(a),
                        target * float(cue_mask[a]),
                        self._heard(received[:, a]),
                        time_frac,
                    ],
                    axis=1,
                )
            )
        return np.stack(per_agent, axis=1)

    def _features(self, item: np.ndarray) -> np.ndarray:
        """Features (n, n_features) of item index `item` in each env: a type one-hot (types task), or a
        color one-hot followed by a shape one-hot (compositional task)."""
        cfg = self.cfg
        if cfg.task == "types":
            return one_hot(item, cfg.n_item_types)
        attrs = self.item_attrs[np.arange(self.n), item]
        return np.concatenate([one_hot(attrs[:, 0], cfg.n_values), one_hot(attrs[:, 1], cfg.n_values)], axis=1)

    def _carrying_block(self, a: int) -> np.ndarray:
        """[nothing] then the carried item's features; in the types task this is v1's one-hot [none, A, B, C, D]."""
        cfg, carried = self.cfg, self.carrying[:, a]
        if cfg.task == "types":
            return one_hot(carried + 1, cfg.n_item_types + 1)
        empty = carried < 0
        features = self._features(np.maximum(carried, 0)) * ~empty[:, None]
        return np.concatenate([empty[:, None].astype(np.float32), features], axis=1)

    def _heard(self, received: np.ndarray) -> np.ndarray:
        """The received-token block (vocab_size + 1 floats). Discrete channel: a one-hot token.
        Vocal channel: a cochlea-like bank of vocab_size Gaussian tuning curves evenly spaced over
        [-1, 1] responding to the heard pitch, plus an is-silent flag (same width, so obs_dim is fixed)."""
        v = self.cfg.vocab_size
        if received.ndim == 2:  # several words: one one-hot block per word position
            return np.concatenate([one_hot(received[:, w], v + 1) for w in range(received.shape[1])], axis=1)
        if self.cfg.channel != "vocal":
            return one_hot(received, v + 1)
        silent = np.isnan(received)
        pitch = np.where(silent, 0.0, received)[:, None]
        centres, width = np.linspace(-1.0, 1.0, v)[None, :], 2.0 / (v - 1)
        tuning = np.exp(-((pitch - centres) ** 2) / (2 * width**2)) * ~silent[:, None]
        return np.concatenate([tuning, silent[:, None]], axis=1).astype(np.float32)

    def step(self, moves: np.ndarray) -> StepResult:
        cfg, n = self.cfg, self.n
        envs = np.arange(n)
        reward = np.full(n, -cfg.step_penalty, dtype=np.float32)
        success = np.zeros(n, dtype=bool)
        done = np.zeros(n, dtype=bool)
        picked = np.full((n, 2), -1, dtype=np.int64)
        for a in range(2):
            move = moves[:, a]
            new = self.pos[:, a] + DELTAS[move]
            inside = np.all((new >= 0) & (new < cfg.grid_size), axis=1)
            clipped = np.clip(new, 0, cfg.grid_size - 1)
            ok = inside & ~self.walls[envs, clipped[:, 0], clipped[:, 1]]
            self.pos[ok, a] = new[ok]

            interact = move == INTERACT
            at_item = np.all(self.item_pos == self.pos[:, a][:, None, :], axis=2) & self.item_present
            which = at_item.argmax(axis=1)
            pick = interact & (self.carrying[:, a] < 0) & at_item.any(axis=1)
            self.carrying[pick, a] = which[pick]
            self.item_present[pick, which[pick]] = False
            picked[pick, a] = which[pick]
            reward[pick] += cfg.pickup_bonus

            at_depot = np.all(self.pos[:, a] == self.depot, axis=1)
            deliver = interact & ~pick & (self.carrying[:, a] >= 0) & at_depot & ~done
            correct = deliver & (self.carrying[:, a] == self.target)
            reward[correct] += 1.0
            success |= correct
            done |= deliver  # a wrong delivery also ends the episode (failure)
            self.carrying[deliver, a] = -1
        self.t += 1
        done |= self.t >= cfg.max_steps
        return StepResult(reward=reward, done=done, success=success, picked=picked)

    def snapshot(self, i: int) -> dict:
        """JSON-serializable state of env i, for replay logs."""
        return {
            "t": int(self.t[i]),
            "pos": self.pos[i].tolist(),
            "carrying": self.carrying[i].tolist(),
            "item_present": self.item_present[i].tolist(),
        }
