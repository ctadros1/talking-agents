"""Authored (scripted) reference agents.

The scripted runner has PRIVILEGED map knowledge (it plans shortest paths on
the true map) but exactly the same target information as a learned runner: the
target only through a received message, or directly under the oracle cue. It is
a reference and headroom check, not a competing learned method.
"""

from __future__ import annotations

import numpy as np

from emcomm.combos import PROPERTIES
from emcomm.env import DELTAS, INTERACT, STAY, GridEnv
from emcomm.layouts import RUNNER, SCOUT, bfs_distances


def _step_toward(walls: np.ndarray, start: tuple[int, int], goal: tuple[int, int]) -> int:
    if start == goal:
        return INTERACT
    dist = bfs_distances(walls, goal)  # unreachable goal: no neighbour matches, so STAY
    for move in (1, 2, 3, 4):  # UP, DOWN, LEFT, RIGHT: fixed tie-break order
        r, c = start[0] + DELTAS[move][0], start[1] + DELTAS[move][1]
        if 0 <= r < walls.shape[0] and 0 <= c < walls.shape[1] and dist[r, c] == dist[start] - 1:
            return move
    return STAY


class ScriptedTeam:
    """Scout says the target's index every step; runner decodes it with the same fixed codebook."""

    def __init__(self, n_envs: int, runner_sees_cue: bool):
        self.runner_sees_cue = runner_sees_cue
        self.belief = np.full(n_envs, -1, dtype=np.int64)

    def reset(self, i: int) -> None:
        self.belief[i] = -1

    def act(self, env: GridEnv, obs: np.ndarray, received: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        n, k = env.n, env.cfg.n_item_types
        moves = np.full((n, 2), STAY, dtype=np.int64)
        sent = np.zeros((n, 2), dtype=np.int64)
        sent[:, SCOUT] = env.target
        heard = received[:, RUNNER]
        self.belief = np.where(heard < k, heard, self.belief)
        if self.runner_sees_cue:
            self.belief = env.target.copy()
        for i in range(n):
            walls, here = env.walls[i], tuple(int(v) for v in env.pos[i, RUNNER])
            if env.carrying[i, RUNNER] >= 0:
                goal = tuple(int(v) for v in env.depot[i])
            else:
                present = np.flatnonzero(env.item_present[i])
                if self.belief[i] >= 0 and env.item_present[i, self.belief[i]]:
                    choice = int(self.belief[i])
                else:  # no information: go for the nearest item (a 1/k guess)
                    dist = bfs_distances(walls, here)
                    choice = int(min(present, key=lambda j: (dist[tuple(env.item_pos[i, j])], j)))
                goal = tuple(int(v) for v in env.item_pos[i, choice])
            moves[i, RUNNER] = _step_toward(walls, here, goal)
        return moves, sent


class ScriptedWordsTeam:
    """Privileged-navigation reference for the compositional task.

    Word one states the target color and word two states its shape. The runner
    remembers each stated property it hears and takes the nearest matching item.
    """

    def __init__(self, n_envs: int, runner_sees_cue: bool, states: tuple[str, ...] = PROPERTIES):
        self.runner_sees_cue, self.states = runner_sees_cue, states
        self.known = np.full((n_envs, 2), -1, dtype=np.int64)

    def reset(self, i: int) -> None:
        self.known[i] = -1

    def act(self, env: GridEnv, obs: np.ndarray, received: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        cfg, n = env.cfg, env.n
        target = env.item_attrs[np.arange(n), env.target]
        moves = np.full((n, 2), STAY, dtype=np.int64)
        sent = np.zeros((n, 2, cfg.msg_words), dtype=np.int64)
        for p, name in enumerate(PROPERTIES):
            if name in self.states:
                sent[:, SCOUT, p] = target[:, p]
                heard = received[:, RUNNER, p]
                self.known[:, p] = np.where(heard < cfg.n_values, heard, self.known[:, p])
        if self.runner_sees_cue:
            self.known = target.copy()
        for i in range(n):
            walls, here = env.walls[i], tuple(int(v) for v in env.pos[i, RUNNER])
            if env.carrying[i, RUNNER] >= 0:
                goal = tuple(int(v) for v in env.depot[i])
            else:
                present = [int(j) for j in np.flatnonzero(env.item_present[i])]
                fits = [
                    j
                    for j in present
                    if all(self.known[i, p] < 0 or env.item_attrs[i, j, p] == self.known[i, p] for p in range(2))
                ]
                dist = bfs_distances(walls, here)
                choice = min(fits or present, key=lambda j: (dist[tuple(env.item_pos[i, j])], j))
                goal = tuple(int(v) for v in env.item_pos[i, choice])
            moves[i, RUNNER] = _step_toward(walls, here, goal)
        return moves, sent
