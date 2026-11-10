"""Run a batch of fixed episodes to completion under one channel mode."""

from __future__ import annotations

from typing import Protocol

import numpy as np

from emcomm.channel import PERMUTED_WORDS, cue_mask, episode_permutations, initial_received, route
from emcomm.config import EnvConfig
from emcomm.env import STAY, GridEnv
from emcomm.layouts import RUNNER, SCOUT, EpisodeSpec


class Team(Protocol):
    def reset(self, i: int) -> None: ...

    def act(self, env: GridEnv, obs: np.ndarray, received: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return moves (n, 2) and sent symbols (n, 2[, words])."""
        ...


def run_episodes(
    cfg: EnvConfig,
    specs: list[EpisodeSpec],
    team: Team,
    condition: str,
    mode: str,
    seed: int,
    record: tuple[int, ...] = (),
) -> tuple[list[dict], dict[int, list[dict]]]:
    """Play every spec once in lockstep. Returns per-episode records and, for the
    episode indices in `record`, step traces: one frame per step (state before the
    step's actions) plus a terminal frame. Deterministic given (specs, team, seed)."""
    n = len(specs)
    env = GridEnv(cfg, n)
    for i, spec in enumerate(specs):
        env.load(i, spec)
        team.reset(i)
    rng = np.random.default_rng(seed)
    perm = (
        episode_permutations(n, cfg.vocab_size, seed + 1, cfg.msg_words, PERMUTED_WORDS[mode])
        if mode in PERMUTED_WORDS
        else None
    )
    mask = cue_mask(condition)
    vocal = cfg.channel == "vocal"
    received = initial_received(n, cfg.vocab_size, vocal, cfg.msg_words)
    active = np.ones(n, dtype=bool)
    ret = np.zeros(n, dtype=np.float64)
    steps = np.zeros(n, dtype=np.int64)
    success = np.zeros(n, dtype=bool)
    runner_pick = np.full(n, -1, dtype=np.int64)
    if vocal:
        scout_msgs = np.full((n, cfg.max_steps), np.nan)
    else:
        shape = (n, cfg.max_steps) if cfg.msg_words == 1 else (n, cfg.max_steps, cfg.msg_words)
        scout_msgs = np.full(shape, -1, dtype=np.int64)
    traces: dict[int, list[dict]] = {i: [] for i in record}
    for t in range(cfg.max_steps):
        obs = env.observe(received, mask)
        moves, sent = team.act(env, obs, received)
        moves = np.where(active[:, None], moves, STAY)
        scout_msgs[active, t] = sent[active, SCOUT]
        for i in record:
            if active[i]:
                traces[i].append(
                    {
                        **env.snapshot(i),
                        "moves": moves[i].tolist(),
                        "sent": sent[i].tolist(),
                        "received": received[i].tolist(),
                    }
                )
        result = env.step(moves)
        ret[active] += result.reward[active]
        steps[active] += 1
        success |= result.success & active
        first_pick = active & (runner_pick < 0) & (result.picked[:, RUNNER] >= 0)
        runner_pick[first_pick] = result.picked[first_pick, RUNNER]
        received = route(sent, mode, cfg.vocab_size, rng, perm, noise=cfg.vocal_noise)
        for i in record:
            if active[i] and result.done[i]:  # terminal frame: state after the last action
                traces[i].append({**env.snapshot(i), "moves": None, "sent": None, "received": None})
        active &= ~result.done
        if not active.any():
            break
    records = []
    for i in range(n):
        rec = {
            "episode": i,
            "layout_id": specs[i].layout.layout_id,
            "target": specs[i].target,
            "success": bool(success[i]),
            "return": float(ret[i]),
            "steps": int(steps[i]),
            "runner_opt": specs[i].runner_opt,
            "path_efficiency": float(specs[i].runner_opt / steps[i]) if success[i] else float("nan"),
            "runner_pick": int(runner_pick[i]),
            "scout_msgs": scout_msgs[i, : steps[i]].tolist(),
        }
        if cfg.task == "compositional":
            rec["target_attrs"] = env.item_attrs[i, specs[i].target].tolist()
            rec["picked_attrs"] = env.item_attrs[i, runner_pick[i]].tolist() if runner_pick[i] >= 0 else None
        records.append(rec)
    return records, traces
