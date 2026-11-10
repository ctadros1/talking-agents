"""Train one (condition, seed) run with independent recurrent PPO.

Usage: python -m emcomm.train --config configs/base.toml --condition comm --seed 0 --out runs/comm/seed_00
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from emcomm.channel import TRAIN_CHANNEL, cue_mask, initial_received, route
from emcomm.config import CONDITIONS, EnvConfig, PPOConfig, load_config
from emcomm.env import GridEnv
from emcomm.layouts import EpisodeSampler, make_layout_splits
from emcomm.policy import AgentNet
from emcomm.ppo import Rollout, compute_gae, ppo_update

METRIC_FIELDS = [
    "update",
    "env_steps",
    "episodes",
    "success_rate",
    "mean_return",
    "mean_length",
    *(
        f"{k}_{a}"
        for a in (0, 1)
        for k in ("policy_loss", "value_loss", "entropy_move", "entropy_msg", "msg_signal", "approx_kl")
    ),
    "sps",
]


def set_determinism(seed: int) -> None:
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)


def make_nets(env_cfg: EnvConfig, ppo_cfg: PPOConfig) -> list[AgentNet]:
    vocal = env_cfg.channel == "vocal"
    return [
        AgentNet(env_cfg.obs_dim, env_cfg.n_moves, env_cfg.vocab_size, ppo_cfg.hidden, vocal, env_cfg.msg_words)
        for _ in range(2)
    ]


def save_checkpoint(
    path: Path, nets: list[AgentNet], env_cfg: EnvConfig, ppo_cfg: PPOConfig, condition: str, seed: int, update: int
) -> None:
    tmp = path.with_suffix(".tmp")
    torch.save(
        {
            "condition": condition,
            "seed": seed,
            "update": update,
            "env_steps": update * ppo_cfg.steps_per_update,
            "env_cfg": asdict(env_cfg),
            "ppo_cfg": asdict(ppo_cfg),
            "nets": [net.state_dict() for net in nets],
        },
        tmp,
    )
    tmp.replace(path)  # atomic: a crash never leaves a half-written final.pt


def load_checkpoint(path: str | Path) -> tuple[list[AgentNet], EnvConfig, PPOConfig, dict]:
    ckpt = torch.load(path, weights_only=True)
    env_cfg, ppo_cfg = EnvConfig(**ckpt["env_cfg"]), PPOConfig(**ckpt["ppo_cfg"])
    nets = make_nets(env_cfg, ppo_cfg)
    for net, state in zip(nets, ckpt["nets"]):
        net.load_state_dict(state)
        net.eval()
    return nets, env_cfg, ppo_cfg, ckpt


def train(env_cfg: EnvConfig, ppo_cfg: PPOConfig, condition: str, seed: int, out_dir: str | Path) -> Path:
    if condition not in CONDITIONS:
        raise ValueError(f"unknown condition {condition!r}")
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "run.json").write_text(
        json.dumps(
            {"condition": condition, "seed": seed, "env_cfg": asdict(env_cfg), "ppo_cfg": asdict(ppo_cfg)}, indent=2
        )
    )
    set_determinism(seed)
    n, steps, vocab = ppo_cfg.n_envs, ppo_cfg.rollout_len, env_cfg.vocab_size
    train_layouts, _ = make_layout_splits(env_cfg)
    sampler = EpisodeSampler(train_layouts, env_cfg, seed)
    env = GridEnv(env_cfg, n)
    for i in range(n):
        env.load(i, sampler.sample())
    nets = make_nets(env_cfg, ppo_cfg)
    opts = [torch.optim.Adam(net.parameters(), lr=ppo_cfg.lr, eps=1e-5) for net in nets]
    mode, mask = TRAIN_CHANNEL[condition], cue_mask(condition)
    channel_rng = np.random.default_rng(seed + 1)
    vocal = env_cfg.channel == "vocal"
    received = initial_received(n, vocab, vocal, env_cfg.msg_words)
    msg_shape = (n, 2) if env_cfg.msg_words == 1 else (n, 2, env_cfg.msg_words)
    joint_values = env_cfg.n_values if env_cfg.task == "compositional" else 0
    silence = np.nan if vocal else vocab
    first = torch.ones(n, dtype=torch.bool)
    h = [net.initial_state(n) for net in nets]
    ep_return, ep_len = np.zeros(n), np.zeros(n, dtype=np.int64)

    with open(out / "metrics.csv", "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=METRIC_FIELDS)
        writer.writeheader()
        for update in range(1, ppo_cfg.n_updates + 1):
            started = time.perf_counter()
            h0 = [x.clone() for x in h]
            buf = [{"obs": [], "first": [], "moves": [], "msgs": [], "logp": [], "values": []} for _ in range(2)]
            rewards, dones, finished = [], [], []
            for _ in range(steps):
                obs = torch.from_numpy(env.observe(received, mask))
                moves = np.zeros((n, 2), dtype=np.int64)
                sent = np.zeros(msg_shape, dtype=np.float64 if vocal else np.int64)
                with torch.no_grad():
                    for a in range(2):
                        move_logits, msg_logits, value, h[a] = nets[a].forward_sequence(
                            obs[None, :, a], h[a], first[None]
                        )
                        move, msg, logp = nets[a].sample(move_logits[0], msg_logits[0])
                        for key, val in (
                            ("obs", obs[:, a]),
                            ("first", first),
                            ("moves", move),
                            ("msgs", msg),
                            ("logp", logp),
                            ("values", value[0]),
                        ):
                            buf[a][key].append(val)
                        moves[:, a], sent[:, a] = (
                            move.numpy(),
                            np.clip(msg.numpy(), -1.0, 1.0) if vocal else msg.numpy(),
                        )
                result = env.step(moves)
                received = route(sent, mode, vocab, channel_rng, noise=env_cfg.vocal_noise)
                ep_return += result.reward
                ep_len += 1
                for i in np.flatnonzero(result.done):
                    finished.append((bool(result.success[i]), float(ep_return[i]), int(ep_len[i])))
                    ep_return[i], ep_len[i] = 0.0, 0
                    env.load(i, sampler.sample())
                    received[i] = silence
                rewards.append(torch.from_numpy(result.reward))
                dones.append(torch.from_numpy(result.done.astype(np.float32)))
                first = torch.from_numpy(result.done.copy())

            obs = torch.from_numpy(env.observe(received, mask))
            row: dict[str, float] = {"update": update, "env_steps": update * ppo_cfg.steps_per_update}
            for a in range(2):
                with torch.no_grad():
                    last_value = nets[a].forward_sequence(obs[None, :, a], h[a], first[None])[2][0]
                ro = Rollout(
                    obs=torch.stack(buf[a]["obs"]),
                    first=torch.stack(buf[a]["first"]),
                    moves=torch.stack(buf[a]["moves"]),
                    msgs=torch.stack(buf[a]["msgs"]),
                    logp=torch.stack(buf[a]["logp"]),
                    values=torch.stack(buf[a]["values"]),
                    rewards=torch.stack(rewards),
                    dones=torch.stack(dones),
                    h0=h0[a],
                )
                adv, returns = compute_gae(
                    ro.rewards, ro.values, ro.dones, last_value, ppo_cfg.gamma, ppo_cfg.gae_lambda
                )
                for key, val in ppo_update(
                    nets[a], opts[a], ro, adv, returns, ppo_cfg, env_cfg.cue_slice, env_cfg.vocal_noise, joint_values
                ).items():
                    row[f"{key}_{a}"] = val
            row["episodes"] = len(finished)
            row["success_rate"] = float(np.mean([f[0] for f in finished])) if finished else float("nan")
            row["mean_return"] = float(np.mean([f[1] for f in finished])) if finished else float("nan")
            row["mean_length"] = float(np.mean([f[2] for f in finished])) if finished else float("nan")
            row["sps"] = ppo_cfg.steps_per_update / (time.perf_counter() - started)
            writer.writerow(row)
            fh.flush()
            if update % ppo_cfg.checkpoint_every == 0:
                save_checkpoint(out / f"update_{update:05d}.pt", nets, env_cfg, ppo_cfg, condition, seed, update)
    final = out / "final.pt"
    save_checkpoint(final, nets, env_cfg, ppo_cfg, condition, seed, ppo_cfg.n_updates)
    return final


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--condition", required=True, choices=CONDITIONS)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--total-env-steps", type=int, default=None, help="override the config budget (pilots only)")
    args = parser.parse_args()
    env_cfg, ppo_cfg = load_config(args.config)
    if args.total_env_steps is not None:
        ppo_cfg = PPOConfig(**{**asdict(ppo_cfg), "total_env_steps": args.total_env_steps})
    print(train(env_cfg, ppo_cfg, args.condition, args.seed, args.out))


if __name__ == "__main__":
    main()
