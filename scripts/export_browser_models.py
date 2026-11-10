"""Export representative final policies and independent Python parity fixtures.

Exports both v1 and v2 to web/models, web/data/scenarios*.json and web/fixtures/parity*.json.
It never trains, selects checkpoints or changes the frozen evaluation outputs.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from emcomm.benchmark import episode_sets
from emcomm.benchmark_v2 import episode_sets as episode_sets_v2
from emcomm.env import GridEnv
from emcomm.train import load_checkpoint

ROOT = Path(__file__).resolve().parents[1]


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, separators=(",", ":")) + "\n")


def serialize(spec):
    result = {
        "layout_id": int(spec.layout.layout_id),
        "walls": spec.layout.walls.tolist(),
        "item_pos": spec.item_pos.tolist(),
        "depot": list(spec.depot),
        "agent_pos": np.asarray(spec.agent_pos).tolist(),
        "target": int(spec.target),
        "runner_opt": int(spec.runner_opt),
    }
    if spec.item_attrs is not None:
        result["item_attrs"] = spec.item_attrs.tolist()
    return result


def export_phase(phase):
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    rng = np.random.default_rng(20261109 if phase == "v1" else 20261110)
    cases = {"policies": [], "worlds": []}
    for condition in ("comm", "no_comm"):
        seed = 19 if phase == "v1" else 9
        key = condition if phase == "v1" else f"v2-{condition}"
        checkpoint = ROOT / "demo_checkpoints" / phase / condition / f"seed_{seed:04d}/final.pt"
        nets, cfg, ppo, saved = load_checkpoint(checkpoint)
        splits = (episode_sets if phase == "v1" else episode_sets_v2)(cfg)
        specs = splits["held_out"]
        env = GridEnv(cfg, 1)
        tensors, chunks, offset = [], [], 0
        for agent, net in enumerate(nets):
            manifest = {}
            for name, tensor in net.state_dict().items():
                values = tensor.detach().numpy().astype("<f4")
                manifest[name] = {"offset": offset, "shape": list(values.shape)}
                chunks.append(values.tobytes())
                offset += values.size
            tensors.append(manifest)
            for step in (0, 9, 28):
                env.load(0, specs[step])
                env.t[0] = step
                heard = rng.integers(0, cfg.vocab_size + 1, (1, 2) if phase == "v1" else (1, 2, 2))
                obs = env.observe(heard, np.array([True, False]))[0, agent]
                hidden = rng.normal(0, 0.3, ppo.hidden).astype(np.float32)
                with torch.no_grad():
                    x = net.encoder(torch.from_numpy(obs)[None])
                    h = net.gru(x, torch.from_numpy(hidden)[None])
                    expected = {
                        "move": net.move_head(h)[0].tolist(),
                        "message": net.msg_head(h)[0].tolist(),
                        "hidden": h[0].tolist(),
                        "value": float(net.value_head(h)[0, 0]),
                    }
                cases["policies"].append(
                    {
                        "condition": key,
                        "agent": agent,
                        "obs": obs.tolist(),
                        "hidden": hidden.tolist(),
                        "expected": expected,
                    }
                )
        binary = b"".join(chunks)
        (ROOT / "web/models" / f"{key}.bin").write_bytes(binary)
        write_json(
            ROOT / "web/models" / f"{key}.json",
            {
                "condition": condition,
                "seed": int(saved["seed"]),
                "hidden": ppo.hidden,
                "config": asdict(cfg),
                "agents": tensors,
                "format": "little-endian float32, row-major; offsets in float elements",
                "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                "binary_sha256": hashlib.sha256(binary).hexdigest(),
            },
        )
    bank = (
        [serialize(s) for s in specs]
        if phase == "v1"
        else {split: [serialize(s) for s in rows] for split, rows in splits.items() if split != "train"}
    )
    write_json(ROOT / "web/data" / ("scenarios.json" if phase == "v1" else "scenarios-v2.json"), bank)
    for i in range(28):
        spec = specs[i]
        env.load(0, spec)
        heard = rng.integers(0, 9, (1, 2) if phase == "v1" else (1, 2, 2))
        moves = rng.integers(0, 6, (1, 2))
        state = None
        if i >= 20:
            a = i % 2
            if i < 23:  # successful delivery, wrong delivery, second-agent delivery
                env.pos[0, a] = env.depot[0]
                env.carrying[0, a] = (spec.target + (i == 21)) % 4
                env.item_present[0, env.carrying[0, a]] = False
            elif i < 26:  # pickup, including same-square competition
                env.pos[0, a] = env.item_pos[0, 0]
                if i == 25:
                    env.pos[0, 1 - a] = env.item_pos[0, 0]
                    moves[0, 1 - a] = 5
            else:  # timeout and boundary move
                env.t[0] = 31
                env.pos[0, a] = [0, 0]
            moves[0, a] = 5 if i < 26 else 1
            state = {
                "pos": env.pos[0].tolist(),
                "carrying": env.carrying[0].tolist(),
                "item_present": env.item_present[0].tolist(),
                "t": int(env.t[0]),
            }
        obs = env.observe(heard, np.array([True, False]))[0].tolist()
        result = env.step(moves)
        cases["worlds"].append(
            {
                "spec": serialize(spec),
                "state": state,
                "heard": heard[0].tolist(),
                "obs": obs,
                "moves": moves[0].tolist(),
                "expected": {
                    "snapshot": env.snapshot(0),
                    "reward": float(result.reward[0]),
                    "done": bool(result.done[0]),
                    "success": bool(result.success[0]),
                },
            }
        )
    filename = "parity.json" if phase == "v1" else "parity-v2.json"
    write_json(ROOT / "web/fixtures" / filename, cases)
    print(f"Exported {phase}: two final policy pairs and independent Python parity fixtures.")


def main():
    export_phase("v1")
    export_phase("v2")


if __name__ == "__main__":
    main()
