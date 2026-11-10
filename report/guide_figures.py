"""Figures for the learning guide (docs/guide/how-it-works.md), drawn from the real environment code.

Usage: uv run python report/guide_figures.py
Writes docs/guide/figures/map.png (a held-out episode) and docs/guide/figures/runner_view.png
(the 7x7 window the runner actually observes at the start of that episode).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Circle, Rectangle  # noqa: E402

from emcomm.benchmark import episode_sets  # noqa: E402
from emcomm.channel import cue_mask, initial_received  # noqa: E402
from emcomm.config import load_config  # noqa: E402
from emcomm.env import GridEnv  # noqa: E402
from emcomm.layouts import RUNNER, SCOUT  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "guide" / "figures"
ITEM_COLORS = ["#dc2626", "#ca8a04", "#0891b2", "#c026d3"]
NAMES = "ABCD"


def draw_map(spec, cfg, ax) -> None:
    g = cfg.grid_size
    for r in range(g):
        for c in range(g):
            if spec.layout.walls[r, c]:
                color = "#374151"
            elif c < cfg.wall_col:
                color = "#ede9fe"
            else:
                color = "#ffffff"
            ax.add_patch(Rectangle((c, r), 1, 1, facecolor=color, edgecolor="#d1d5db", lw=0.5))
    sr, sc = spec.agent_pos[RUNNER]
    rad = cfg.view_radius
    ax.add_patch(
        Rectangle(
            (sc - rad, sr - rad),
            2 * rad + 1,
            2 * rad + 1,
            facecolor="#2563eb",
            alpha=0.10,
            lw=1.5,
            edgecolor="#2563eb",
            linestyle="--",
        )
    )
    dr, dc = spec.depot
    ax.add_patch(Rectangle((dc + 0.1, dr + 0.1), 0.8, 0.8, facecolor="none", edgecolor="#059669", lw=2.5))
    for k, (r, c) in enumerate(spec.item_pos):
        ax.add_patch(
            Rectangle(
                (c + 0.18, r + 0.18),
                0.64,
                0.64,
                facecolor=ITEM_COLORS[k],
                edgecolor="black" if k == spec.target else "none",
                lw=2,
            )
        )
        ax.text(c + 0.5, r + 0.53, NAMES[k], ha="center", va="center", color="white", fontsize=9, weight="bold")
    for (r, c), color, name in ((spec.agent_pos[SCOUT], "#7c3aed", "S"), (spec.agent_pos[RUNNER], "#2563eb", "R")):
        ax.add_patch(Circle((c + 0.5, r + 0.5), 0.36, facecolor=color))
        ax.text(c + 0.5, r + 0.53, name, ha="center", va="center", color="white", fontsize=9, weight="bold")
    ax.text(cfg.wall_col / 2, -0.4, "control room", ha="center", fontsize=8, color="#5b21b6")
    ax.text((cfg.wall_col + 1 + g) / 2, -0.4, "warehouse", ha="center", fontsize=8, color="#374151")
    ax.set_xlim(0, g)
    ax.set_ylim(g, -0.9)
    ax.set_aspect("equal")
    ax.axis("off")


def draw_runner_view(env: GridEnv, cfg, ax) -> None:
    """Decode the runner's real observation vector back into a picture."""
    obs = env.observe(initial_received(1, cfg.vocab_size), cue_mask("comm"))[0, RUNNER]
    size, chans = cfg.view_size, cfg.n_view_channels
    view = obs[: size * size * chans].reshape(chans, size, size)
    for r in range(size):
        for c in range(size):
            color = "#374151" if view[0, r, c] else "#ffffff"
            ax.add_patch(Rectangle((c, r), 1, 1, facecolor=color, edgecolor="#d1d5db", lw=0.5))
            for k in range(cfg.n_item_types):
                if view[1 + k, r, c]:
                    ax.add_patch(Rectangle((c + 0.18, r + 0.18), 0.64, 0.64, facecolor=ITEM_COLORS[k]))
                    ax.text(
                        c + 0.5, r + 0.53, NAMES[k], ha="center", va="center", color="white", fontsize=9, weight="bold"
                    )
            if view[1 + cfg.n_item_types, r, c]:
                ax.add_patch(Rectangle((c + 0.1, r + 0.1), 0.8, 0.8, facecolor="none", edgecolor="#059669", lw=2.5))
    centre = cfg.view_radius
    ax.add_patch(Circle((centre + 0.5, centre + 0.5), 0.36, facecolor="#2563eb"))
    ax.text(centre + 0.5, centre + 0.53, "R", ha="center", va="center", color="white", fontsize=9, weight="bold")
    ax.set_xlim(0, size)
    ax.set_ylim(size, 0)
    ax.set_aspect("equal")
    ax.axis("off")


def main() -> None:
    cfg, _ = load_config(ROOT / "configs" / "base.toml")
    spec = episode_sets(cfg)["held_out"][0]
    env = GridEnv(cfg, 1)
    env.load(0, spec)
    OUT.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(4.6, 4.8))
    draw_map(spec, cfg, ax)
    fig.tight_layout()
    fig.savefig(OUT / "map.png", dpi=220)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(2.6, 2.6))
    draw_runner_view(env, cfg, ax)
    fig.tight_layout()
    fig.savefig(OUT / "runner_view.png", dpi=220)
    plt.close(fig)
    print(f"target {NAMES[spec.target]}; figures -> {OUT}")


if __name__ == "__main__":
    main()
