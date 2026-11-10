"""Report figures from results/per_run.csv, runs/*/metrics.csv and runs/comm/*/eval.jsonl.

Usage: python -m emcomm.plots --runs runs --results results
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from emcomm.analysis import load_rows, summarize  # noqa: E402
from emcomm.channel import TRAIN_CHANNEL  # noqa: E402
from emcomm.config import CONDITIONS  # noqa: E402

COLORS = {"comm": "#2563eb", "no_comm": "#6b7280", "random_msg": "#d97706", "oracle": "#059669"}
LABELS = {"comm": "Learned comm", "no_comm": "No comm", "random_msg": "Random msgs", "oracle": "Oracle (full info)"}


def fig_success(summary: dict, out: Path) -> None:
    conds = [c for c in CONDITIONS if c in summary["conditions"]]
    means = [summary["conditions"][c]["success"]["mean"] * 100 for c in conds]
    lo = [m - summary["conditions"][c]["success"]["lo"] * 100 for m, c in zip(means, conds)]
    hi = [summary["conditions"][c]["success"]["hi"] * 100 - m for m, c in zip(means, conds)]
    fig, ax = plt.subplots(figsize=(4.2, 3.0))
    ax.bar([LABELS[c] for c in conds], means, yerr=[lo, hi], capsize=4, color=[COLORS[c] for c in conds])
    for name, style in (("scripted_comm", "--"), ("scripted_no_comm", ":")):
        if name in summary["scripted"]:
            ax.axhline(summary["scripted"][name] * 100, ls=style, lw=1, color="black", label=name.replace("_", " "))
    ax.set_ylabel("Held-out success (%)")
    ax.set_ylim(0, 105)
    if summary["scripted"]:
        ax.legend(fontsize=7, loc="upper left")
    ax.tick_params(axis="x", labelsize=7)
    fig.tight_layout()
    fig.savefig(out / "fig_success.png", dpi=200)
    plt.close(fig)


def fig_training_curves(runs: Path, out: Path, name: str = "fig_training_curves.png") -> None:
    fig, ax = plt.subplots(figsize=(4.2, 3.0))
    for c in CONDITIONS:
        curves = []
        for path in sorted(runs.glob(f"{c}/seed_*/metrics.csv")):
            with open(path) as fh:
                rows = list(csv.DictReader(fh))
            curves.append([(float(r["env_steps"]), float(r["success_rate"])) for r in rows])
        if not curves:
            continue
        length = min(len(cv) for cv in curves)
        steps = np.array([p[0] for p in curves[0][:length]]) / 1e6
        values = np.array([[p[1] for p in cv[:length]] for cv in curves]) * 100
        mean = values.mean(axis=0)
        sem = values.std(axis=0, ddof=1) / np.sqrt(len(curves)) if len(curves) > 1 else np.zeros_like(mean)
        ax.plot(steps, mean, color=COLORS[c], label=f"{LABELS[c]} (n={len(curves)})", lw=1.2)
        ax.fill_between(steps, mean - 1.96 * sem, mean + 1.96 * sem, color=COLORS[c], alpha=0.2, lw=0)
    ax.set_xlabel("Environment steps (millions)")
    ax.set_ylabel("Training success (%)")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out / name, dpi=200)
    plt.close(fig)


def fig_intervention(rows: list[dict], out: Path, modes: tuple[str, ...] = ("intact", "permuted", "silenced")) -> None:
    per_seed: dict[int, dict[str, float]] = {}
    for r in rows:
        if r["condition"] == "comm" and r["split"] == "held_out":
            per_seed.setdefault(r["seed"], {})[r["intervention"]] = r["success"] * 100
    fig, ax = plt.subplots(figsize=(4.2, 3.0))
    for values in per_seed.values():
        ax.plot(
            range(len(modes)),
            [values.get(m, np.nan) for m in modes],
            color=COLORS["comm"],
            alpha=0.35,
            lw=0.8,
            marker="o",
            ms=2,
        )
    means = [np.mean(vals) if (vals := [v[m] for v in per_seed.values() if m in v]) else np.nan for m in modes]
    ax.plot(range(len(modes)), means, color="black", lw=2, marker="o", label="mean over seeds")
    ax.set_xticks(range(len(modes)), [m.capitalize() for m in modes])
    ax.set_ylabel("Held-out success (%)")
    ax.set_ylim(0, 105)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out / "fig_intervention.png", dpi=200)
    plt.close(fig)


def representative_seed(rows: list[dict], condition: str = "comm") -> int:
    """Seed whose intact held-out success is the median of its condition (lower median on ties)."""
    ranked = sorted(
        (r["success"], r["seed"])
        for r in rows
        if r["condition"] == condition and r["split"] == "held_out" and r["intervention"] == TRAIN_CHANNEL[condition]
    )
    return ranked[(len(ranked) - 1) // 2][1]


def fig_protocol(runs: Path, rows: list[dict], out: Path, vocab_size: int, n_targets: int) -> None:
    seed = representative_seed(rows)
    counts = np.zeros((n_targets, vocab_size))
    with open(runs / "comm" / f"seed_{seed:04d}" / "eval.jsonl") as fh:
        for line in fh:
            r = json.loads(line)
            if r["split"] == "held_out" and r["intervention"] == "intact":
                counts[r["target"], r["scout_msgs"][0]] += 1
    probs = counts / counts.sum(axis=1, keepdims=True)
    fig, ax = plt.subplots(figsize=(4.2, 2.4))
    im = ax.imshow(probs, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(vocab_size))
    ax.set_yticks(range(n_targets), [f"target {'ABCD'[k]}" for k in range(n_targets)])
    ax.set_xlabel("Scout's first symbol")
    ax.set_title(f"P(symbol | target), comm seed {seed}", fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.03)
    fig.tight_layout()
    fig.savefig(out / "fig_protocol.png", dpi=200)
    plt.close(fig)


def fig_pitch(runs: Path, rows: list[dict], out: Path, n_targets: int) -> None:
    """Vocal channel: the scout's first pitch per target. Separate peaks = distinct learned "notes"."""
    seed = representative_seed(rows)
    pitches: dict[int, list[float]] = {k: [] for k in range(n_targets)}
    with open(runs / "comm" / f"seed_{seed:04d}" / "eval.jsonl") as fh:
        for line in fh:
            r = json.loads(line)
            if r["split"] == "held_out" and r["intervention"] == "intact":
                pitches[r["target"]].append(r["scout_msgs"][0])
    fig, ax = plt.subplots(figsize=(4.2, 2.6))
    bins = np.linspace(-1, 1, 41)
    for k, values in pitches.items():
        ax.hist(values, bins=bins, alpha=0.6, label=f"target {'ABCD'[k]}")
    ax.set_xlabel("Scout's first pitch")
    ax.set_ylabel("Episodes")
    ax.set_title(f"Learned pitch per target, vocal seed {seed}", fontsize=8)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out / "fig_pitch.png", dpi=200)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", default="runs")
    parser.add_argument("--results", default="results")
    parser.add_argument("--vocab-size", type=int, default=8)
    parser.add_argument("--n-targets", type=int, default=4)
    args = parser.parse_args()
    runs, results = Path(args.runs), Path(args.results)
    figures = results / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    rows = load_rows(results / "per_run.csv")
    fig_success(summarize(rows), figures)
    fig_training_curves(runs, figures)
    if any(r["intervention"] == "shuffled" for r in rows):  # vocal channel
        fig_intervention(rows, figures, modes=("intact", "shuffled", "silenced"))
        fig_pitch(runs, rows, figures, args.n_targets)
    else:
        fig_intervention(rows, figures)
        fig_protocol(runs, rows, figures, args.vocab_size, args.n_targets)
    print(f"figures -> {figures}")


if __name__ == "__main__":
    main()
