"""v2 figures from per-run statistics, learned dictionary, and training curves.

Usage: python -m emcomm.plots_v2 --runs runs_v2 --results results/v2
Run emcomm.lexicon first.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

from emcomm.analysis_v2 import load_rows, summarize  # noqa: E402
from emcomm.combos import COLOR_NAMES, SHAPE_NAMES  # noqa: E402
from emcomm.config import CONDITIONS  # noqa: E402
from emcomm.plots import COLORS, LABELS, fig_training_curves  # noqa: E402

SPLIT_LABELS = {"held_out": "Seen combinations", "unseen": "Never-seen combinations"}
MODES = ("intact", "permuted", "permuted_w1", "permuted_w2")
MODE_LABELS = ("Intact", "Both words\nscrambled", "Word 1\nscrambled", "Word 2\nscrambled")
ERRORS = (
    ("twin_color", "same color, wrong shape", "#2563eb"),
    ("twin_shape", "same shape, wrong color", "#d97706"),
    ("twin_opposite", "both wrong", "#6b7280"),
)


def fig_v2_success(summary: dict, out: Path) -> None:
    conds = [c for c in CONDITIONS if c in summary["conditions"]]
    x, width = np.arange(len(conds)), 0.38
    fig, ax = plt.subplots(figsize=(5.4, 3.8))
    for i, (split, hatch) in enumerate((("held_out", ""), ("unseen", "//"))):
        cis = [summary["conditions"][c][split]["success"] for c in conds]
        means = np.array([ci["mean"] for ci in cis]) * 100
        err = np.array([[ci["mean"] - ci["lo"], ci["hi"] - ci["mean"]] for ci in cis]).T * 100
        ax.bar(
            x + (i - 0.5) * width,
            means,
            width,
            yerr=err,
            capsize=3,
            color=[COLORS[c] for c in conds],
            hatch=hatch,
            edgecolor="white",
        )
    handles = [
        Patch(facecolor="#9ca3af", label=SPLIT_LABELS["held_out"]),
        Patch(facecolor="#9ca3af", hatch="//", edgecolor="white", label=SPLIT_LABELS["unseen"]),
    ]
    refs = summary["scripted"].get("held_out", {})
    for name, style, label in (
        ("scripted_color_only", "--", "one property known (scripted)"),
        ("scripted_no_comm", ":", "no information (scripted)"),
    ):
        if name in refs:
            handles.append(ax.axhline(refs[name] * 100, ls=style, lw=1, color="black", label=label))
    ax.set_xticks(x, [LABELS[c] for c in conds], fontsize=7)
    ax.set_ylabel("Success on held-out layouts (%)")
    ax.set_ylim(0, 105)
    fig.legend(handles=handles, fontsize=6.5, loc="upper center", bbox_to_anchor=(0.5, 1), ncol=2)
    fig.tight_layout(rect=(0, 0, 1, 0.85))
    fig.savefig(out / "fig_v2_success.png", dpi=200)
    plt.close(fig)


def fig_v2_interventions(rows: list[dict], out: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(6.4, 2.9), sharey=True)
    for ax, split in zip(axes, SPLIT_LABELS):
        per_seed: dict[int, dict[str, float]] = {}
        for r in rows:
            if r["condition"] == "comm" and r["split"] == split and r["intervention"] in MODES:
                per_seed.setdefault(r["seed"], {})[r["intervention"]] = r["success"] * 100
        for values in per_seed.values():
            ax.plot(
                range(len(MODES)),
                [values.get(m, np.nan) for m in MODES],
                color=COLORS["comm"],
                alpha=0.3,
                lw=0.8,
                marker="o",
                ms=2,
            )
        means = [np.mean([v[m] for v in per_seed.values() if m in v] or [np.nan]) for m in MODES]
        ax.plot(range(len(MODES)), means, color="black", lw=2, marker="o", label="mean over seeds")
        ax.set_xticks(range(len(MODES)), MODE_LABELS, fontsize=6.5)
        ax.set_title(SPLIT_LABELS[split], fontsize=8)
        ax.set_ylim(0, 105)
    axes[0].set_ylabel("Success (%)")
    axes[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out / "fig_v2_interventions.png", dpi=200)
    plt.close(fig)


def fig_v2_errors(summary: dict, out: Path) -> None:
    interventions = summary["conditions"]["comm"]["interventions"]
    fig, axes = plt.subplots(1, 2, figsize=(6.4, 3.4), sharey=True)
    for ax, split in zip(axes, SPLIT_LABELS):
        modes = [m for m in MODES if m in interventions[split]]
        bottom = np.zeros(len(modes))
        for key, label, color in ERRORS:
            values = np.array([interventions[split][m][key]["mean"] or 0.0 for m in modes]) * 100
            ax.bar(range(len(modes)), values, bottom=bottom, color=color, label=label)
            bottom += values
        ax.set_xticks(range(len(modes)), [MODE_LABELS[MODES.index(m)] for m in modes], fontsize=6.5)
        ax.set_title(SPLIT_LABELS[split], fontsize=8)
    axes[0].set_ylabel("Share of wrong picks (%)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, fontsize=6.5, loc="upper center", bbox_to_anchor=(0.5, 1), ncol=3)
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    fig.savefig(out / "fig_v2_errors.png", dpi=200)
    plt.close(fig)


def fig_v2_lexicon(tables: dict, seed: int, out: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(6.4, 4.4))
    for row, word in enumerate(("word1", "word2")):
        for col, (prop, names) in enumerate((("color", COLOR_NAMES), ("shape", SHAPE_NAMES))):
            ax, probs = axes[row, col], np.array(tables[f"{word}_given_{prop}"])
            ax.imshow(probs, cmap="Blues", vmin=0, vmax=1)
            ax.set_xticks(range(probs.shape[1]))
            ax.set_yticks(range(len(names)), names, fontsize=7)
            ax.tick_params(axis="x", labelsize=7)
            ax.set_title(f"P(word {row + 1} = symbol | {prop})", fontsize=8)
    for ax in axes[1]:
        ax.set_xlabel("Symbol", fontsize=7)
    fig.suptitle(f"Learned dictionary of comm seed {seed} (scout's first message)", fontsize=9)
    fig.tight_layout()
    fig.savefig(out / "fig_v2_lexicon.png", dpi=200)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", default="runs_v2")
    parser.add_argument("--results", default="results/v2")
    args = parser.parse_args()
    runs, results = Path(args.runs), Path(args.results)
    figures = results / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    rows = load_rows(results / "per_run.csv")
    summary = summarize(rows)
    fig_v2_success(summary, figures)
    fig_training_curves(runs, figures, name="fig_v2_training_curves.png")
    fig_v2_interventions(rows, figures)
    fig_v2_errors(summary, figures)
    lexicon = json.loads((results / "lexicon.json").read_text())
    seed = lexicon["representative_seed"]
    fig_v2_lexicon(lexicon["seeds"][str(seed)], seed, figures)
    print(f"figures -> {figures}")


if __name__ == "__main__":
    main()
