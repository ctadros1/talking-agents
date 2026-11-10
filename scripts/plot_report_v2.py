"""Report-only redraw: scripted references belong exclusively to the seen split."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def main():
    summary = json.loads((ROOT / "results/v2/summary.json").read_text())
    conditions = ("comm", "no_comm", "random_msg", "oracle")
    colors = ("#2563eb", "#6b7280", "#d97706", "#059669")
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.7), sharey=True)
    for ax, split, title in zip(axes, ("held_out", "unseen"), ("Seen combinations", "Never-seen combinations")):
        cis = [summary["conditions"][c][split]["success"] for c in conditions]
        means = np.array([v["mean"] for v in cis]) * 100
        errors = np.array([[v["mean"] - v["lo"], v["hi"] - v["mean"]] for v in cis]).T * 100
        ax.bar(range(4), means, yerr=errors, color=colors, capsize=3)
        ax.set_xticks(range(4), ["Comm", "No comm", "Random", "Oracle"], rotation=20, fontsize=10)
        ax.set_title(title, fontsize=12)
        ax.set_ylim(0, 105)
        ax.spines[["top", "right"]].set_visible(False)
    refs = summary["scripted"]["held_out"]
    for key, style, label in (
        ("scripted_color_only", "--", "One property, seen only"),
        ("scripted_no_comm", ":", "No information, seen only"),
    ):
        axes[0].axhline(refs[key] * 100, ls=style, lw=1.3, color="black", label=label)
    axes[0].legend(fontsize=8, loc="upper left")
    axes[0].set_ylabel("Success on held-out layouts (%)", fontsize=11)
    fig.tight_layout()
    out = ROOT / "report/figures/fig_v2_success_scoped.png"
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, dpi=230)
    plt.close(fig)
    print(out)


if __name__ == "__main__":
    main()
