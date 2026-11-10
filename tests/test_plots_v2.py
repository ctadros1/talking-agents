import csv

import numpy as np

from emcomm import plots, plots_v2
from emcomm.analysis_v2 import summarize
from emcomm.benchmark_v2 import ROW_FIELDS, RUN_LEVEL_FIELDS
from emcomm.sweep import run_dir

OTHER = ("mean_return", "steps_success", "path_efficiency", "mi_first_msg", "runner_pick_rate")


def _row(condition, seed, split, mode, success):
    return {
        "condition": condition,
        "seed": seed,
        "split": split,
        "intervention": mode,
        "n": 200.0,
        "success": success + seed / 100,
        **dict.fromkeys((*OTHER, *ROW_FIELDS, *RUN_LEVEL_FIELDS), 0.3),
    }


def _fake(tmp_path):
    runs = tmp_path / "runs"
    rows = [
        _row("scripted_color_only", -1, "held_out", "intact", 0.5),
        _row("scripted_no_comm", -1, "held_out", "silenced", 0.25),
    ]
    for seed in range(3):
        for split, level in (("held_out", 0.8), ("unseen", 0.6)):
            for mode, drop in (("intact", 0.0), ("permuted", 0.5), ("permuted_w1", 0.3), ("permuted_w2", 0.3)):
                rows.append(_row("comm", seed, split, mode, level - drop))
            rows.append(_row("no_comm", seed, split, "silenced", 0.25))
        rows += [
            _row("comm", seed, "held_out", "silenced", 0.25),
            _row("comm", seed, "train", "intact", 0.8),
            _row("no_comm", seed, "train", "silenced", 0.25),
        ]
        for condition in ("comm", "no_comm"):
            out = run_dir(runs, condition, seed)
            out.mkdir(parents=True)
            with open(out / "metrics.csv", "w", newline="") as fh:
                writer = csv.DictWriter(fh, fieldnames=["env_steps", "success_rate"])
                writer.writeheader()
                writer.writerows({"env_steps": 8192 * u, "success_rate": 0.1 * u} for u in range(1, 6))
    return runs, rows


def test_all_v2_figures_render(tmp_path):
    runs, rows = _fake(tmp_path)
    summary = summarize(rows)
    out = tmp_path / "figures"
    out.mkdir()
    plots_v2.fig_v2_success(summary, out)
    plots_v2.fig_v2_interventions(rows, out)
    plots_v2.fig_v2_errors(summary, out)
    tables = {f"word{w}_given_{p}": np.eye(5, 8).tolist() for w in (1, 2) for p in ("color", "shape")}
    plots_v2.fig_v2_lexicon(tables, seed=1, out=out)
    plots.fig_training_curves(runs, out, name="fig_v2_training_curves.png")
    assert {p.name for p in out.iterdir()} == {
        "fig_v2_success.png",
        "fig_v2_interventions.png",
        "fig_v2_errors.png",
        "fig_v2_lexicon.png",
        "fig_v2_training_curves.png",
    }
