import csv
import json

from emcomm import plots
from emcomm.analysis import load_rows, summarize
from emcomm.benchmark import RUN_FIELDS
from emcomm.replays import VARIANTS
from emcomm.sweep import run_dir


def _row(condition, seed, split, intervention, success):
    return {
        "condition": condition,
        "seed": seed,
        "split": split,
        "intervention": intervention,
        "n": 200,
        "success": success,
        "mean_return": success,
        "steps_success": 15.0,
        "path_efficiency": 0.8,
        "mi_first_msg": 1.5,
        "runner_pick_rate": 1.0,
    }


def _vocal_results(tmp_path):
    runs, results = tmp_path / "runs_vocal", tmp_path / "results"
    results.mkdir()
    rows = []
    for s in range(4):
        rows += [
            _row("comm", s, "held_out", "intact", 0.7 + s / 100),
            _row("comm", s, "held_out", "shuffled", 0.2),
            _row("comm", s, "held_out", "silenced", 0.25),
            _row("comm", s, "train", "intact", 0.72),
            _row("no_comm", s, "held_out", "silenced", 0.24),
            _row("no_comm", s, "train", "silenced", 0.25),
        ]
        out = run_dir(runs, "comm", s)
        out.mkdir(parents=True)
        with open(out / "eval.jsonl", "w") as fh:
            for k in range(40):
                pitch = (-0.75, -0.25, 0.25, 0.75)[k % 4]
                fh.write(
                    json.dumps({"split": "held_out", "intervention": "intact", "target": k % 4, "scout_msgs": [pitch]})
                    + "\n"
                )
    with open(results / "per_run.csv", "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=RUN_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    return runs, load_rows(results / "per_run.csv")


def test_vocal_summary_uses_the_shuffle_test(tmp_path):
    _, rows = _vocal_results(tmp_path)
    summary = summarize(rows)
    assert set(summary["comparisons"]) == {"comm_minus_no_comm", "intact_minus_shuffled", "intact_minus_silenced"}
    assert abs(summary["comparisons"]["intact_minus_shuffled"]["mean"] - 0.515) < 1e-9
    assert summary["criterion"] == {}  # exploratory arm: the pre-registered criterion does not apply


def test_vocal_figures_render(tmp_path):
    runs, rows = _vocal_results(tmp_path)
    out = tmp_path / "figures"
    out.mkdir()
    plots.fig_intervention(rows, out, modes=("intact", "shuffled", "silenced"))
    plots.fig_pitch(runs, rows, out, n_targets=4)
    assert {p.name for p in out.iterdir()} == {"fig_intervention.png", "fig_pitch.png"}


def test_replay_variants_per_channel():
    assert VARIANTS["discrete"] == (("comm", "intact"), ("comm", "permuted"), ("no_comm", "silenced"))
    assert VARIANTS["vocal"] == (("comm", "intact"), ("comm", "shuffled"), ("no_comm", "silenced"))
