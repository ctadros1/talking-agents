import csv
import json

from emcomm import plots
from emcomm.analysis import load_rows, summarize
from emcomm.benchmark import RUN_FIELDS
from emcomm.sweep import run_dir


def _fake_results(tmp_path):
    runs, results = tmp_path / "runs", tmp_path / "results"
    results.mkdir()
    rows = []
    for seed in range(3):
        for cond, mode, success in (("comm", "intact", 0.8), ("no_comm", "silenced", 0.25)):
            rows.append(
                {
                    "condition": cond,
                    "seed": seed,
                    "split": "held_out",
                    "intervention": mode,
                    "n": 200,
                    "success": success + seed / 100,
                    "mean_return": 0.5,
                    "steps_success": 15,
                    "path_efficiency": 0.8,
                    "mi_first_msg": 1.2,
                    "runner_pick_rate": 1.0,
                }
            )
            out = run_dir(runs, cond, seed)
            out.mkdir(parents=True)
            with open(out / "metrics.csv", "w", newline="") as fh:
                writer = csv.DictWriter(fh, fieldnames=["env_steps", "success_rate"])
                writer.writeheader()
                writer.writerows({"env_steps": 8192 * u, "success_rate": success * u / 5} for u in range(1, 6))
        rows.append({**rows[-2], "intervention": "permuted", "success": 0.3})
        with open(run_dir(runs, "comm", seed) / "eval.jsonl", "w") as fh:
            for k in range(8):
                fh.write(
                    json.dumps({"split": "held_out", "intervention": "intact", "target": k % 4, "scout_msgs": [k % 4]})
                    + "\n"
                )
    with open(results / "per_run.csv", "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=RUN_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    return runs, results


def test_all_figures_render(tmp_path):
    runs, results = _fake_results(tmp_path)
    rows = load_rows(results / "per_run.csv")
    out = results / "figures"
    out.mkdir()
    plots.fig_success(summarize(rows), out)
    plots.fig_training_curves(runs, out)
    plots.fig_intervention(rows, out)
    plots.fig_protocol(runs, rows, out, vocab_size=8, n_targets=4)
    names = {p.name for p in out.iterdir()}
    assert names == {"fig_success.png", "fig_training_curves.png", "fig_intervention.png", "fig_protocol.png"}
    assert plots.representative_seed(rows) == 1  # median of 0.80, 0.81, 0.82
