# Supplementary original-method comparison

This is the separate post-hoc experiment declared in
`SUPPLEMENTARY_ABLATION.md`. The declaration is retained unchanged. It does not
replace either frozen benchmark or turn the comparison into a prospective
confirmatory test.

The coursework snapshot contains the complete per-run table and statistical
summary under `results/ablation_unbiased/`. The summary embeds the original
launch record, experimental source/configuration hashes and all twenty final
checkpoint hashes. Full training checkpoints are retained in the author's
ignored run directory; the ZIP's four representative demo checkpoints belong
to the original v1 and v2 experiments, not this supplementary arm.

The saved table is enough to reproduce the reported seed-bootstrap estimates
without training or loading checkpoints. From the extracted project directory:

```bash
uv sync --frozen
uv run python - <<'PY'
import csv
import importlib.util
import json
from pathlib import Path

root = Path.cwd()
spec = importlib.util.spec_from_file_location("ablation", root / "scripts/analyze_ablation.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
with (root / "results/per_run.csv").open() as file:
    original = list(csv.DictReader(file))
with (root / "results/ablation_unbiased/per_run.csv").open() as file:
    control = list(csv.DictReader(file))
actual = module.summarize_comparison(original, control)
saved = json.loads((root / "results/ablation_unbiased/summary.json").read_text())
assert actual == {key: saved[key] for key in actual}
print(json.dumps(actual, indent=2))
print("All saved statistical estimates reproduced exactly.")
PY
```

For a computationally expensive new training run, use a separate extracted
copy and preserve the packaged result tables. The following commands write
new runs and evaluations to new directories:

```bash
caffeinate -i uv run python -m emcomm.sweep \
  --config configs/ablation_unbiased.toml \
  --root reproduction/ablation_unbiased \
  --seeds 0-19 --conditions comm --workers 6
uv run python scripts/check_sweep.py \
  --runs reproduction/ablation_unbiased --seeds 0-19 --conditions comm
uv run python -m emcomm.evaluate \
  --config configs/ablation_unbiased.toml \
  --runs reproduction/ablation_unbiased \
  --out reproduction/ablation_results --workers 4
```

On Linux or Windows, omit the macOS-only `caffeinate -i` prefix. Record the new
runtime and launch time separately; the embedded original launch record
describes the November 9 run and must not be relabeled as a new launch. To
analyze the newly produced table, use the same Python snippet with `control`
read from `reproduction/ablation_results/per_run.csv` and remove the comparison
with the historical saved summary. CPU determinism was checked within the
recorded software boundary; identical bytes across hardware are not promised.
