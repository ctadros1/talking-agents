# Talking Agents

**[Open the simulator →](https://ctad.dev/talking-agents/)**

Two independently learning agents must coordinate in a split-information grid world. The scout knows the requested item; the runner can reach and deliver it. Their symbol meanings are learned with a disclosed cue-directed signalling objective, never assigned in a dictionary.

The public site runs **v1 and v2 live**, with a 3D view for the compositional world, channel interventions, smoothly animated evidence charts and results with confidence intervals. Archived v1, v2 and vocal-channel episodes are also available. It uses no LLM, API key or external model service.

## Run the simulator locally

You only need Git and **Python 3.10+**. You do not need the scientific Python dependencies for the website.

```bash
git clone https://github.com/ctadros1/talking-agents.git
cd talking-agents
python3 scripts/serve.py --open
```

Open **http://localhost:8000/**. On Windows, `py -3 scripts/serve.py --open` also works. If the port is busy, add `--port 8001`. Stop with Ctrl+C. Do not open `web/index.html` directly: browser model loading requires the local server. After downloading the repository, the site runs without internet access.

- Choose **v1 · Four-item world** or **v2 · Color + shape · live 3D**. V2 opens by default. Press **Run episode** to execute the trained policies, or **Step** to inspect each decision.
- In v2, choose familiar or never-seen target combinations, and scramble both words or either word separately. Drag the 3D view to orbit; use the camera buttons to zoom/reset. With the canvas focused, arrow keys orbit, `+`/`-` zoom, and `0` resets the camera. Devices without WebGL fall back to a live 2D view.
- Choose **Learned messages**, **Scramble symbol meanings**, **Silence the trained communicators**, or the separately **trained no-message model**.
- Keep the scenario and action seed fixed to compare channel effects. Each condition resets recurrent memory and starts from the same geometry. Policy changes can change subsequent random sampling and trajectories.
- The map shows the full state to the human. Each agent receives only its own observation. The runner never receives the target cue or the other agent's position. A shaded square marks its local view.
- Message symbols arrive one step later. Symbol numbers do not encode a programmed meaning. A symbol's learned meaning is specific to its training run.
- The browser uses actual exported final weights: v1 training seed 19 and v2 training seed 9, with stochastic action sampling. V1 has 200 fixed held-out scenarios; v2 has 200 familiar and 200 never-seen-combination scenarios, both on held-out maps. Its JavaScript sampling and floating-point operations differ from PyTorch. It is an interactive illustration, **not a replacement for the canonical benchmark**.

The **Replay lab** contains saved trajectories. v2's 3D scene is a rendering of the grid world, not a 3D physics environment. Sound is opt-in in the replay viewers.

## Understand and check the results

Start with the site's **Results** section, the [research report](report/report.pdf), and the [illustrated learning guide (PDF)](docs/guide/how-it-works.pdf). The guide and report Markdown contain templates; the saved PDF is the rendered paper. The prose explains the environment, independent PPO, signalling objective, interventions and limitations.

Every displayed website statistic is loaded from the committed `results/*/summary.json` files. No measured outcome is hard-coded into the site. `results/*/per_run.csv` is the seed-level source table. The four groups are:

| Experiment | What it asks | Scientific interpretation |
|---|---|---|
| v1 (`results/`) | Do learned symbols improve coordination on held-out maps? | The frozen communication criterion passed. |
| v2 (`results/v2/`) | Do two-word protocols generalize to unseen color–shape combinations? | Communication claim A passed; generalization claim B failed; claim C was not reached. |
| Vocal (`results/vocal/`) | Does a continuous pitch channel support coordination? | Exploratory; its budget differs from v1. |
| Objective ablation (`results/ablation_unbiased/`) | Does removing only the cue-directed signalling objective hurt performance? | Full-budget, 20-seed comparison; explicitly post-hoc and exploratory. |

Each condition uses **20 independent training seeds**, and each seed is evaluated on **200 fixed episodes per split/intervention**. Seeds, not individual reused episodes, are the independent units for 95% bootstrap intervals. v1 and v2 each have 80 runs; vocal has 40; the supplementary objective ablation adds 20. Repeated evaluations are not thousands of independently sampled scenarios.

The no-communication baseline is independently trained. Silencing or permuting messages intervenes on already trained communicators and answers a different question. The oracle is a learned reference with target knowledge; scripted references have privileged map information. Neither should be presented as an equally informed competing method.

The signalling objective is an adaptation of prior positive-signalling work, not a new invention of symbolic communication or PPO. A separate support-aware audit identifies a shortcut in the unseen v2 target support; the theoretical identification rate is not a learned policy success rate. See [observation audit](docs/OBSERVATION_AUDIT.md) and [support audit](results/v2/support_reference.json). A null generalization result is retained honestly.

### Reproduce saved statistics without retraining

Install **[uv](https://docs.astral.sh/uv/getting-started/installation/)**. The experiment uses **Python 3.12** and the committed dependency lock.

```bash
uv sync --frozen
uv run python scripts/verify_results.py
uv run pytest -q
uv run ruff check .
uv run ruff format --check .
```

The verifier recomputes v1, v2, vocal and supplementary summaries from the saved seed tables without writing any files, and checks the frozen-file hashes. It does not overwrite results, retrain agents or use an API. Exact numerical reproduction is verified within the locked software boundary; cross-platform bit-for-bit training is not promised.

### Run fresh Python inference

Four representative **final** checkpoints are included under `demo_checkpoints/`: v1 communication/no-communication seed 19 and v2 communication/no-communication seed 9. These are demonstration models, not all benchmark checkpoints. v1 seed 19 and v2 seed 9 follow the existing replay representatives; they are not used to estimate the aggregate benchmark statistics.

```bash
uv run python scripts/demo_inference.py --phase both
```

This runs the canonical Python evaluation on the communicating representatives and exports fresh episode logs, metrics, execution receipts and replay pages into a new `demo_output/<timestamp>/` directory. It does not alter committed results. Open the exported `viewer/index.html` or `viewer/v2.html` after the run. Use `--phase v1` or `--phase v2` for one experiment. For standalone original replays, open `viewer/index.html`, `viewer/v2.html` or `viewer/vocal.html`; these work from disk.

### Retrain a benchmark (optional; substantial compute)

The saved statistics and browser demo require no training. For a full reproduction, write new outputs separately:

```bash
uv run python -m emcomm.sweep --config configs/base.toml --root reproduction/v1/runs --seeds 0-19 --workers 6
uv run python -m emcomm.evaluate --config configs/base.toml --runs reproduction/v1/runs --out reproduction/v1/results --workers 6
uv run python -m emcomm.analysis --results reproduction/v1/results
```

Use `configs/v2.toml` and `emcomm.analysis_v2` for v2. On macOS, put `caffeinate -i` before long commands to prevent sleep. The sweep resumes by skipping runs with a final checkpoint. Preserve the frozen configs, metrics and pre-registrations. Benchmark seeds and unseen splits are not for tuning; use separate pilot seeds and training splits for new research. [Supplementary reproduction instructions](docs/ABLATION_REPRODUCIBILITY.md).

## Website development and deployment

The website build uses only the Python standard library. To run the browser inference and Worker routing tests, install Node.js 22+:

```bash
npm test
python3 scripts/build_site.py
```

For Cloudflare deployment, `npm ci` installs the pinned Wrangler. Authenticate to your **own** account and edit `wrangler.jsonc` with your account and routes before deploying a fork. The original deployment uses only `ctad.dev/talking-agents` and `ctad.dev/talking-agents/*`, leaving other site routes separate.

```bash
npm ci
npx wrangler deploy --dry-run
npm run deploy
```

`cloudflare/worker.mjs` strips the publication prefix and serves only allowlisted static assets. No secrets or server-side model inference are used. `scripts/export_browser_models.py` regenerates browser weights and parity fixtures from the included final v1 and v2 checkpoints; run it with `uv run python` if changing the adapter. Tests compare GRU outputs, private observations, carrying features, transitions and two-word channel timing against independently produced PyTorch/Python fixtures.

Shared evidence bars and metric cards remain mounted when you switch tabs: widths and numbers animate from the currently displayed value, including rapid tab changes. Comparisons absent from a tab collapse in place. Reduced-motion preferences disable these animations. Icons use a small locally vendored [Lucide](https://lucide.dev/) SVG subset (ISC licensed); `scripts/vendor_icons.py` records its pinned source and package digest, and normal builds need no external icon service.

## Project map

| Path | Contents |
|---|---|
| `web/` | Browser UI, policy inference adapter and exported weights |
| `cloudflare/` | Isolated path routing for the public deployment |
| `src/emcomm/` | Canonical environment, policies, PPO and evaluation |
| `configs/` | Frozen experiments and separate pilot configs |
| `demo_checkpoints/` | Four representative final models and metadata |
| `results/` | Saved tables, summaries, diagnostics and figures |
| `viewer/` | Offline recorded episode explorers |
| `report/` | Research paper and typesetting sources |
| `tests/`, `public-tests/` | Scientific/engineering checks and browser/Worker parity tests |
| `docs/` | Protocols, audits, reproducibility notes and learning guide |

This is a clean public snapshot of the author's project; it intentionally has no private coursework administration or old development history. [Provenance](research-provenance.json) records the source commit and hashes of unchanged scientific files. Original frozen source/configurations were copied byte for byte; the browser adapter is additive. Historical source hashes in preregistration and result records refer to the original research repository. Public history begins with this snapshot and does not pretend to contain those historical tags. [Deviations and scope](docs/DEVIATIONS.md). [AI assistance disclosure](docs/AI_ASSISTANCE.md).

Code and author-owned documentation/models are MIT licensed; bundled third-party components keep their own notices. See [LICENSE](LICENSE).
