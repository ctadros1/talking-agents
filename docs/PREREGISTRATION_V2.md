# Pre-registration: v2 (compositional communication)

Frozen at Gate V2-G2 on 2026-10-06 (git tag `prereg-v2`). Any later change goes through `docs/DEVIATIONS.md` under its v2 heading.

- **Spec:** `docs/superpowers/specs/2026-10-03-v2-compositional-communication-design.md` at commit `96941ead2c3a7c685c964f41a9ffc2bcfc67726b`. Sections 1–9 are the pre-registered design. The plan is `docs/superpowers/plans/2026-10-03-v2-compositional-communication.md`.
- **Benchmark config:** `configs/v2.toml`, SHA-256 `3e67be8d66766cf100f12d4966ce7878a7ef68417112847c53deb2b03a07336b`. It loads identically to `configs/v2_pilot_rung1.toml` (rung 1).
- **Held-out combinations** (`combo_seed = 20261003`): red sphere, yellow ring, green cylinder, blue cube, purple cone. They never appear in training. No learned checkpoint was evaluated on them before this tag: the pilots used the `train` split only.
- **Seeds:** benchmark 0–19 (`runs_v2/`); pilots 2000–2002 (`runs_v2/pilot/`). `EVAL_SEED` is 424242: `held_out` uses it, `train` uses +1, and `unseen` uses +2. Bootstrap seed 0 uses 10,000 resamples.
- **Metric and analysis code:** `src/emcomm/benchmark_v2.py`, `src/emcomm/analysis_v2.py`, and the v2 functions in `src/emcomm/stats.py`, as of this tag.

## Hypotheses and decision rule

Fixed-sequence testing: A, then B, then C, each at α = 0.05 (95% percentile bootstrap CI). A later claim is made only if every earlier claim holds.

- **H1:** `held_out` success(comm) − success(no_comm) ≥ 0.15, and the independent-bootstrap CI lower bound > 0.
- **H2:** `held_out` success(comm intact) − success(comm permuted) ≥ 0.10, and the paired-bootstrap CI lower bound > 0.
- **H3:** `unseen` success(comm) − success(no_comm) ≥ 0.15, and the independent-bootstrap CI lower bound > 0.
- **H4:** `unseen` success(comm intact) > 0.50, and the one-sample CI lower bound > 0.50.
- **Claims:** A = H1 ∧ H2 (“communication works in v2”); B = A ∧ H3 (“the language generalizes”); C = B ∧ H4 (“unseen success above 50%”). They are computed by `emcomm.analysis_v2.claims`, including the `h4_mean_above_reference` and `h4_ci_above_reference` tests.
- **Everything else is descriptive:** per-property accuracy, the twin-error split, word-level interventions, mutual information, topographic similarity, and positional disentanglement.
- **Unseen target-prior limitation:** H4 is a fixed performance reference. The unseen targets form a one-to-one color–shape matching, so a known property plus that matching identifies the pair. A support-aware no-message identifier can reach 87.5% before navigation. H4 does not establish transfer of both properties or compositionality.

## Pilot configurations tried (seeds 2000–2002, `train` split only)

Values below are means across pilot seeds, rounded to six decimal places from `results/v2/pilot_rung1.json` and `results/v2/pilot_rung2.json`.

| Rung | Change | oracle | no_comm | comm | gain | Decision |
|---|---|---:|---:|---:|---:|---|
| 1 | Frozen v1 settings, 20M steps | 0.851667 | 0.195000 | 0.348333 | 0.153333 | Passed all three checks |
| 2 | `sig_coef = 0.5` | 0.858333 | 0.195000 | 0.285000 | 0.090000 | Failed comm gain check |

Decision: rung 1 (plan Task 12, row 1). No rung 3 was needed. All 18 final pilot checkpoints completed at the common 20,004,864-step budget (2,442 updates); the reports evaluated the `train` split only.

## Commitments

- Evaluate `final.pt` only, with no checkpoint selection.
- Report every claim's outcome, including nulls. Do not tune after this tag.
- v1 stays reproducible: `tests/test_regression.py` passes, and `scripts/check_v1_reproduces.py` runs at V2-G4.
