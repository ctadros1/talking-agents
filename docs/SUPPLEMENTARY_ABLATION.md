# Original-method ablation for the course submission

Declared November 9, 2026, before launching this supplementary experiment.
The v1 and v2 benchmark outcomes and the three-seed v1 pilot outcomes have
already been seen. This is a **post-hoc exploratory comparison**, not a new
confirmatory test or an amendment to either frozen benchmark.

The method being modified is independent recurrent PPO (IPPO). The task-specific
modification adds a private-cue mutual-information objective to the speaker,
following the positive-signalling idea of Eccles et al. (2019). The baseline
removes that objective. This does not reproduce the Eccles environments or
establish novelty of positive signalling itself.

## Fixed comparison

- Train unmodified `comm` on seeds 0–19 with `configs/ablation_unbiased.toml`.
- The sole change from `configs/base.toml` is `sig_coef: 0.2 → 0`. Architecture,
  observations, actions, reward, optimizer, training layouts, sampling and the
  full 10,002,432-step budget are equal.
- Save new runs in `runs/ablation_unbiased/comm/seed_XXXX/`. Preserve all original
  runs, results and checkpoints.
- Evaluate **final.pt only** with the unchanged v1 evaluator: fixed 200 episodes
  per split and the same intact/permuted/silenced interventions. Write the
  additional table to `results/ablation_unbiased/`.
- Compare held-out intact success with the frozen biased-comm results, using
  the existing independent seed bootstrap, 10,000 resamples and bootstrap seed
  0. Report both means and their difference with 95% CIs. Report the unbiased
  baseline's paired permutation drop descriptively.
- No tuning, early stopping, checkpoint selection, threshold change or repeat
  experiment to improve the outcome follows this comparison.
- All 20 seeds must finish. Partial results remain incomplete and are never
  merged with the three-seed pilot or presented as a complete experiment.

The source commit, configuration/source hashes, start time and launch command
are recorded with the run before training. Original v1/v2 analysis, configuration,
pre-registration and summary bytes remain unchanged.

The contribution is a controlled implementation and evaluation of cue-directed
signalling and its limits in a split-information task. It is not a claim to
have invented PPO, mutual information or emergent communication. This comparison
addresses the course's original-method control, conditional on this small task
and the already selected configuration.
