# Response to the independent scientific-track review

Revised November 9, 2026, after the original benchmark and independent review were seen. This is reporting and retrospective analysis, not a new confirmatory experiment.

- The structural table now names each evaluation pool. Word MI uses 200 seen episodes per seed; topographic similarity and positional disentanglement use pooled 200 seen + 200 unseen episodes. The frozen values are retained.
- Figure 7 is a report-specific redraw. Scripted seen-split references appear exclusively in the seen panel. The original canonical plot is retained for provenance. `uv run python scripts/plot_report_v2.py` reproduces the corrected figure from the original summary.
- The report and full narration script distinguish a point-estimate minimum plus a CI excluding zero from a lower confidence bound at the minimum.
- The uncertainty statement explicitly conditions seed-bootstrap intervals on the fixed scenarios. These are not uncertainty estimates across task generators or hold-out matchings.
- The course connection now explains information boundaries, recurrent internal state, factorized actions, decentralized actor–critics, shared-reward credit assignment and communication interventions. These implement Topic 7's broad RL/agent concepts; specific lecture coverage of PPO or GRU is not asserted.
- The novelty statement precisely describes grouping predicted message distributions by cue. The objective is a task-specific adaptation of positive signalling. The ablation establishes its value relative to unbiased IPPO here, not superiority to the full Eccles method.
- A new outcome audit partitions preserved comm and oracle evaluations into no pickup, wrong pickup, correct pickup without delivery and success. All 80 seed/split rows, hashes of 40 source logs and seed-bootstrap intervals are saved. `uv run python scripts/report_diagnostics.py` reproduces the summary from packaged counts; `--runs runs_v2` rebuilds counts from original complete logs. No checkpoint, training run, selection rule, metric or configuration changed.
- The video review revision replaces the displayed static Python log with a real process launched during capture, streamed stdout and an exit code. It retains the actual browser inference captures. The 3D scene is a grid rendering, not a separate physics experiment.
- Review narration is explicitly temporary local macOS Samantha synthesis. The full word-for-word script remains available for the student's recording. No claim is made that the student's own narration or live presentation is complete.

## Remaining scientific limits

One constructed environment and one hold-out matching limit external significance. The v2 support shortcut is disclosed; it cannot be removed retrospectively while retaining a prospective interpretation. Navigation and identification are not causally separated by the descriptive failure audit. The closest full prior method is not experimentally compared. The direct objective ablation remains post-hoc. None of these limitations is hidden by the reporting fixes.

A stronger next study would prospectively compare the original trajectory-based objective with the cue-directed term at equal budgets, and preregister a redesigned combination split with multiple matchings and independent task replication. Those studies are not represented as completed work.
