# Status

## Independent-review reporting revision — November 9, 2026

The report remains eight CVPR two-column pages including references. Structural metrics now identify their actual evaluation pools, and the report-specific Figure 7 limits scripted reference lines to the seen split. Course relevance, the precise cue-group objective, conditional uncertainty and the registered minimum-effect rule are explained more directly. A declared retrospective outcome audit from 40 preserved v2 evaluation logs retains 80 seed/split count rows, source hashes and seed-bootstrap intervals; it does not alter a frozen experiment or promote a generalization claim. Reproduce it with `uv run python scripts/report_diagnostics.py`.

The complete word-for-word narration script and local recording interface are included under `presentation/`. The separate review video uses disclosed local synthesis and actual Python launch-to-completion capture. The student's own narration and later live presentation remain separate actions. `docs/REPORT_REVIEW_FIXES.md` and `docs/SCIENTIFIC_QA.md` document the corrections and remaining limitations.

Validation: all 249 Python tests and eleven Node tests pass; all four canonical summaries reproduce and all 29 protected scientific-file hashes match. Ruff, formatting and diff checks pass. The diagnostic summary reproduces from the packaged count table. Publication verification follows this source commit; historical entries below describe earlier snapshots.

# Public project status

The public edition includes live v1 and v2 browser inference, a controllable 3D v2 view, v1/v2/vocal replay explorers, source-linked results and confidence intervals, a readable setup guide, the rendered research report, and downloadable seed tables/protocols. Four representative final Python checkpoints are bundled. No LLM service or API key is used.

Publication destination: https://ctad.dev/talking-agents/ · public source: https://github.com/ctadros1/talking-agents . The original private development history and course administration are excluded. Scientific source/configuration/pre-registration bytes and canonical results are preserved; `research-provenance.json` pins the source and frozen-file hashes.

Validation before initial publication: 237 original tests passed, plus three public-distribution checks; six Node tests cover real PyTorch GRU parity, environment observation/transition parity, seeded sampling, delayed messages/reset behavior and isolated Worker routing. All four saved statistical summaries reproduce exactly and all 29 protected files match. Fresh Python inference reproduces all 14 representative v1/v2 summary rows exactly. Ruff and format checks pass. Browser checks cover live playback, channel remapping and responsive results; final public deployment verification follows publication.

Publication verified on November 9, 2026: the GitHub repository is PUBLIC with main as its default branch; unauthenticated clone and project ZIP downloads include the exact trained models and result tables. Initial Cloudflare publication: Worker `talking-agents`, version `88173d22-0fcf-4666-9004-67717ef9e773`, serves only the two declared publication routes. Production browser checks verify fresh live model execution, channel scrambling/silencing, no-message model loading, all result views and replay playback. HTML/model/report/results ZIP requests succeed; nonexistent assets return 404. The existing protected site's visible content and forms are unchanged. A fresh public clone builds without npm installation, passes all six Node tests, reproduces all four statistical summaries and completes fresh v1/v2 Python inference.

Next action: use the public simulator or follow the README to run and inspect it locally. This publication does not submit the course assignment or create the author's narrated video.


## Live v2 and interface update: November 9, 2026

The browser now defaults to live v2 using the original final communication/no-communication checkpoints from seed 9. Both familiar and never-seen target banks contain 200 fixed held-out scenarios. Observations reproduce the 630-feature Python encoding, including the scout's hidden warehouse, the runner's hidden target, carrying attributes and two delayed words. Intact, both-word permutation, either-word permutation, silence and the separately trained no-message model are supported. The 3D renderer consumes live environment state and does not read recorded trajectories; v1 and archived replays remain available. Camera orbit/zoom/reset, keyboard access, offscreen rendering suspension and a 2D WebGL fallback are included.

The evidence view preserves cards and semantically matched bar nodes across tabs. Bar widths and displayed numbers animate from their current values, missing comparisons collapse, and the selected-tab indicator slides. Reduced-motion preferences bypass the transitions. All measured values still come from frozen summary JSON. Locally vendored Lucide Static 1.70.0 (ISC) replaces UI glyphs across the homepage and served replay pages, including download/external-link icons and stateful transport/audio controls. Its license, pinned source and package digest are included; no icon CDN or npm install is needed to run the site.

Local verification: 241 Python tests passed. Eleven Node tests cover both phases' independent PyTorch/environment fixtures, cue privacy, two-word timing/interventions/reset, persistent comparison keys, camera projection synchronization and Worker routing. All four summaries and all 29 protected scientific files still verify exactly. Browser checks exercise live v2 success, all channel modes, target splits, both baseline checkpoints, v1/v2 switching, camera controls, replay icon updates, mobile page width and a bar visibly draining from its v1 value toward the v2 value. Production publication and final verification follow this source commit.

Next action: publish and verify the live v2 update at the existing public URL.


Production update verified: Cloudflare version `ec2a0b19-baa6-453e-a0aa-1e1558dd912e`, from public source commit `9c87d53`. The public page executes live two-word inference and completes a correct delivery; on-site channel interventions, baseline loading and responsive layout work without application errors. A production evidence-tab switch visibly drains the shared communication bar from the v1 value toward the v2 target. Both v2 binary payloads, the icon sprite, scene/evidence modules and results ZIP match the reviewed build byte for byte. The public project ZIP contains the exact new model and interface files. A fresh public clone builds the site, passes all eleven Node checks and verifies all four summaries and 29 protected scientific files. The existing protected site route continues to return its protected page.

Next action: open the public site and run v2 live, or follow the README to run the same version locally.
