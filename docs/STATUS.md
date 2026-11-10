# Public project status

The public edition includes a live v1 browser sandbox, v1/v2/vocal replay explorers, source-linked results and confidence intervals, a readable setup guide, the rendered research report, and downloadable seed tables/protocols. Four representative final Python checkpoints are bundled. No LLM service or API key is used.

Publication destination: https://ctad.dev/talking-agents/ · public source: https://github.com/ctadros1/talking-agents . The original private development history and course administration are excluded. Scientific source/configuration/pre-registration bytes and canonical results are preserved; `research-provenance.json` pins the source and frozen-file hashes.

Validation before initial publication: 237 original tests passed, plus three public-distribution checks; six Node tests cover real PyTorch GRU parity, environment observation/transition parity, seeded sampling, delayed messages/reset behavior and isolated Worker routing. All four saved statistical summaries reproduce exactly and all 29 protected files match. Fresh Python inference reproduces all 14 representative v1/v2 summary rows exactly. Ruff and format checks pass. Browser checks cover live playback, channel remapping and responsive results; final public deployment verification follows publication.

Next action: publish the reviewed snapshot and verify the deployed route, downloads and replay viewers.
