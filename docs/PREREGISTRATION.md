# Pre-registration: Emergent Communication in a Grid World

Frozen 2026-09-30 at git tag `prereg-v1`, before any benchmark seed (0–19) was trained.

## Frozen artifacts

- **Benchmark config:** `configs/base.toml`, SHA-256 `fce5e4b264b4b9f009d49593ebb1a349c86bf7d0d5a5ac37b1d141dc0f940561`. It is identical to `configs/pilot_rung1.toml` (rung 1: `sig_coef = 0.2`).
- **Design spec:** `docs/superpowers/specs/2026-09-30-emergent-communication-design.md` at commit `3a2c80e846032d26709bfc47708218a0541a20db`. Sections 2–8 define the environment, conditions, learning setup (including the §4.1 cue-informativeness bias), metrics, evaluation protocol, statistics, and seeds.
- **Metric and statistics code:** `src/emcomm/{benchmark,stats,analysis}.py` at this tag.
- **Budget:** 1221 updates × 8,192 = 10,002,432 env steps per run, identical for all 80 runs.

## Hypotheses and decision rule

- **H1:** held-out success(comm) − success(no_comm) ≥ 0.15, with the lower bound of the 95% bootstrap CI > 0. The CI is independent across conditions, uses 10,000 resamples, and uses seed 0.
- **H2:** held-out success(comm, intact) − success(comm, permuted) ≥ 0.10, with the lower bound of the paired 95% bootstrap CI > 0.
- The proposal's success criterion is met if and only if H1 and H2 both hold. All other quantities are descriptive.

## Pilot configurations tried (pilot seeds 1000–1002, training layouts only)

| Rung | Change | oracle | no_comm | comm | Decision |
|---|---|---|---|---|---|
| 0 | pure IPPO defaults | 0.872 | 0.208 | 0.218 | gain +0.010 < 0.15: plan Task 14 table row 2 did not fire |
| 1 | `sig_coef = 0.2` | 0.915 | 0.208 | 0.832 | gain +0.623: row 3 fired, **frozen** |

Per-seed values are in `results/pilot_rung0.json` and `results/pilot_rung1.json`. The no_comm values are bit-identical across the two rungs, because the bias only affects the scout's undelivered symbols in that condition.

Prototype configurations tried on 2026-09-30, before this pilot, are listed in plan Task 14:
- **Rejected:** a FiLM encoder, lr 3e-4, whole-state informativeness, symbol-head init gain 1.0, and no pickup bonus.
- **Not chosen:** `sig_coef = 0.05`.

## Declared exploratory work (not confirmatory)

Spec §15 declares two extensions before this tag:
- **E1:** sonified replays.
- **E2:** a time-boxed vocal-channel arm, with its own run tree and descriptive statistics only.

Neither bears on H1 or H2.

## Commitments

- Evaluate `final.pt` only, on 200 held-out and 200 train-layout episodes per run, with `EVAL_SEED = 424242`.
- Log every departure after this tag in `docs/DEVIATIONS.md` before acting on it.
- Report a null or partial result as such.
