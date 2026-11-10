| Condition | Seen (held_out) | Never-seen (unseen) | Train layouts | Topsim | Posdis |
|---|---|---|---|---|---|
| comm | 37.1% [35.1, 39.0] | 26.7% [23.9, 29.8] | 36.1% [34.2, 37.9] | 0.24 [0.21, 0.27] | 0.16 [0.12, 0.20] |
| no_comm | 21.8% [20.7, 22.9] | 19.7% [18.8, 20.6] | 21.2% [20.0, 22.3] | 0.24 [0.20, 0.27] | 0.13 [0.10, 0.17] |
| random_msg | 20.5% [19.8, 21.3] | 18.4% [17.6, 19.2] | 21.2% [20.6, 21.9] | 0.23 [0.20, 0.27] | 0.13 [0.10, 0.17] |
| oracle | 84.6% [83.8, 85.5] | 73.3% [70.8, 75.7] | 83.6% [82.5, 84.6] | 0.23 [0.19, 0.25] | 0.13 [0.10, 0.17] |

| Scripted reference | held_out | unseen | train |
|---|---|---|---|
| scripted_color_only | 49.0% | 49.0% | 52.5% |
| scripted_comm | 100.0% | 100.0% | 100.0% |
| scripted_no_comm | 29.5% | 22.5% | 26.0% |
| scripted_oracle | 100.0% | 100.0% | 100.0% |
| scripted_random_msg | 29.5% | 21.5% | 27.5% |
| scripted_shape_only | 53.0% | 46.5% | 53.0% |

| Comparison (held_out) | Mean [95% CI] |
|---|---|
| comm_minus_no_comm | +15.3 pp [+13.0, +17.6] |
| comm_minus_random_msg | +16.6 pp [+14.5, +18.6] |
| intact_minus_permuted | +22.3 pp [+20.2, +24.3] |
| intact_minus_permuted_w1 | +10.0 pp [+8.4, +11.5] |
| intact_minus_permuted_w2 | +10.2 pp [+8.4, +11.9] |
| intact_minus_silenced | +24.6 pp [+22.2, +27.0] |

| Comparison (unseen) | Mean [95% CI] |
|---|---|
| comm_minus_no_comm | +7.0 pp [+4.1, +10.2] |
| comm_minus_random_msg | +8.4 pp [+5.4, +11.5] |
| intact_minus_permuted | +12.2 pp [+9.2, +15.3] |
| intact_minus_permuted_w1 | +5.5 pp [+4.0, +7.1] |
| intact_minus_permuted_w2 | +6.9 pp [+4.8, +9.1] |

**Claim A (communication works): True** · **Claim B (generalizes): False** · **Claim C (unseen success above 50% reference): False**

- h1_gain_at_least_15pp: True
- h1_ci_excludes_zero: True
- h2_drop_at_least_10pp: True
- h2_ci_excludes_zero: True
- h3_gain_at_least_15pp: False
- h3_ci_excludes_zero: True
- h4_mean_above_reference: False
- h4_ci_above_reference: False
