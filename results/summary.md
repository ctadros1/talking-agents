| Condition | Held-out success | Train-layout success | Gen. gap | Steps / success | MI(target; first msg) bits |
|---|---|---|---|---|---|
| comm | 87.6% [86.3, 88.6] | 83.7% [82.6, 84.8] | -3.8 pp [-5.2, -2.4] | 16.9 [16.8, 17.0] | 2.0 [2.0, 2.0] |
| no_comm | 22.5% [21.8, 23.3] | 21.0% [20.3, 21.7] | -1.5 pp [-2.4, -0.7] | 15.5 [15.1, 15.9] | 2.0 [2.0, 2.0] |
| random_msg | 22.3% [21.5, 23.2] | 21.2% [20.4, 22.1] | -1.1 pp [-2.2, -0.1] | 15.3 [15.0, 15.6] | 2.0 [2.0, 2.0] |
| oracle | 93.1% [92.4, 93.8] | 90.6% [89.8, 91.3] | -2.5 pp [-3.4, -1.6] | 15.9 [15.8, 16.0] | 2.0 [2.0, 2.0] |
| scripted_comm (reference) | 100.0% | | | | |
| scripted_no_comm (reference) | 26.5% | | | | |
| scripted_random_msg (reference) | 25.0% | | | | |
| scripted_oracle (reference) | 100.0% | | | | |

| Comparison | Mean [95% CI] |
|---|---|
| comm_minus_no_comm | 65.0 pp [63.5, 66.3] |
| comm_minus_random_msg | 65.2 pp [63.7, 66.5] |
| intact_minus_permuted | 67.6 pp [66.0, 69.1] |
| intact_minus_silenced | 74.9 pp [73.0, 77.0] |

**Pre-registered success criterion met: True**

- gain_at_least_15pp: True
- gain_ci_excludes_zero: True
- permutation_drop_at_least_10pp: True
- permutation_drop_ci_excludes_zero: True
