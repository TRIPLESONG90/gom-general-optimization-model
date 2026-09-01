# SCIP v0.5 native-LP research run

This report records a negative result rather than selecting only the earlier
six-instance smoke result.

## Protocol

- Teacher corpus: 512 generated problems, seed 61000
- Native SCIP LP branch states: 27,775
- States with candidate rankings: 27,757 (99.94%)
- Mean strong-branch candidates: 7.07
- Model: 464,101 parameters, five epochs, ranking weight 0.75
- Split: by problem ID, 311 train / 55 validation problems
- Best validation top-1: 31.74%
- Benchmark: 64 held-out problems, seed 99001, ten seconds per run

## Held-out result

| Policy | Optimal | Mean gap | Mean time | Mean nodes |
|---|---:|---:|---:|---:|
| SCIP default | 62/64 | 0.000193 | 1.914s | 1,395.5 |
| SCIP fullstrong | 61/64 | 0.000318 | 1.832s | 509.3 |
| GOM native LP | 48/64 | 0.001477 | 4.238s | 742.3 |

GOM beat default on node count in 22/64 pairs, but beat it on final gap in
0/64. Lower node count therefore did not demonstrate a better policy: GOM's
5.31ms inference overhead reduced the amount of search completed within the
time budget.

Mean live latency per branch decision:

| Component | Time |
|---|---:|
| SCIP graph extraction | 1.10ms |
| Tensorization | 1.19ms |
| Model forward | 3.00ms |
| Total | 5.31ms |

## Confidence gate

Confidence was calibrated on 2,000 branch states. Higher confidence correlated
with higher top-1 accuracy, but a 16-problem threshold sweep did not recover
SCIP performance. Even at threshold 0.50, GOM intervened in only 0.57% of
observed states while still paying inference cost before every abstention.

| Threshold | Intervention rate | Optimal | Mean gap | Mean time |
|---:|---:|---:|---:|---:|
| 0.30 | 18.97% | 12/16 | 0.001571 | 3.762s |
| 0.40 | 3.15% | 12/16 | 0.001703 | 3.621s |
| 0.50 | 0.57% | 12/16 | 0.001710 | 3.800s |
| SCIP default | n/a | 15/16 | 0.000536 | 2.037s |

## Decision

Do not scale the current always-invoke policy. The next experiment must avoid
neural inference on most states using a cheap pre-gate, then predict candidate
regret or advantage rather than only imitating the teacher's top-1 variable.
A depth-limited policy is the first controlled pre-gate baseline.
