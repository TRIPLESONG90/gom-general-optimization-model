# SCIP v0.6 depth pre-gate

## Hypothesis

The v0.5 confidence gate paid neural inference cost before abstaining. A cheap
depth pre-gate can skip graph extraction, tensorization, and model inference
entirely on deeper branch states.

## Protocol

- Checkpoint: v0.5 464K native-LP ranking model
- Held-out seed: 99001
- Time limit: 10 seconds
- Sweep: maximum GOM depth 0, 1, 3, 5, and 10 on 16 problems
- Confirmation: depth 0 and 1 on all 64 held-out problems

## Result

| Policy | Optimal | Mean gap | Mean time | Mean nodes | Paired wins vs default |
|---|---:|---:|---:|---:|---:|
| SCIP default | 62/64 | 0.000193 | 1.914s | 1,395.5 | n/a |
| GOM depth 0 | 62/64 | 0.000191 | 1.981s | 1,335.9 | 27/64 |
| GOM depth 1 | 62/64 | 0.000216 | 1.879s | 1,272.0 | 26/64 |

Root-only GOM made only 48 neural decisions across 64 problems. Depth-one GOM
made 142. The pre-gate therefore eliminated the v0.5 inference collapse and
restored SCIP's optimal rate. Root-only reduced mean nodes by 4.3% but increased
mean time by 3.5%. Depth one reduced mean nodes by 8.9% and mean time by 1.8%,
but worsened mean gap. Neither policy won a majority of paired comparisons.

## Decision

Keep the depth pre-gate as infrastructure, but do not claim an improvement over
SCIP default. The next policy should predict candidate regret or expected
advantage over SCIP's choice. Intervention should be trained against downstream
search value, not only top-1 imitation of a strong-branch teacher.
