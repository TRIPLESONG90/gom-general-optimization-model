# GOM Roadmap

GOM explores a **general optimization policy model** that learns how to control search rather than replacing mathematical optimization solvers outright.

## v0.5 — native solver-state representation

- [x] Capture SCIP native current-LP bipartite graphs.
- [x] Connect native-LP checkpoints to a live SCIP branch rule.
- [x] Vectorize live edge tensorization and remove the generic copy pass.
- [x] Validate native ranking distillation against top-1 imitation.
- [ ] Scale trajectory collection and run the 64-problem research benchmark.

## v0.2 — executable research scaffold

- Unified linear Optimization IR
- Relation-aware graph Transformer
- Solver/action/variable/value heads
- Synthetic Knapsack and Assignment teachers
- Search-state features
- SCIP strong-branching trajectory collector
- OR-Tools CP-SAT adapter

## v0.3 — dynamic solver policy

Implemented scaffold:

- Train on `(problem, solver_state) -> branch variable`
- Replayable SCIP trajectory JSONL dataset
- Candidate-masked branch imitation loss
- GOM branching rule with safe SCIP fallback
- Default branching vs full-strong branching vs GOM benchmark harness
- Runtime metrics: objective, primal/dual bound, gap, nodes, wall time, GOM inference overhead

Next measurement work:

- primal integral / dual integral traces
- larger heterogeneous MILP corpus
- held-out-family generalization benchmark

## v0.4 — learned neighborhood search

- LNS action vocabulary
- Variable fixing / relaxation policies
- Neighborhood size prediction
- Policy history encoder

## v0.5 — multi-family generalization

Training families:

- Knapsack
- Assignment
- Set Cover
- Bin Packing
- Facility Location
- Job Shop

Hold out one complete problem family and evaluate zero-shot, few-shot and fine-tuned performance.

## v1.0 — General Optimization Model research baseline

A single backbone should jointly support:

- solver selection,
- branching,
- LNS control,
- value prediction,
- cross-instance transfer,
- cross-problem transfer.

The central research question is not whether GOM beats every specialized solver. It is whether a shared learned search policy transfers useful optimization knowledge across problem families and improves classical solvers under fixed compute budgets.
