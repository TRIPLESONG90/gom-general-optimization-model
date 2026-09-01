# GOM Roadmap

GOM explores a **general optimization policy model** that learns how to control search rather than replacing mathematical optimization solvers outright.

## v0.2 — executable research scaffold

- Unified linear Optimization IR
- Relation-aware graph Transformer
- Solver/action/variable/value heads
- Synthetic Knapsack and Assignment teachers
- Search-state features
- SCIP strong-branching trajectory collector
- OR-Tools CP-SAT adapter

## v0.3 — dynamic solver policy

- Train on `(problem, solver_state) -> branch action`
- Replayable SCIP trajectory dataset
- Default branching vs strong branching vs GOM branching benchmark
- Metrics: primal integral, dual integral, gap-vs-time, inference overhead

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
