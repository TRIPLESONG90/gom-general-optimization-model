# GOM-30M — General Optimization Model MVP

A runnable research scaffold for a **general optimization policy model** rather than a one-shot neural solver.

The core loop is:

```text
Optimization Problem
        ↓
Unified Optimization IR
        ↓
Variable / Constraint / Global graph
        ↓
Relation-aware Transformer (~30M params)
        ↓
┌────────────┬────────────┬───────────────┐
│ Solver Head│ Action Head│ Variable Head │ + Value Head
└────────────┴────────────┴───────────────┘
        ↓
Classical solver / LNS / branching controller
```

## What v0.2 implements

- Linear Optimization IR for binary/integer/continuous variables.
- Unified graph encoding: GLOBAL, VARIABLE, CONSTRAINT nodes.
- Relation-aware multi-head attention with learned structural bias.
- Four outputs: solver selection, search action, variable selection, value.
- Infinite synthetic data generation for Knapsack and Assignment.
- Exact small-instance teachers (DP).
- Optional OR-Tools CP-SAT adapter.
- Smoke tests and synthetic imitation training.

See [ROADMAP.md](ROADMAP.md).
