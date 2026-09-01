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
- Four outputs:
  - solver selection logits,
  - search action logits,
  - variable selection logits,
  - value vector.
- Infinite synthetic data generation for Knapsack and Assignment.
- Exact small-instance teachers (DP), so training can run without external solvers.
- Optional OR-Tools CP-SAT adapter.
- Smoke tests and a synthetic imitation-training script.

This version deliberately does **not** claim to be a general solver yet. It is the executable backbone required to run the experiments that can test that hypothesis.

## Repository status

**v0.2 research scaffold.** The immediate target is a reproducible SCIP branching benchmark comparing default branching, strong branching, and GOM-guided branching under the same time budget. See [ROADMAP.md](ROADMAP.md).

## Install

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\\Scripts\\activate
pip install -e .
```

Optional classical solvers:

```bash
pip install -e '.[solver]'
```

## Smoke test

```bash
python scripts/demo.py
pytest -q
```

## Train a tiny smoke-test model

```bash
python scripts/train_synthetic.py --tiny --steps 100 --batch-size 8
```

## Train the default ~30M model

```bash
python scripts/train_synthetic.py --steps 10000 --batch-size 16 --device cuda
```

The default model is intentionally small enough for iteration. The path to GOM-100M/300M/1B is scaling `d_model`, `n_layers`, context size, data diversity, and solver trajectory supervision after the architecture is validated.

## IR example

```python
from gom.ir import Variable, LinearConstraint, OptimizationProblem

p = OptimizationProblem(
    id="demo",
    sense="max",
    variables=[
        Variable("x", "binary", 0, 1),
        Variable("y", "integer", 0, 10),
    ],
    objective={"x": 5, "y": 2},
    constraints=[
        LinearConstraint("capacity", {"x": 3, "y": 1}, "<=", 8)
    ],
)
```

## Research roadmap

### Phase A — validate universal representation

Train on Knapsack + Assignment + Set Cover + Bin Packing and measure whether one backbone matches problem-specific models at equal parameter/compute budgets.

### Phase B — solver-state learning

Add state features:

- incumbent objective,
- best bound / gap,
- variable incumbent values,
- LP relaxation values,
- reduced cost / pseudo-cost,
- constraint activity / slack,
- node depth,
- elapsed time,
- historical improvement.

Then collect SCIP branching trajectories and LNS trajectories. Change training examples from `(problem → expert variable)` to:

```text
(problem, search_state, history) → next_search_action
```

### Phase C — GOM policy

Use an action vocabulary such as:

```text
BRANCH(variable, direction)
FIX(variable, value)
RELAX(variable)
LNS(neighborhood_id, size)
RESTART(strategy)
SWITCH_SOLVER(solver)
SET_PARAM(parameter, bucket)
```

Train with behavior cloning first, then online RL using reward based on primal/dual integral, objective gap, solve time, and constraint violations.

### Phase D — generalization experiment

Train problem families:

- Knapsack
- Assignment
- TSP/CVRP
- Job Shop
- Bin Packing
- Set Cover

Hold out entire families such as Facility Location. Evaluate zero-shot → few-shot → fine-tuned performance. This is the experiment that tests whether the model deserves the word **General**.

## Primary benchmark metrics

Do not measure only final objective. For solver control, use:

- primal gap vs time,
- dual gap vs time,
- primal integral,
- time to first feasible solution,
- time to target gap,
- solved count under fixed budget,
- inference overhead,
- cross-size generalization,
- cross-problem generalization.

## Immediate next engineering step

Implement a SCIP trajectory collector and an LNS environment. That will convert this from a static problem encoder into the intended **Next Optimization Action Prediction** system.

## Strong-branching trajectory collection (v0.2 path)

GOM now has a solver-state schema and an optional SCIP teacher collector. PySCIPOpt exposes LP branching candidates, fractionalities, strong-branching information and custom branching rules, which makes SCIP a much better source of **Next Optimization Action** labels than CP-SAT's high-level Python API.

Install a compatible SCIP/PySCIPOpt pair:

```bash
pip install -e '.[scip]'
```

Collect trajectories:

```bash
python scripts/collect_scip.py --count 1000 --time-limit 10 --max-steps 128 --out data/scip_train.jsonl
```

Each branching step records:

```text
problem
+ primal/dual bound, gap, depth, node count, elapsed time
+ LP candidate values/fractionalities/local bounds
→ strong-branching chosen variable + score
```

`featurize_problem(problem, state)` injects those dynamic features into the same graph used by the static synthetic pretraining pipeline. The next training milestone is to mix static synthetic examples with these dynamic SCIP branch examples.
