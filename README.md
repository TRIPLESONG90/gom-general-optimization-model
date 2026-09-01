# GOM-30M — General Optimization Model

## v0.5 native LP branching

The live SCIP policy consumes SCIP's current bipartite LP graph (active columns,
rows/cuts, coefficients, LP/basis features) instead of reconstructing solver
state from the original static MILP. Live tensorization uses a vectorized
single-snapshot path that avoids Python edge loops and generic batch padding.

For native-LP training, ranking distillation is recommended:

```bash
python scripts/train_scip_lp_branch.py scip_trajectories.jsonl \
  --ranking-weight 0.75 --ranking-temperature 1.0 \
  --out gom_scip_lp_branch.pt

python scripts/benchmark_scip.py \
  --checkpoint gom_scip_lp_branch.pt --gom-input lp
```

GOM is a research scaffold for a **general optimization policy model**: a neural model that learns how to control mathematical optimization search instead of trying to replace classical solvers with one-shot prediction.

```text
Optimization Problem + Solver State
                ↓
        Unified Optimization IR
                ↓
 Variable / Constraint / Global graph
                ↓
   Relation-aware Transformer (~34M)
                ↓
 ┌────────┬────────┬────────┬────────┐
 │ Solver │ Action │ Branch │ Value  │
 │  Head  │  Head  │  Head  │  Head │
 └────────┴────────┴────────┴────────┘
                ↓
       SCIP / CP-SAT / LNS
```

## v0.3 status

The v0.3 branch adds the first closed solver-control loop:

- linear Optimization IR for binary/integer/continuous variables;
- relation-aware variable/constraint graph Transformer;
- solver, action, variable-selection and value heads;
- synthetic Knapsack and Assignment teachers;
- SCIP strong-branching trajectory collection;
- replayable trajectory JSONL dataset;
- `(problem, solver_state) -> branch variable` imitation training;
- candidate-masked branch cross-entropy;
- GOM LP branching rule with safe fallback to SCIP;
- SCIP default vs full-strong vs GOM benchmark harness;
- runtime metrics for objective, bounds, gap, nodes, wall time and GOM inference overhead;
- sanitization of non-finite solver bounds before neural featurization.

This is **not yet a claim that GOM is a general solver**. It is the executable infrastructure required to test whether a shared learned search policy can transfer useful optimization knowledge across problem families.

## Install

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -e .
```

Optional solvers:

```bash
pip install -e '.[solver]'
pip install -e '.[scip]'
```

## Smoke test

```bash
python scripts/demo.py
pytest -q
```

## Synthetic pretraining

Tiny smoke run:

```bash
python scripts/train_synthetic.py --tiny --steps 100 --batch-size 8
```

Default ~34M model:

```bash
python scripts/train_synthetic.py --steps 10000 --batch-size 16 --device cuda
```

## Collect SCIP strong-branch trajectories

```bash
python scripts/collect_scip.py \
  --count 1000 \
  --time-limit 10 \
  --max-steps 128 \
  --out data/scip_train.jsonl
```

Each branch record contains the optimization problem plus dynamic state such as primal/dual bounds, gap, depth, node count, elapsed time, LP candidate values, fractionalities and local bounds, followed by the strong-branching teacher choice.

## Train the branch policy

```bash
python scripts/train_scip_branch.py \
  data/scip_train.jsonl \
  --device cuda \
  --epochs 10 \
  --batch-size 16 \
  --out gom_scip_branch.pt
```

You can also initialize from a synthetic-pretraining checkpoint:

```bash
python scripts/train_scip_branch.py \
  data/scip_train.jsonl \
  --init gom_checkpoint.pt \
  --device cuda \
  --out gom_scip_branch.pt
```

## Benchmark solver control

```bash
python scripts/benchmark_scip.py \
  --checkpoint gom_scip_branch.pt \
  --count 20 \
  --time-limit 10
```

The benchmark runs the same generated instances under:

1. SCIP default branching;
2. SCIP full-strong branching;
3. GOM-guided LP branching.

The GOM branch rule only selects among the current SCIP LP branching candidates. If inference fails or the predicted variable cannot be used, it returns control to SCIP rather than breaking the solve.

## Research target

The central hypothesis is:

> A shared neural search policy can learn reusable optimization structure across problem families and improve classical solvers under fixed compute budgets.

The strongest future experiment is to train on several families, hold out one complete family, then measure zero-shot, few-shot and fine-tuned solver-control performance.

See [ROADMAP.md](ROADMAP.md) for the planned progression toward LNS control, history encoding and cross-family generalization.
