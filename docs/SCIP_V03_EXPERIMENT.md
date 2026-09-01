# SCIP v0.3 experiment

This experiment tests whether GOM can imitate SCIP strong-branching decisions and improve a time-budgeted branch-and-bound run.

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[scip,dev]'
```

A compatible SCIP/PySCIPOpt installation is required for trajectory collection and solver benchmarking.

## End-to-end smoke run

```bash
python scripts/run_experiment.py --profile smoke
```

The runner performs:

1. strong-branching trajectory collection,
2. problem-level train/validation split,
3. GOM branch-policy imitation training,
4. paired `default` vs `fullstrong` vs `gom` benchmarking,
5. Markdown and JSON result summarization.

## Research profile

```bash
python scripts/run_experiment.py --profile research
```

The research profile shards strong-branching data collection across multiple processes. Seeds used for benchmark instances are intentionally disjoint from trajectory collection/training seeds.

Configuration lives in `experiments/scip_v03.yaml`.

## Outputs

Each run creates a directory under `runs/` containing:

```text
manifest.json
trajectories-000.jsonl
trajectories-001.jsonl
...
gom_scip_branch.pt
benchmark.jsonl
summary.json
REPORT.md
```

`manifest.json` records the git commit, Python/Torch versions, CUDA information, profile, and effective experiment parameters.

## Benchmark interpretation

Every benchmark problem is solved using the same time limit with:

- `default`: SCIP default branching,
- `fullstrong`: SCIP full strong branching,
- `gom`: learned GOM branching policy.

The report contains aggregate metrics plus problem-paired wins against default branching.

Pairwise comparison is deliberately simple and explicit:

1. lower final optimality gap wins,
2. if gap ties, lower wall time wins,
3. if wall time ties, lower explored node count wins.

A useful first signal is not merely high imitation top-1 accuracy. The important result is whether GOM improves paired solver outcomes while keeping inference overhead reasonable.

## Dry-run / resume

Inspect commands without executing them:

```bash
python scripts/run_experiment.py --profile research --dry-run
```

Reuse an existing run directory and skip completed phases:

```bash
python scripts/run_experiment.py \
  --profile research \
  --run-dir runs/my-run \
  --skip-collect
```
