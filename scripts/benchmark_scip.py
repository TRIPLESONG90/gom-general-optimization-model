from __future__ import annotations

import argparse
import json
import random
from statistics import mean

from gom.generators import generate_assignment, generate_knapsack
from gom.solvers.scip_policy import solve_scip_baseline, solve_with_gom_branching


def main():
    p = argparse.ArgumentParser(description="SCIP default/fullstrong/GOM branching benchmark")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--count", type=int, default=20)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--time-limit", type=float, default=10.0)
    p.add_argument("--device", default="cpu")
    p.add_argument("--out", default="scip_benchmark.jsonl")
    args = p.parse_args()

    rng = random.Random(args.seed)
    rows = []
    with open(args.out, "w", encoding="utf-8") as f:
        for i in range(args.count):
            problem = (
                generate_knapsack(rng, rng.randint(30, 80))
                if i % 2 == 0
                else generate_assignment(rng, rng.randint(6, 12))
            )
            for policy in ("default", "fullstrong", "gom"):
                if policy == "gom":
                    result = solve_with_gom_branching(
                        problem,
                        args.checkpoint,
                        time_limit_s=args.time_limit,
                        device=args.device,
                    )
                else:
                    result = solve_scip_baseline(
                        problem,
                        time_limit_s=args.time_limit,
                        policy=policy,
                    )
                row = {"problem_id": problem.id, "problem_type": problem.problem_type, **result.to_dict()}
                rows.append(row)
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                print(problem.id, policy, result.status, result.gap, result.nodes, f"{result.wall_time_s:.3f}s")

    print("\nsummary")
    for policy in ("default", "fullstrong", "gom"):
        selected = [r for r in rows if r["policy"] == policy]
        finite_gaps = [r["gap"] for r in selected if r["gap"] is not None]
        print(
            policy,
            f"mean_time={mean(r['wall_time_s'] for r in selected):.3f}s",
            f"mean_nodes={mean(r['nodes'] for r in selected):.1f}",
            f"mean_gap={mean(finite_gaps):.6f}" if finite_gaps else "mean_gap=n/a",
        )


if __name__ == "__main__":
    main()
