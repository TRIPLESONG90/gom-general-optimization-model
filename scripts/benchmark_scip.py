from __future__ import annotations

import argparse
import json
import random
from statistics import mean

from gom.generators import generate_knapsack, generate_multidimensional_knapsack
from gom.solvers.scip_policy import solve_scip_baseline, solve_with_gom_branching
from gom.solvers.scip_lp_policy import solve_with_gom_lp_branching


def main():
    p = argparse.ArgumentParser(description="SCIP default/fullstrong/GOM branching benchmark")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--count", type=int, default=20)
    p.add_argument("--seed", type=int, default=10000, help="Use a seed disjoint from training data")
    p.add_argument("--time-limit", type=float, default=10.0)
    p.add_argument("--device", default="cpu")
    p.add_argument("--gom-input", choices=("static", "lp"), default="static", help="GOM state representation")
    p.add_argument("--threads", type=int, default=1, help="CPU intra-op threads for native LP GOM")
    p.add_argument("--out", default="scip_benchmark.jsonl")
    args = p.parse_args()

    gom_policy_name = "gom-lp" if args.gom_input == "lp" else "gom"
    rng = random.Random(args.seed)
    rows = []
    with open(args.out, "w", encoding="utf-8") as f:
        for i in range(args.count):
            if i % 4 == 0:
                problem = generate_knapsack(rng, rng.randint(80, 160))
            else:
                problem = generate_multidimensional_knapsack(
                    rng,
                    n=rng.randint(70, 150),
                    m=rng.randint(5, 12),
                )
            for requested_policy in ("default", "fullstrong", "gom"):
                if requested_policy == "gom":
                    if args.gom_input == "lp":
                        result = solve_with_gom_lp_branching(
                            problem,
                            args.checkpoint,
                            time_limit_s=args.time_limit,
                            device=args.device,
                            threads=args.threads,
                        )
                    else:
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
                        policy=requested_policy,
                    )
                row = {"problem_id": problem.id, "problem_type": problem.problem_type, **result.to_dict()}
                rows.append(row)
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                print(problem.id, result.policy, result.status, result.gap, result.nodes, f"{result.wall_time_s:.3f}s")

    print("\nsummary")
    for policy in ("default", "fullstrong", gom_policy_name):
        selected = [r for r in rows if r["policy"] == policy]
        if not selected:
            continue
        finite_gaps = [r["gap"] for r in selected if r["gap"] is not None]
        print(
            policy,
            f"mean_time={mean(r['wall_time_s'] for r in selected):.3f}s",
            f"mean_nodes={mean(r['nodes'] for r in selected):.1f}",
            f"mean_gap={mean(finite_gaps):.6f}" if finite_gaps else "mean_gap=n/a",
        )
    gom = [r for r in rows if r["policy"] == gom_policy_name]
    if gom:
        total_decisions = sum(r["gom_decisions"] for r in gom)
        total_inference = sum(r["gom_inference_ms"] for r in gom)
        print(
            "gom diagnostics",
            f"decisions={total_decisions}",
            f"fallbacks={sum(r['gom_fallbacks'] for r in gom)}",
            f"inference_ms={total_inference:.1f}",
            f"ms_per_decision={total_inference / max(1, total_decisions):.3f}",
        )


if __name__ == "__main__":
    main()
