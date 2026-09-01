from __future__ import annotations

import argparse
import json
import random

from gom.generators import generate_knapsack, generate_multidimensional_knapsack
from gom.solvers.scip_lp_policy import solve_with_gom_lp_branching


def main() -> None:
    parser = argparse.ArgumentParser(description="Sweep cheap depth pre-gates for GOM native-LP branching")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--depths", type=int, nargs="+", required=True)
    parser.add_argument("--count", type=int, default=16)
    parser.add_argument("--seed", type=int, default=99001)
    parser.add_argument("--time-limit", type=float, default=10.0)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--out", default="scip_depth_sweep.jsonl")
    args = parser.parse_args()

    rng = random.Random(args.seed)
    problems = []
    for i in range(args.count):
        if i % 4 == 0:
            problem = generate_knapsack(rng, rng.randint(80, 160))
        else:
            problem = generate_multidimensional_knapsack(
                rng, n=rng.randint(70, 150), m=rng.randint(5, 12)
            )
        problems.append(problem)

    with open(args.out, "w", encoding="utf-8") as output:
        for max_depth in args.depths:
            for problem in problems:
                result = solve_with_gom_lp_branching(
                    problem,
                    args.checkpoint,
                    time_limit_s=args.time_limit,
                    threads=args.threads,
                    max_gom_depth=max_depth,
                )
                row = {
                    "problem_id": problem.id,
                    "problem_type": problem.problem_type,
                    "max_gom_depth": max_depth,
                    **result.to_dict(),
                }
                output.write(json.dumps(row, ensure_ascii=False) + "\n")
                output.flush()
                print(
                    f"depth={max_depth}", problem.id, result.status, result.gap,
                    f"nodes={result.nodes}", f"decisions={result.gom_decisions}",
                    f"skips={result.gom_pregate_skips}",
                )


if __name__ == "__main__":
    main()
