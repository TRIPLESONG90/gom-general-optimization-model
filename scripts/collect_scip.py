from __future__ import annotations

import argparse
import json
import random

from gom.generators import generate_knapsack, generate_multidimensional_knapsack
from gom.solvers.scip_trajectory import collect_strong_branching_trajectory


def main():
    p = argparse.ArgumentParser(description="Collect SCIP strong-branching trajectories")
    p.add_argument("--count", type=int, default=20)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--time-limit", type=float, default=10.0)
    p.add_argument("--max-steps", type=int, default=128)
    p.add_argument("--out", default="scip_trajectories.jsonl")
    args = p.parse_args()

    rng = random.Random(args.seed)
    total_steps = 0
    with open(args.out, "w", encoding="utf-8") as f:
        for i in range(args.count):
            # Assignment is deliberately excluded here: its LP relaxation is
            # integral, so it is a poor source of branch-decision supervision.
            if i % 4 == 0:
                problem = generate_knapsack(rng, rng.randint(60, 140))
            else:
                problem = generate_multidimensional_knapsack(
                    rng,
                    n=rng.randint(50, 120),
                    m=rng.randint(4, 10),
                )
            traj = collect_strong_branching_trajectory(
                problem,
                time_limit_s=args.time_limit,
                max_steps=args.max_steps,
            )
            total_steps += len(traj.steps)
            f.write(json.dumps(traj.to_dict(), ensure_ascii=False) + "\n")
            print(i + 1, problem.problem_type, len(traj.steps), traj.final_status, traj.final_objective)
    print(f"total branch steps: {total_steps:,}")
    if total_steps == 0:
        print("WARNING: no branch decisions were collected; inspect SCIP presolve/settings and instance hardness")


if __name__ == "__main__":
    main()
