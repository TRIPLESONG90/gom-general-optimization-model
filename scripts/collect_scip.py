from __future__ import annotations

import argparse
import json
import random

from gom.generators import generate_knapsack, generate_assignment
from gom.solvers.scip_trajectory import collect_strong_branching_trajectory


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--count", type=int, default=20)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--time-limit", type=float, default=10.0)
    p.add_argument("--max-steps", type=int, default=128)
    p.add_argument("--out", default="scip_trajectories.jsonl")
    args = p.parse_args()

    rng = random.Random(args.seed)
    with open(args.out, "w", encoding="utf-8") as f:
        for i in range(args.count):
            problem = generate_knapsack(rng, rng.randint(20, 60)) if i % 2 == 0 else generate_assignment(rng, rng.randint(5, 10))
            traj = collect_strong_branching_trajectory(problem, time_limit_s=args.time_limit, max_steps=args.max_steps)
            f.write(json.dumps(traj.to_dict(), ensure_ascii=False) + "\n")
            print(i + 1, problem.problem_type, len(traj.steps), traj.final_status, traj.final_objective)


if __name__ == "__main__":
    main()
