from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Iterable

from .ir import LinearConstraint, OptimizationProblem, Variable


@dataclass
class LabeledProblem:
    problem: OptimizationProblem
    solution: dict[str, int | float]
    objective_value: float
    expert_variable: str
    solver_class: int = 0
    action_class: int = 1  # default expert action: fix a high-confidence variable


def generate_knapsack(rng: random.Random, n: int = 20) -> OptimizationProblem:
    weights = [rng.randint(1, 30) for _ in range(n)]
    values = [rng.randint(1, 50) for _ in range(n)]
    capacity = max(1, int(sum(weights) * rng.uniform(0.25, 0.55)))
    variables = [Variable(f"x{i}", "binary", 0, 1) for i in range(n)]
    objective = {f"x{i}": float(values[i]) for i in range(n)}
    constraint = LinearConstraint(
        "capacity", {f"x{i}": float(weights[i]) for i in range(n)}, "<=", float(capacity)
    )
    return OptimizationProblem(
        id=f"kp-{rng.getrandbits(64):016x}",
        sense="max",
        variables=variables,
        objective=objective,
        constraints=[constraint],
        problem_type="knapsack",
        metadata={"weights": weights, "values": values, "capacity": capacity},
    )


def solve_knapsack_dp(problem: OptimizationProblem) -> LabeledProblem:
    weights = list(map(int, problem.metadata["weights"]))
    values = list(map(int, problem.metadata["values"]))
    capacity = int(problem.metadata["capacity"])
    n = len(weights)
    dp = [[0] * (capacity + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        w, v = weights[i - 1], values[i - 1]
        for c in range(capacity + 1):
            dp[i][c] = dp[i - 1][c]
            if w <= c:
                dp[i][c] = max(dp[i][c], dp[i - 1][c - w] + v)
    c = capacity
    sol = {f"x{i}": 0 for i in range(n)}
    for i in range(n, 0, -1):
        if dp[i][c] != dp[i - 1][c]:
            sol[f"x{i - 1}"] = 1
            c -= weights[i - 1]

    # Teacher variable: selected item with largest value-density, otherwise best density overall.
    density = [values[i] / weights[i] for i in range(n)]
    selected = [i for i in range(n) if sol[f"x{i}"] == 1]
    expert_i = max(selected or range(n), key=lambda i: density[i])
    return LabeledProblem(problem, sol, float(dp[n][capacity]), f"x{expert_i}")


def generate_assignment(rng: random.Random, n: int = 6) -> OptimizationProblem:
    costs = [[rng.randint(1, 100) for _ in range(n)] for _ in range(n)]
    variables = [Variable(f"x_{i}_{j}", "binary", 0, 1) for i in range(n) for j in range(n)]
    objective = {f"x_{i}_{j}": float(costs[i][j]) for i in range(n) for j in range(n)}
    constraints = []
    for i in range(n):
        constraints.append(LinearConstraint(f"worker_{i}", {f"x_{i}_{j}": 1.0 for j in range(n)}, "==", 1.0))
    for j in range(n):
        constraints.append(LinearConstraint(f"task_{j}", {f"x_{i}_{j}": 1.0 for i in range(n)}, "==", 1.0))
    return OptimizationProblem(
        id=f"assign-{rng.getrandbits(64):016x}", sense="min", variables=variables,
        objective=objective, constraints=constraints, problem_type="assignment", metadata={"costs": costs}
    )


def solve_assignment_exact(problem: OptimizationProblem) -> LabeledProblem:
    # Exact DP over task subsets: O(n 2^n), enough for synthetic MVP data.
    costs = problem.metadata["costs"]
    n = len(costs)
    inf = 10**18
    dp = [inf] * (1 << n)
    parent: list[tuple[int, int] | None] = [None] * (1 << n)
    dp[0] = 0
    for mask in range(1 << n):
        i = mask.bit_count()
        if i >= n or dp[mask] == inf:
            continue
        for j in range(n):
            if mask & (1 << j):
                continue
            nxt = mask | (1 << j)
            cand = dp[mask] + costs[i][j]
            if cand < dp[nxt]:
                dp[nxt] = cand
                parent[nxt] = (mask, j)
    sol = {f"x_{i}_{j}": 0 for i in range(n) for j in range(n)}
    mask = (1 << n) - 1
    chosen = []
    for i in range(n - 1, -1, -1):
        prev, j = parent[mask]  # type: ignore[misc]
        sol[f"x_{i}_{j}"] = 1
        chosen.append((i, j))
        mask = prev
    # Teacher variable: chosen assignment with strongest regret margin.
    def regret(pair: tuple[int, int]) -> int:
        i, j = pair
        alternatives = sorted(costs[i])
        return alternatives[1] - costs[i][j] if len(alternatives) > 1 else 0
    i, j = max(chosen, key=regret)
    return LabeledProblem(problem, sol, float(dp[-1]), f"x_{i}_{j}")


def generate_multidimensional_knapsack(
    rng: random.Random,
    n: int = 60,
    m: int = 6,
    density: float = 0.75,
) -> OptimizationProblem:
    """Generate a denser 0/1 packing MILP intended for branching experiments.

    Unlike assignment, its LP relaxation is generally fractional. The all-zero
    solution is feasible, which keeps synthetic generation robust while a
    positive objective creates a non-trivial packing search problem.
    """
    if n <= 0 or m <= 0:
        raise ValueError("n and m must be positive")
    variables = [Variable(f"x{i}", "binary", 0, 1) for i in range(n)]
    values = [rng.randint(10, 100) for _ in range(n)]
    objective = {f"x{i}": float(values[i]) for i in range(n)}
    constraints = []
    weights_by_constraint: list[list[int]] = []
    capacities: list[int] = []
    for j in range(m):
        weights = [rng.randint(1, 40) if rng.random() < density else 0 for _ in range(n)]
        if not any(weights):
            weights[rng.randrange(n)] = rng.randint(1, 40)
        total = sum(weights)
        capacity = max(1, int(total * rng.uniform(0.28, 0.48)))
        coeffs = {f"x{i}": float(w) for i, w in enumerate(weights) if w}
        constraints.append(LinearConstraint(f"pack_{j}", coeffs, "<=", float(capacity)))
        weights_by_constraint.append(weights)
        capacities.append(capacity)
    return OptimizationProblem(
        id=f"mkp-{rng.getrandbits(64):016x}",
        sense="max",
        variables=variables,
        objective=objective,
        constraints=constraints,
        problem_type="multidimensional_knapsack",
        metadata={
            "values": values,
            "weights": weights_by_constraint,
            "capacities": capacities,
        },
    )


def synthetic_stream(seed: int = 0, count: int = 1000) -> Iterable[LabeledProblem]:
    rng = random.Random(seed)
    for k in range(count):
        if k % 2 == 0:
            yield solve_knapsack_dp(generate_knapsack(rng, n=rng.randint(10, 28)))
        else:
            yield solve_assignment_exact(generate_assignment(rng, n=rng.randint(4, 8)))
