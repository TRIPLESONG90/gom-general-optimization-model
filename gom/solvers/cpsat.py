from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

from ..ir import OptimizationProblem


@dataclass
class SolverResult:
    status: str
    objective: float | None
    values: Dict[str, float]
    wall_time: float | None = None


def solve(problem: OptimizationProblem, time_limit_s: float = 10.0) -> SolverResult:
    try:
        from ortools.sat.python import cp_model
    except ImportError as e:
        raise RuntimeError("OR-Tools is optional. Install with: pip install '.[solver]'") from e
    problem.validate()
    model = cp_model.CpModel()
    vars_ = {}
    for v in problem.variables:
        if v.type == "continuous":
            raise ValueError("CP-SAT adapter does not support continuous variables")
        vars_[v.id] = model.new_int_var(int(round(v.lb)), int(round(v.ub)), v.id)
    for c in problem.constraints:
        terms = sum(int(round(a)) * vars_[vid] for vid, a in c.coefficients.items())
        rhs = int(round(c.rhs))
        if c.sense == "<=": model.add(terms <= rhs)
        elif c.sense == ">=": model.add(terms >= rhs)
        else: model.add(terms == rhs)
    obj = sum(int(round(a)) * vars_[vid] for vid, a in problem.objective.items())
    model.minimize(obj) if problem.sense == "min" else model.maximize(obj)
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit_s
    status_code = solver.solve(model)
    status = solver.status_name(status_code)
    feasible = status_code in (cp_model.OPTIMAL, cp_model.FEASIBLE)
    return SolverResult(status=status, objective=float(solver.objective_value) if feasible else None, values={vid: float(solver.value(v)) for vid, v in vars_.items()} if feasible else {}, wall_time=float(solver.wall_time))
