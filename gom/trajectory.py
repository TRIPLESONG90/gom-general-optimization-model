from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import List

from .ir import OptimizationProblem
from .state import SearchState


@dataclass(slots=True)
class BranchStep:
    state: SearchState
    chosen_variable: str
    chosen_value: float
    score: float

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(slots=True)
class SolverTrajectory:
    problem: OptimizationProblem
    solver: str
    steps: List[BranchStep]
    final_status: str = "unknown"
    final_objective: float | None = None

    def to_dict(self) -> dict:
        return {"problem": self.problem.to_dict(), "solver": self.solver, "steps": [s.to_dict() for s in self.steps], "final_status": self.final_status, "final_objective": self.final_objective}
