from __future__ import annotations

from dataclasses import dataclass, asdict, field
from typing import List

from .ir import OptimizationProblem
from .state import SearchState


@dataclass(slots=True)
class BranchStep:
    state: SearchState
    chosen_variable: str
    chosen_value: float
    score: float
    candidate_scores: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "BranchStep":
        return cls(
            state=SearchState.from_dict(data["state"]),
            chosen_variable=data["chosen_variable"],
            chosen_value=float(data.get("chosen_value", 0.0)),
            score=float(data.get("score", 0.0)),
            candidate_scores={
                str(variable_id): float(score)
                for variable_id, score in data.get("candidate_scores", {}).items()
            },
        )


@dataclass(slots=True)
class SolverTrajectory:
    problem: OptimizationProblem
    solver: str
    steps: List[BranchStep]
    final_status: str = "unknown"
    final_objective: float | None = None

    def to_dict(self) -> dict:
        return {
            "problem": self.problem.to_dict(),
            "solver": self.solver,
            "steps": [s.to_dict() for s in self.steps],
            "final_status": self.final_status,
            "final_objective": self.final_objective,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "SolverTrajectory":
        objective = data.get("final_objective")
        return cls(
            problem=OptimizationProblem.from_dict(data["problem"]),
            solver=str(data.get("solver", "unknown")),
            steps=[BranchStep.from_dict(s) for s in data.get("steps", [])],
            final_status=str(data.get("final_status", "unknown")),
            final_objective=None if objective is None else float(objective),
        )
