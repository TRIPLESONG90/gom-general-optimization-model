from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Dict, List, Literal

VarType = Literal["binary", "integer", "continuous"]
Sense = Literal["min", "max"]
ConstraintSense = Literal["<=", ">=", "=="]


@dataclass(slots=True)
class Variable:
    id: str
    type: VarType = "continuous"
    lb: float = 0.0
    ub: float = 1.0

    def validate(self) -> None:
        if self.type not in {"binary", "integer", "continuous"}:
            raise ValueError(f"unsupported variable type: {self.type}")
        if self.lb > self.ub:
            raise ValueError(f"{self.id}: lb > ub")
        if self.type == "binary" and (self.lb < 0 or self.ub > 1):
            raise ValueError(f"{self.id}: binary domain must be inside [0, 1]")


@dataclass(slots=True)
class LinearConstraint:
    id: str
    coefficients: Dict[str, float]
    sense: ConstraintSense
    rhs: float

    def validate(self, variable_ids: set[str]) -> None:
        if self.sense not in {"<=", ">=", "=="}:
            raise ValueError(f"{self.id}: unsupported sense {self.sense}")
        unknown = set(self.coefficients) - variable_ids
        if unknown:
            raise ValueError(f"{self.id}: unknown variables: {sorted(unknown)}")


@dataclass(slots=True)
class OptimizationProblem:
    id: str
    sense: Sense
    variables: List[Variable]
    objective: Dict[str, float]
    constraints: List[LinearConstraint] = field(default_factory=list)
    problem_type: str = "generic_milp"
    metadata: Dict[str, object] = field(default_factory=dict)

    def validate(self) -> None:
        if self.sense not in {"min", "max"}:
            raise ValueError(f"unsupported objective sense: {self.sense}")
        ids = [v.id for v in self.variables]
        if len(ids) != len(set(ids)):
            raise ValueError("duplicate variable ids")
        for var in self.variables:
            var.validate()
        variable_ids = set(ids)
        unknown_obj = set(self.objective) - variable_ids
        if unknown_obj:
            raise ValueError(f"objective has unknown variables: {sorted(unknown_obj)}")
        for con in self.constraints:
            con.validate(variable_ids)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "OptimizationProblem":
        variables = [Variable(**v) for v in data["variables"]]
        constraints = [LinearConstraint(**c) for c in data.get("constraints", [])]
        problem = cls(
            id=data["id"],
            sense=data["sense"],
            variables=variables,
            objective=dict(data.get("objective", {})),
            constraints=constraints,
            problem_type=data.get("problem_type", "generic_milp"),
            metadata=dict(data.get("metadata", {})),
        )
        problem.validate()
        return problem
