from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Dict


@dataclass(slots=True)
class SearchState:
    """Solver state projected into the GOM input space.

    All fields are optional-by-convention: missing dictionaries/default zeros mean
    "feature unavailable". This lets CP-SAT, SCIP, HiGHS and custom search loops
    share the same state schema.
    """

    primal_bound: float = 0.0
    dual_bound: float = 0.0
    gap: float = 0.0
    depth: int = 0
    nodes: int = 0
    elapsed_s: float = 0.0
    variable_lp: Dict[str, float] = field(default_factory=dict)
    variable_fractionality: Dict[str, float] = field(default_factory=dict)
    variable_incumbent: Dict[str, float] = field(default_factory=dict)
    variable_lb: Dict[str, float] = field(default_factory=dict)
    variable_ub: Dict[str, float] = field(default_factory=dict)
    branch_candidates: Dict[str, bool] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "SearchState":
        return cls(**data)
