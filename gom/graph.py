from __future__ import annotations

from dataclasses import dataclass
from typing import List

import torch

from .ir import OptimizationProblem
from .state import SearchState

GLOBAL, VARIABLE, CONSTRAINT = 0, 1, 2
NUM_NODE_TYPES = 3
REL_NONE, REL_SELF, REL_GLOBAL, REL_VAR_CON = 0, 1, 2, 3
NUM_RELATIONS = 4
BASE_FEATURE_DIM = 16


@dataclass
class ProblemGraph:
    x: torch.Tensor
    node_type: torch.Tensor
    relation: torch.Tensor
    variable_mask: torch.Tensor
    problem_type: str


@dataclass
class GraphBatch:
    x: torch.Tensor
    node_type: torch.Tensor
    relation: torch.Tensor
    padding_mask: torch.Tensor
    variable_mask: torch.Tensor
    problem_types: List[str]

    def to(self, device: torch.device | str) -> "GraphBatch":
        return GraphBatch(x=self.x.to(device), node_type=self.node_type.to(device), relation=self.relation.to(device), padding_mask=self.padding_mask.to(device), variable_mask=self.variable_mask.to(device), problem_types=self.problem_types)


def _safe_scale(v: float) -> float:
    return float(v / (1.0 + abs(v)))


def featurize_problem(problem: OptimizationProblem, state: SearchState | None = None) -> ProblemGraph:
    problem.validate()
    n_var = len(problem.variables)
    n_con = len(problem.constraints)
    n = 1 + n_var + n_con
    x = torch.zeros(n, BASE_FEATURE_DIM, dtype=torch.float32)
    node_type = torch.empty(n, dtype=torch.long)
    relation = torch.zeros(n, n, dtype=torch.long)
    variable_mask = torch.zeros(n, dtype=torch.bool)
    node_type[0] = GLOBAL
    x[0, 0] = 1.0 if problem.sense == "max" else -1.0
    x[0, 1] = _safe_scale(n_var)
    x[0, 2] = _safe_scale(n_con)
    if state is not None:
        x[0, 3] = _safe_scale(state.primal_bound)
        x[0, 4] = _safe_scale(state.dual_bound)
        x[0, 5] = _safe_scale(state.gap)
        x[0, 6] = _safe_scale(state.depth)
        x[0, 7] = _safe_scale(state.nodes)
        x[0, 8] = _safe_scale(state.elapsed_s)
    var_index = {v.id: i + 1 for i, v in enumerate(problem.variables)}
    for i, var in enumerate(problem.variables, start=1):
        node_type[i] = VARIABLE
        variable_mask[i] = True
        obj = problem.objective.get(var.id, 0.0)
        x[i, 0] = _safe_scale(obj)
        x[i, 1] = _safe_scale(var.lb)
        x[i, 2] = _safe_scale(var.ub)
        x[i, 3] = 1.0 if var.type == "binary" else 0.0
        x[i, 4] = 1.0 if var.type == "integer" else 0.0
        x[i, 5] = 1.0 if var.type == "continuous" else 0.0
        if state is not None:
            x[i, 9] = _safe_scale(state.variable_lp.get(var.id, 0.0))
            x[i, 10] = _safe_scale(state.variable_fractionality.get(var.id, 0.0))
            x[i, 11] = _safe_scale(state.variable_incumbent.get(var.id, 0.0))
            x[i, 12] = _safe_scale(state.variable_lb.get(var.id, var.lb))
            x[i, 13] = _safe_scale(state.variable_ub.get(var.id, var.ub))
            x[i, 14] = 1.0 if state.branch_candidates.get(var.id, False) else 0.0
    for j, con in enumerate(problem.constraints):
        idx = 1 + n_var + j
        node_type[idx] = CONSTRAINT
        x[idx, 0] = _safe_scale(con.rhs)
        x[idx, 1] = 1.0 if con.sense == "<=" else 0.0
        x[idx, 2] = 1.0 if con.sense == ">=" else 0.0
        x[idx, 3] = 1.0 if con.sense == "==" else 0.0
        coeffs = list(con.coefficients.values())
        if coeffs:
            x[idx, 4] = _safe_scale(sum(abs(c) for c in coeffs) / len(coeffs))
            x[idx, 5] = _safe_scale(max(abs(c) for c in coeffs))
        x[idx, 6] = _safe_scale(len(coeffs))
        for var_id, coeff in con.coefficients.items():
            vi = var_index[var_id]
            x[vi, 6] += _safe_scale(coeff)
            x[vi, 7] += _safe_scale(abs(coeff))
            x[vi, 8] += 1.0
            relation[vi, idx] = REL_VAR_CON
            relation[idx, vi] = REL_VAR_CON
    if n_con:
        x[1:1 + n_var, 8] /= max(1, n_con)
    relation[0, :] = REL_GLOBAL
    relation[:, 0] = REL_GLOBAL
    for i in range(n):
        relation[i, i] = REL_SELF
    return ProblemGraph(x=x, node_type=node_type, relation=relation, variable_mask=variable_mask, problem_type=problem.problem_type)


def collate_graphs(graphs: list[ProblemGraph]) -> GraphBatch:
    if not graphs:
        raise ValueError("graphs cannot be empty")
    b = len(graphs)
    max_n = max(g.x.shape[0] for g in graphs)
    feat = graphs[0].x.shape[1]
    x = torch.zeros(b, max_n, feat, dtype=torch.float32)
    node_type = torch.zeros(b, max_n, dtype=torch.long)
    relation = torch.zeros(b, max_n, max_n, dtype=torch.long)
    padding_mask = torch.ones(b, max_n, dtype=torch.bool)
    variable_mask = torch.zeros(b, max_n, dtype=torch.bool)
    for bi, g in enumerate(graphs):
        n = g.x.shape[0]
        x[bi, :n] = g.x
        node_type[bi, :n] = g.node_type
        relation[bi, :n, :n] = g.relation
        padding_mask[bi, :n] = False
        variable_mask[bi, :n] = g.variable_mask
    return GraphBatch(x=x, node_type=node_type, relation=relation, padding_mask=padding_mask, variable_mask=variable_mask, problem_types=[g.problem_type for g in graphs])
