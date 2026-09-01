from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any

import torch

from .graph import (
    GLOBAL,
    VARIABLE,
    CONSTRAINT,
    REL_GLOBAL,
    REL_SELF,
    REL_VAR_CON,
    BASE_FEATURE_DIM,
    ProblemGraph,
)


@dataclass(slots=True)
class SCIPLPGraphSnapshot:
    """Solver-native current LP bipartite graph returned by PySCIPOpt.

    PySCIPOpt 6.2.x exposes 19 column features, 14 row features, and sparse
    edges `[col_idx, row_idx, coefficient]`. Candidate column indices refer to
    the LP column ordering and therefore map directly to `col_features` rows.
    """

    col_features: list[list[float]]
    edge_features: list[list[float]]
    row_features: list[list[float]]
    candidate_columns: dict[str, int]
    feature_map: dict[str, dict[str, int]] | None = None
    global_features: list[float] | None = None

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SCIPLPGraphSnapshot":
        return cls(
            col_features=[[float(x) for x in row] for row in data.get("col_features", [])],
            edge_features=[[float(x) for x in row] for row in data.get("edge_features", [])],
            row_features=[[float(x) for x in row] for row in data.get("row_features", [])],
            candidate_columns={str(k): int(v) for k, v in data.get("candidate_columns", {}).items()},
            feature_map=data.get("feature_map"),
            global_features=[float(x) for x in data.get("global_features", [])] if data.get("global_features") is not None else None,
        )


def snapshot_to_problem_graph(snapshot: SCIPLPGraphSnapshot, problem_type: str = "scip_lp") -> ProblemGraph:
    """Project SCIP's dynamic LP graph into the existing GOM graph container."""
    n_col = len(snapshot.col_features)
    n_row = len(snapshot.row_features)
    n = 1 + n_col + n_row

    x = torch.zeros(n, BASE_FEATURE_DIM, dtype=torch.float32)
    node_type = torch.empty(n, dtype=torch.long)
    relation = torch.zeros(n, n, dtype=torch.long)
    edge_value = torch.zeros(n, n, dtype=torch.float32)
    variable_mask = torch.zeros(n, dtype=torch.bool)

    node_type[0] = GLOBAL
    if snapshot.global_features:
        width = min(BASE_FEATURE_DIM, len(snapshot.global_features))
        x[0, :width] = torch.tensor(snapshot.global_features[:width], dtype=torch.float32)

    if n_col:
        col = torch.tensor(snapshot.col_features, dtype=torch.float32)
        width = min(BASE_FEATURE_DIM, col.shape[1])
        x[1:1 + n_col, :width] = col[:, :width]
        node_type[1:1 + n_col] = VARIABLE
        variable_mask[1:1 + n_col] = True

    row_offset = 1 + n_col
    if n_row:
        row = torch.tensor(snapshot.row_features, dtype=torch.float32)
        width = min(BASE_FEATURE_DIM, row.shape[1])
        x[row_offset:row_offset + n_row, :width] = row[:, :width]
        node_type[row_offset:row_offset + n_row] = CONSTRAINT

    for edge in snapshot.edge_features:
        if len(edge) < 3:
            continue
        col_idx, row_idx, coefficient = int(edge[0]), int(edge[1]), float(edge[2])
        if not (0 <= col_idx < n_col and 0 <= row_idx < n_row):
            continue
        ci = 1 + col_idx
        ri = row_offset + row_idx
        relation[ci, ri] = REL_VAR_CON
        relation[ri, ci] = REL_VAR_CON
        scaled = coefficient / (1.0 + abs(coefficient))
        edge_value[ci, ri] = scaled
        edge_value[ri, ci] = scaled

    relation[0, :] = REL_GLOBAL
    relation[:, 0] = REL_GLOBAL
    for i in range(n):
        relation[i, i] = REL_SELF

    return ProblemGraph(
        x=x,
        node_type=node_type,
        relation=relation,
        edge_value=edge_value,
        variable_mask=variable_mask,
        problem_type=problem_type,
    )
