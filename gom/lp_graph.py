from __future__ import annotations

from dataclasses import dataclass, asdict
import math
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


def _signed_squash(value: float) -> float:
    value = float(value)
    if not math.isfinite(value):
        return 0.0
    return value / (1.0 + abs(value))


# SCIP 10 / PySCIPOpt 6.2 native graph schema. Binary flags and already bounded
# values are left untouched; unbounded magnitude features are squashed to (-1, 1).
_DEFAULT_COL_SCALE = {
    "obj_coef", "red_cost", "best_incumbent_val", "avg_incumbent_val", "age"
}
_DEFAULT_ROW_SCALE = {"n_non_zeros", "bias", "norm", "dual_sol", "age"}


def _normalize_feature_rows(
    rows: list[list[float]],
    names: dict[str, int] | None,
    scale_names: set[str],
) -> torch.Tensor:
    if not rows:
        return torch.empty((0, 0), dtype=torch.float32)
    tensor = torch.tensor(rows, dtype=torch.float32)
    if names:
        indices = [index for name, index in names.items() if name in scale_names]
    else:
        indices = []
    for index in indices:
        if 0 <= index < tensor.shape[1]:
            values = tensor[:, index]
            finite = torch.isfinite(values)
            squashed = values / (1.0 + values.abs())
            tensor[:, index] = torch.where(finite, squashed, torch.zeros_like(values))
    tensor = torch.nan_to_num(tensor, nan=0.0, posinf=0.0, neginf=0.0)
    return tensor


def snapshot_to_problem_graph(snapshot: SCIPLPGraphSnapshot, problem_type: str = "scip_lp") -> ProblemGraph:
    """Project SCIP's dynamic LP graph into the existing GOM graph container.

    Magnitude-sensitive native features are semantically normalized while one-hot,
    basis-state, and already bounded LP features retain their original meaning.
    """
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
        global_values = [_signed_squash(v) if abs(float(v)) > 1.0 else float(v) for v in snapshot.global_features[:width]]
        x[0, :width] = torch.tensor(global_values, dtype=torch.float32)

    feature_map = snapshot.feature_map or {}
    if n_col:
        col = _normalize_feature_rows(
            snapshot.col_features,
            feature_map.get("col"),
            _DEFAULT_COL_SCALE,
        )
        width = min(BASE_FEATURE_DIM, col.shape[1])
        x[1:1 + n_col, :width] = col[:, :width]
        node_type[1:1 + n_col] = VARIABLE
        variable_mask[1:1 + n_col] = True

    row_offset = 1 + n_col
    if n_row:
        row = _normalize_feature_rows(
            snapshot.row_features,
            feature_map.get("row"),
            _DEFAULT_ROW_SCALE,
        )
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
        scaled = _signed_squash(coefficient)
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
