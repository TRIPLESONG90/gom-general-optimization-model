from __future__ import annotations

import math
import time
from typing import Any

import torch

from ..ir import OptimizationProblem
from ..lp_graph import SCIPLPGraphSnapshot, snapshot_to_graph_batch
from ..model import GOMConfig, GOMModel
from ..trajectory_dataset import masked_branch_logits
from .scip_mapping import build_transformed_variable_map, resolve_original_variable_id
from .scip_policy import SCIPRunResult, _build_model, _import_scip, _result_from_model


def _finite(value: Any, default: float = 0.0) -> float:
    try:
        value = float(value)
    except Exception:
        return default
    return value if math.isfinite(value) else default


def _squash(value: float) -> float:
    value = _finite(value)
    return value / (1.0 + abs(value))


def load_lp_gom_checkpoint(path: str, device: str | torch.device = "cpu") -> GOMModel:
    payload = torch.load(path, map_location=device)
    representation = payload.get("input_representation")
    if representation not in (None, "scip_native_lp"):
        raise ValueError(f"checkpoint input representation is {representation!r}, expected 'scip_native_lp'")
    cfg = GOMConfig(**payload["config"])
    model = GOMModel(cfg).to(device)
    model.load_state_dict(payload["model"])
    model.eval()
    return model


def _predict_lp_branch_variable_timed(
    model: GOMModel,
    snapshot: SCIPLPGraphSnapshot,
    *,
    problem_type: str,
    device: str | torch.device = "cpu",
) -> tuple[str, float, float, float]:
    if not snapshot.candidate_columns:
        raise ValueError("LP snapshot has no mapped branch candidates")

    tensor_start = time.perf_counter()
    batch = snapshot_to_graph_batch(snapshot, problem_type, device=device)
    candidate_mask = torch.zeros_like(batch.variable_mask)
    inverse: dict[int, str] = {}
    for variable_id, col_idx in snapshot.candidate_columns.items():
        if 0 <= col_idx < len(snapshot.col_features):
            node_idx = 1 + col_idx
            candidate_mask[0, node_idx] = True
            inverse[node_idx] = variable_id
    if not inverse:
        raise ValueError("no branch candidates map to active LP columns")
    tensor_ms = (time.perf_counter() - tensor_start) * 1000.0

    model_start = time.perf_counter()
    with torch.inference_mode():
        output = model(batch)
        logits = masked_branch_logits(output["variable_logits"], candidate_mask)
        probs = torch.softmax(logits, dim=-1)
        node_idx = int(logits[0].argmax().item())
        confidence = float(probs[0, node_idx].item())
    model_ms = (time.perf_counter() - model_start) * 1000.0

    chosen = inverse.get(node_idx)
    if chosen is None:
        raise RuntimeError("model selected an unmapped LP column")
    return chosen, confidence, tensor_ms, model_ms


def predict_lp_branch_variable(
    model: GOMModel,
    snapshot: SCIPLPGraphSnapshot,
    *,
    problem_type: str,
    device: str | torch.device = "cpu",
) -> tuple[str, float]:
    chosen, confidence, _, _ = _predict_lp_branch_variable_timed(
        model,
        snapshot,
        problem_type=problem_type,
        device=device,
    )
    return chosen, confidence


def solve_with_gom_lp_branching(
    problem: OptimizationProblem,
    checkpoint: str,
    *,
    time_limit_s: float = 10.0,
    device: str | torch.device = "cpu",
    priority: int = 10_000_000,
    threads: int = 1,
    min_confidence: float = 0.0,
) -> SCIPRunResult:
    """Solve a MILP while GOM chooses LP branch variables from SCIP's current LP graph.

    The neural policy observes exactly the dynamic bipartite representation exposed
    by SCIP: active LP columns, active rows/cuts, LP/basis features, and current
    sparse coefficients. Latency is decomposed into solver graph extraction/list
    conversion, graph tensorization, and neural model scoring.
    """
    Branchrule, _, SCIP_RESULT, _ = _import_scip()
    if not 0.0 <= min_confidence <= 1.0:
        raise ValueError("min_confidence must be in [0, 1]")
    if str(device) == "cpu" and threads > 0:
        torch.set_num_threads(threads)

    scip, var_map = _build_model(problem, time_limit_s)
    gom = load_lp_gom_checkpoint(checkpoint, device)
    stats = {
        "decisions": 0,
        "fallbacks": 0,
        "abstentions": 0,
        "inference_ms": 0.0,
        "extract_ms": 0.0,
        "tensor_ms": 0.0,
        "model_ms": 0.0,
    }
    transformed_map: dict[str, str] = {}

    class GOMLPBranchRule(Branchrule):
        def branchexeclp(self, allowaddcons):
            try:
                cands, cand_sols, cand_fracs, ncands, nprio, nimpl = self.model.getLPBranchCands()
                if nprio <= 0:
                    stats["fallbacks"] += 1
                    return {"result": SCIP_RESULT.DIDNOTRUN}

                if not transformed_map:
                    transformed_map.update(build_transformed_variable_map(self.model, var_map))

                resolved = []
                for cand, sol in zip(cands[:nprio], cand_sols[:nprio]):
                    original_id = resolve_original_variable_id(cand, transformed_map, var_map)
                    if original_id is not None:
                        resolved.append((original_id, cand, float(sol)))
                if not resolved:
                    stats["fallbacks"] += 1
                    return {"result": SCIP_RESULT.DIDNOTRUN}

                total_start = time.perf_counter()
                extract_start = total_start
                col_features, edge_features, row_features, feature_map = self.model.getBipartiteGraphRepresentation(
                    suppress_warnings=True
                )
                candidate_columns: dict[str, int] = {}
                candidate_by_id: dict[str, Any] = {}
                candidate_sol: dict[str, float] = {}
                for original_id, cand, sol in resolved:
                    col = cand.getCol()
                    if col is None:
                        continue
                    lp_pos = int(col.getLPPos())
                    if 0 <= lp_pos < len(col_features):
                        candidate_columns[original_id] = lp_pos
                        candidate_by_id[original_id] = cand
                        candidate_sol[original_id] = sol
                if not candidate_columns:
                    stats["fallbacks"] += 1
                    return {"result": SCIP_RESULT.DIDNOTRUN}

                current = self.model.getCurrentNode()
                snapshot = SCIPLPGraphSnapshot(
                    col_features=[[float(x) for x in row] for row in col_features],
                    edge_features=[[float(x) for x in edge] for edge in edge_features],
                    row_features=[[float(x) for x in row] for row in row_features],
                    candidate_columns=candidate_columns,
                    feature_map={
                        str(group): {str(name): int(index) for name, index in mapping.items()}
                        for group, mapping in feature_map.items()
                    },
                    global_features=[
                        _squash(_finite(self.model.getPrimalbound())),
                        _squash(_finite(self.model.getDualbound())),
                        _squash(_finite(self.model.getGap())),
                        _squash(int(current.getDepth()) if current is not None else 0),
                        _squash(int(self.model.getNNodes())),
                        _squash(float(self.model.getSolvingTime())),
                    ],
                )
                extract_ms = (time.perf_counter() - extract_start) * 1000.0

                chosen_id, confidence, tensor_ms, model_ms = _predict_lp_branch_variable_timed(
                    gom,
                    snapshot,
                    problem_type=problem.problem_type,
                    device=device,
                )
                total_ms = (time.perf_counter() - total_start) * 1000.0
                stats["inference_ms"] += total_ms
                stats["extract_ms"] += extract_ms
                stats["tensor_ms"] += tensor_ms
                stats["model_ms"] += model_ms

                if confidence < min_confidence:
                    stats["abstentions"] += 1
                    return {"result": SCIP_RESULT.DIDNOTRUN}

                chosen = candidate_by_id.get(chosen_id)
                if chosen is None:
                    stats["fallbacks"] += 1
                    return {"result": SCIP_RESULT.DIDNOTRUN}
                self.model.branchVarVal(chosen, candidate_sol[chosen_id])
                stats["decisions"] += 1
                return {"result": SCIP_RESULT.BRANCHED}
            except Exception:
                stats["fallbacks"] += 1
                return {"result": SCIP_RESULT.DIDNOTRUN}

    rule = GOMLPBranchRule()
    scip.includeBranchrule(
        rule,
        "gom_native_lp_branch_policy",
        "GOM branch policy using SCIP native current-LP graph",
        priority=priority,
        maxdepth=-1,
        maxbounddist=1.0,
    )
    scip.optimize()
    return _result_from_model(
        scip,
        "gom-lp-hybrid" if min_confidence > 0 else "gom-lp",
        gom_decisions=int(stats["decisions"]),
        gom_fallbacks=int(stats["fallbacks"]),
        gom_abstentions=int(stats["abstentions"]),
        gom_inference_ms=float(stats["inference_ms"]),
        gom_extract_ms=float(stats["extract_ms"]),
        gom_tensor_ms=float(stats["tensor_ms"]),
        gom_model_ms=float(stats["model_ms"]),
    )
