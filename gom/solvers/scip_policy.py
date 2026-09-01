from __future__ import annotations

from dataclasses import dataclass, asdict
import math
import time
from typing import Any, Literal

import torch

from ..graph import collate_graphs, featurize_problem
from ..ir import OptimizationProblem
from ..model import GOMConfig, GOMModel
from ..state import SearchState
from ..trajectory_dataset import masked_branch_logits
from .scip_mapping import build_transformed_variable_map, resolve_original_variable_id


@dataclass(slots=True)
class SCIPRunResult:
    policy: str
    status: str
    objective: float | None
    primal_bound: float | None
    dual_bound: float | None
    gap: float | None
    nodes: int
    wall_time_s: float
    gom_decisions: int = 0
    gom_fallbacks: int = 0
    gom_inference_ms: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)


def _optional_finite(value: Any) -> float | None:
    try:
        value = float(value)
    except Exception:
        return None
    return value if math.isfinite(value) else None


def _import_scip():
    try:
        from pyscipopt import Branchrule, Model, SCIP_RESULT, quicksum
    except ImportError as exc:
        raise RuntimeError("PySCIPOpt is optional. Install with: pip install -e '.[scip]'") from exc
    return Branchrule, Model, SCIP_RESULT, quicksum


def _build_model(problem: OptimizationProblem, time_limit_s: float):
    _, Model, _, quicksum = _import_scip()
    problem.validate()
    scip = Model(problem.id)
    scip.hideOutput(True)
    scip.setRealParam("limits/time", float(time_limit_s))

    var_map: dict[str, Any] = {}
    for v in problem.variables:
        vartype = {"binary": "B", "integer": "I", "continuous": "C"}[v.type]
        var_map[v.id] = scip.addVar(
            v.id,
            vtype=vartype,
            lb=v.lb,
            ub=v.ub,
            obj=problem.objective.get(v.id, 0.0),
        )

    for c in problem.constraints:
        expr = quicksum(float(a) * var_map[vid] for vid, a in c.coefficients.items())
        if c.sense == "<=":
            scip.addCons(expr <= c.rhs, name=c.id)
        elif c.sense == ">=":
            scip.addCons(expr >= c.rhs, name=c.id)
        else:
            scip.addCons(expr == c.rhs, name=c.id)

    scip.setMinimize() if problem.sense == "min" else scip.setMaximize()
    return scip, var_map


def _result_from_model(scip, policy: str, **extra) -> SCIPRunResult:
    objective = None
    try:
        if scip.getNSols() > 0:
            objective = _optional_finite(scip.getObjVal())
    except Exception:
        pass
    try:
        primal = _optional_finite(scip.getPrimalbound())
    except Exception:
        primal = None
    try:
        dual = _optional_finite(scip.getDualbound())
    except Exception:
        dual = None
    try:
        gap = _optional_finite(scip.getGap())
    except Exception:
        gap = None
    return SCIPRunResult(
        policy=policy,
        status=str(scip.getStatus()),
        objective=objective,
        primal_bound=primal,
        dual_bound=dual,
        gap=gap,
        nodes=int(scip.getNNodes()),
        wall_time_s=float(scip.getSolvingTime()),
        **extra,
    )


def solve_scip_baseline(
    problem: OptimizationProblem,
    *,
    time_limit_s: float = 10.0,
    policy: Literal["default", "fullstrong"] = "default",
) -> SCIPRunResult:
    scip, _ = _build_model(problem, time_limit_s)
    if policy == "fullstrong":
        scip.setIntParam("branching/fullstrong/priority", 10_000_000)
        scip.setIntParam("branching/fullstrong/maxdepth", -1)
    elif policy != "default":
        raise ValueError(f"unknown SCIP baseline policy: {policy}")
    scip.optimize()
    return _result_from_model(scip, policy)


def load_gom_checkpoint(path: str, device: str | torch.device = "cpu") -> GOMModel:
    payload = torch.load(path, map_location=device)
    cfg = GOMConfig(**payload["config"])
    model = GOMModel(cfg).to(device)
    model.load_state_dict(payload["model"])
    model.eval()
    return model


def predict_branch_variable(
    model: GOMModel,
    problem: OptimizationProblem,
    state: SearchState,
    candidate_ids: list[str],
    *,
    device: str | torch.device = "cpu",
) -> tuple[str, float]:
    if not candidate_ids:
        raise ValueError("candidate_ids cannot be empty")

    variable_ids = [v.id for v in problem.variables]
    known = [v for v in candidate_ids if v in variable_ids]
    if not known:
        raise ValueError("none of the branch candidates exist in the OptimizationProblem")

    batch = collate_graphs([featurize_problem(problem, state)]).to(device)
    candidate_mask = torch.zeros_like(batch.variable_mask)
    for var_id in known:
        candidate_mask[0, 1 + variable_ids.index(var_id)] = True

    with torch.inference_mode():
        output = model(batch)
        logits = masked_branch_logits(output["variable_logits"], candidate_mask)
        probs = torch.softmax(logits, dim=-1)
        node_index = int(logits[0].argmax().item())
        confidence = float(probs[0, node_index].item())

    if node_index <= 0 or node_index > len(variable_ids):
        raise RuntimeError("model selected a non-variable node")
    return variable_ids[node_index - 1], confidence


def solve_with_gom_branching(
    problem: OptimizationProblem,
    checkpoint: str,
    *,
    time_limit_s: float = 10.0,
    device: str | torch.device = "cpu",
    priority: int = 10_000_000,
) -> SCIPRunResult:
    Branchrule, _, SCIP_RESULT, _ = _import_scip()
    try:
        from pyscipopt import SCIP_BRANCHDIR
    except ImportError as exc:
        raise RuntimeError("PySCIPOpt SCIP_BRANCHDIR is required") from exc

    scip, var_map = _build_model(problem, time_limit_s)
    gom = load_gom_checkpoint(checkpoint, device)
    stats = {"decisions": 0, "fallbacks": 0, "inference_ms": 0.0}
    transformed_map: dict[str, str] = {}

    def safe_feature(fn, default: float = 0.0) -> float:
        try:
            value = float(fn())
            return value if math.isfinite(value) else default
        except Exception:
            return default

    class GOMBranchRule(Branchrule):
        def branchexeclp(self, allowaddcons):
            try:
                cands, cand_sols, cand_fracs, ncands, nprio, nimpl = self.model.getLPBranchCands()
                if nprio <= 0:
                    stats["fallbacks"] += 1
                    return {"result": SCIP_RESULT.DIDNOTRUN}

                if not transformed_map:
                    transformed_map.update(build_transformed_variable_map(self.model, var_map))

                resolved = []
                for cand, sol, frac in zip(cands[:nprio], cand_sols[:nprio], cand_fracs[:nprio]):
                    original_id = resolve_original_variable_id(cand, transformed_map, var_map)
                    if original_id is not None:
                        resolved.append((original_id, cand, float(sol), float(frac)))
                if not resolved:
                    stats["fallbacks"] += 1
                    return {"result": SCIP_RESULT.DIDNOTRUN}

                candidate_ids = [original_id for original_id, _, _, _ in resolved]
                candidate_by_id = {original_id: cand for original_id, cand, _, _ in resolved}
                candidate_sol = {original_id: sol for original_id, _, sol, _ in resolved}

                try:
                    primal = float(self.model.getPrimalbound())
                except Exception:
                    primal = 0.0
                try:
                    dual = float(self.model.getDualbound())
                except Exception:
                    dual = 0.0
                try:
                    gap = float(self.model.getGap())
                except Exception:
                    gap = 0.0

                current = self.model.getCurrentNode()
                state = SearchState(
                    primal_bound=primal,
                    dual_bound=dual,
                    gap=gap,
                    depth=int(current.getDepth()) if current is not None else 0,
                    nodes=int(self.model.getNNodes()),
                    elapsed_s=float(self.model.getSolvingTime()),
                    variable_lp={original_id: sol for original_id, _, sol, _ in resolved},
                    variable_fractionality={original_id: frac for original_id, _, _, frac in resolved},
                    variable_lb={original_id: float(cand.getLbLocal()) for original_id, cand, _, _ in resolved},
                    variable_ub={original_id: float(cand.getUbLocal()) for original_id, cand, _, _ in resolved},
                    branch_candidates={original_id: True for original_id, _, _, _ in resolved},
                    variable_reduced_cost={
                        original_id: safe_feature(lambda cand=cand: self.model.getVarRedcost(cand))
                        for original_id, cand, _, _ in resolved
                    },
                    variable_pseudocost_down={
                        original_id: safe_feature(
                            lambda cand=cand: self.model.getVarPseudocost(cand, SCIP_BRANCHDIR.DOWNWARDS)
                        )
                        for original_id, cand, _, _ in resolved
                    },
                    variable_pseudocost_up={
                        original_id: safe_feature(
                            lambda cand=cand: self.model.getVarPseudocost(cand, SCIP_BRANCHDIR.UPWARDS)
                        )
                        for original_id, cand, _, _ in resolved
                    },
                )

                start = time.perf_counter()
                chosen_id, _ = predict_branch_variable(gom, problem, state, candidate_ids, device=device)
                stats["inference_ms"] += (time.perf_counter() - start) * 1000.0

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

    rule = GOMBranchRule()
    scip.includeBranchrule(
        rule,
        "gom_branch_policy",
        "GOM learned LP branching policy",
        priority=priority,
        maxdepth=-1,
        maxbounddist=1.0,
    )
    scip.optimize()
    return _result_from_model(
        scip,
        "gom",
        gom_decisions=int(stats["decisions"]),
        gom_fallbacks=int(stats["fallbacks"]),
        gom_inference_ms=float(stats["inference_ms"]),
    )
