from __future__ import annotations

import math
from typing import Any

from ..ir import OptimizationProblem
from ..lp_graph import SCIPLPGraphSnapshot
from ..state import SearchState
from ..trajectory import BranchStep, SolverTrajectory
from .scip_mapping import build_transformed_variable_map, resolve_original_variable_id


def collect_strong_branching_trajectory(
    problem: OptimizationProblem,
    *,
    time_limit_s: float = 30.0,
    max_steps: int = 512,
    strong_branch_iterations: int = 100,
) -> SolverTrajectory:
    """Run SCIP with a strong-branching teacher and record branch decisions.

    In addition to the original Optimization IR state, each recorded step stores
    SCIP's solver-native current LP bipartite graph. This captures presolved
    columns, active rows/cuts, LP solution/reduced costs, basis information, and
    the exact current coefficient matrix seen by the branching rule.
    """
    try:
        from pyscipopt import Model, Branchrule, SCIP_RESULT, SCIP_BRANCHDIR, quicksum
    except ImportError as e:
        raise RuntimeError(
            "PySCIPOpt is optional. Install a SCIP-compatible PySCIPOpt build, "
            "for example: pip install '.[scip]'"
        ) from e

    problem.validate()
    scip = Model(problem.id)
    scip.hideOutput(True)
    scip.setRealParam("limits/time", float(time_limit_s))

    var_map: dict[str, Any] = {}
    for v in problem.variables:
        vartype = {"binary": "B", "integer": "I", "continuous": "C"}[v.type]
        var_map[v.id] = scip.addVar(v.id, vtype=vartype, lb=v.lb, ub=v.ub, obj=problem.objective.get(v.id, 0.0))

    for c in problem.constraints:
        expr = quicksum(float(a) * var_map[vid] for vid, a in c.coefficients.items())
        if c.sense == "<=":
            scip.addCons(expr <= c.rhs, name=c.id)
        elif c.sense == ">=":
            scip.addCons(expr >= c.rhs, name=c.id)
        else:
            scip.addCons(expr == c.rhs, name=c.id)

    scip.setMinimize() if problem.sense == "min" else scip.setMaximize()
    steps: list[BranchStep] = []
    transformed_map: dict[str, str] = {}

    def safe_feature(fn, default: float = 0.0) -> float:
        try:
            value = float(fn())
            return value if math.isfinite(value) else default
        except Exception:
            return default

    def squash(value: float) -> float:
        value = safe_feature(lambda: value)
        return value / (1.0 + abs(value))

    class StrongBranchTeacher(Branchrule):
        def branchexeclp(self, allowaddcons):
            if len(steps) >= max_steps:
                return {"result": SCIP_RESULT.DIDNOTRUN}

            cands, cand_sols, cand_fracs, ncands, nprio, nimpl = self.model.getLPBranchCands()
            if nprio <= 0:
                return {"result": SCIP_RESULT.DIDNOTRUN}

            if not transformed_map:
                transformed_map.update(build_transformed_variable_map(self.model, var_map))
            candidate_ids = {
                i: original_id
                for i in range(nprio)
                if (original_id := resolve_original_variable_id(cands[i], transformed_map, var_map)) is not None
            }
            eligible = list(candidate_ids)
            if not eligible:
                return {"result": SCIP_RESULT.DIDNOTRUN}

            # Capture the exact solver-native LP graph before strong branching.
            col_features, edge_features, row_features, feature_map = self.model.getBipartiteGraphRepresentation(
                suppress_warnings=True
            )
            candidate_columns: dict[str, int] = {}
            for i in eligible:
                col = cands[i].getCol()
                if col is not None:
                    lp_pos = int(col.getLPPos())
                    if 0 <= lp_pos < len(col_features):
                        candidate_columns[candidate_ids[i]] = lp_pos

            reduced_cost = {
                candidate_ids[i]: safe_feature(lambda i=i: self.model.getVarRedcost(cands[i]))
                for i in eligible
            }
            pseudocost_down = {
                candidate_ids[i]: safe_feature(
                    lambda i=i: self.model.getVarPseudocost(cands[i], SCIP_BRANCHDIR.DOWNWARDS)
                )
                for i in eligible
            }
            pseudocost_up = {
                candidate_ids[i]: safe_feature(
                    lambda i=i: self.model.getVarPseudocost(cands[i], SCIP_BRANCHDIR.UPWARDS)
                )
                for i in eligible
            }

            node_no = self.model.getNNodes()
            lp_obj = self.model.getLPObjVal()
            scores = [float("-inf")] * nprio
            down_bounds = [None] * nprio
            up_bounds = [None] * nprio
            lperror = False

            self.model.startStrongbranch()
            try:
                for i in eligible:
                    cand = cands[i]
                    if self.model.getVarStrongbranchNode(cand) == node_no:
                        down, up, downvalid, upvalid, _, last_lp = self.model.getVarStrongbranchLast(cand)
                        if downvalid:
                            down_bounds[i] = down
                        if upvalid:
                            up_bounds[i] = up
                        down_gain = max(down - last_lp, 0.0) if downvalid else 0.0
                        up_gain = max(up - last_lp, 0.0) if upvalid else 0.0
                        scores[i] = float(self.model.getBranchScoreMultiple(cand, [down_gain, up_gain]))
                        continue

                    result = self.model.getVarStrongbranch(cand, strong_branch_iterations, idempotent=False)
                    down, up, downvalid, upvalid, downinf, upinf, downconflict, upconflict, err = result
                    if err:
                        lperror = True
                        break
                    if downvalid:
                        down_bounds[i] = down
                    if upvalid:
                        up_bounds[i] = up
                    down_gain = max(down - lp_obj, 0.0) if downvalid else 0.0
                    up_gain = max(up - lp_obj, 0.0) if upvalid else 0.0
                    scores[i] = float(self.model.getBranchScoreMultiple(cand, [down_gain, up_gain]))
            finally:
                self.model.endStrongbranch()

            if lperror:
                return {"result": SCIP_RESULT.DIDNOTRUN}

            scored = [i for i in eligible if math.isfinite(scores[i])]
            if not scored:
                return {"result": SCIP_RESULT.DIDNOTRUN}
            best_i = max(scored, key=lambda i: scores[i])
            chosen = cands[best_i]
            chosen_id = candidate_ids[best_i]
            chosen_sol = float(cand_sols[best_i])
            candidate_scores = {candidate_ids[i]: float(scores[i]) for i in scored}

            primal = safe_feature(self.model.getPrimalbound)
            dual = safe_feature(self.model.getDualbound, float(lp_obj))
            gap = safe_feature(self.model.getGap)
            if gap > 1e20:
                gap = 0.0
            current = self.model.getCurrentNode()
            depth = int(current.getDepth()) if current is not None else 0
            nodes = int(self.model.getNNodes())
            elapsed = float(self.model.getSolvingTime())

            state = SearchState(
                primal_bound=primal,
                dual_bound=dual,
                gap=gap,
                depth=depth,
                nodes=nodes,
                elapsed_s=elapsed,
                variable_lp={candidate_ids[i]: float(cand_sols[i]) for i in eligible},
                variable_fractionality={candidate_ids[i]: float(cand_fracs[i]) for i in eligible},
                variable_lb={candidate_ids[i]: float(cands[i].getLbLocal()) for i in eligible},
                variable_ub={candidate_ids[i]: float(cands[i].getUbLocal()) for i in eligible},
                branch_candidates={candidate_ids[i]: True for i in eligible},
                variable_reduced_cost=reduced_cost,
                variable_pseudocost_down=pseudocost_down,
                variable_pseudocost_up=pseudocost_up,
            )
            lp_graph = SCIPLPGraphSnapshot(
                col_features=[[float(x) for x in row] for row in col_features],
                edge_features=[[float(x) for x in edge] for edge in edge_features],
                row_features=[[float(x) for x in row] for row in row_features],
                candidate_columns=candidate_columns,
                feature_map={
                    str(group): {str(name): int(index) for name, index in mapping.items()}
                    for group, mapping in feature_map.items()
                },
                global_features=[
                    squash(primal), squash(dual), squash(gap), squash(depth), squash(nodes), squash(elapsed)
                ],
            )
            steps.append(
                BranchStep(
                    state=state,
                    chosen_variable=chosen_id,
                    chosen_value=chosen_sol,
                    score=float(scores[best_i]),
                    candidate_scores=candidate_scores,
                    lp_graph=lp_graph,
                )
            )

            down_child, eq_child, up_child = self.model.branchVarVal(chosen, chosen_sol)
            if self.model.allColsInLP():
                if down_child is not None and down_bounds[best_i] is not None:
                    self.model.updateNodeLowerbound(down_child, down_bounds[best_i])
                if up_child is not None and up_bounds[best_i] is not None:
                    self.model.updateNodeLowerbound(up_child, up_bounds[best_i])
            return {"result": SCIP_RESULT.BRANCHED}

    teacher = StrongBranchTeacher()
    scip.includeBranchrule(
        teacher,
        "gom_strong_branch_teacher",
        "Collect strong-branching decisions for GOM imitation learning",
        priority=10_000_000,
        maxdepth=-1,
        maxbounddist=1.0,
    )
    scip.optimize()

    status = str(scip.getStatus())
    objective = None
    try:
        if scip.getNSols() > 0:
            objective = float(scip.getObjVal())
    except Exception:
        pass
    return SolverTrajectory(problem=problem, solver="scip-strong-branch", steps=steps, final_status=status, final_objective=objective)
