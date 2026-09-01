from __future__ import annotations

from typing import Any

from ..ir import OptimizationProblem
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

    This is intentionally expensive: it is a DATA GENERATOR, not the production
    inference path. The resulting `(problem, state) -> chosen_variable` records
    are behavior-cloning targets for GOM.

    Requires PySCIPOpt 6.x / SCIP 10.x or another compatible pair.
    """
    try:
        from pyscipopt import Model, Branchrule, SCIP_RESULT, quicksum
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
                        scores[i] = self.model.getBranchScoreMultiple(cand, [down_gain, up_gain])
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
                    scores[i] = self.model.getBranchScoreMultiple(cand, [down_gain, up_gain])
            finally:
                self.model.endStrongbranch()

            if lperror:
                return {"result": SCIP_RESULT.DIDNOTRUN}

            best_i = max(eligible, key=lambda i: scores[i])
            chosen = cands[best_i]
            chosen_id = candidate_ids[best_i]
            chosen_sol = float(cand_sols[best_i])

            try:
                primal = float(self.model.getPrimalbound())
            except Exception:
                primal = 0.0
            try:
                dual = float(self.model.getDualbound())
            except Exception:
                dual = float(lp_obj)
            try:
                gap = float(self.model.getGap())
                if gap > 1e20:
                    gap = 0.0
            except Exception:
                gap = 0.0

            state = SearchState(
                primal_bound=primal,
                dual_bound=dual,
                gap=gap,
                depth=int(self.model.getCurrentNode().getDepth()) if self.model.getCurrentNode() is not None else 0,
                nodes=int(self.model.getNNodes()),
                elapsed_s=float(self.model.getSolvingTime()),
                variable_lp={candidate_ids[i]: float(cand_sols[i]) for i in eligible},
                variable_fractionality={candidate_ids[i]: float(cand_fracs[i]) for i in eligible},
                variable_lb={candidate_ids[i]: float(cands[i].getLbLocal()) for i in eligible},
                variable_ub={candidate_ids[i]: float(cands[i].getUbLocal()) for i in eligible},
                branch_candidates={candidate_ids[i]: True for i in eligible},
            )
            steps.append(BranchStep(state, chosen_id, chosen_sol, float(scores[best_i])))

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
