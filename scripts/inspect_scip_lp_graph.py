from __future__ import annotations

import argparse
import random

from gom.generators import generate_multidimensional_knapsack


def main():
    p = argparse.ArgumentParser(description="Inspect PySCIPOpt native bipartite LP graph inside a branch callback")
    p.add_argument("--seed", type=int, default=51000)
    p.add_argument("--n", type=int, default=60)
    p.add_argument("--constraints", type=int, default=6)
    args = p.parse_args()

    from pyscipopt import Branchrule, Model, SCIP_RESULT, quicksum

    problem = generate_multidimensional_knapsack(random.Random(args.seed), n=args.n, m=args.constraints)
    scip = Model("lp-graph-inspect")
    scip.hideOutput(True)
    scip.setRealParam("limits/time", 5.0)

    var_map = {}
    for v in problem.variables:
        var_map[v.id] = scip.addVar(v.id, vtype="B", lb=v.lb, ub=v.ub, obj=problem.objective.get(v.id, 0.0))
    for c in problem.constraints:
        expr = quicksum(float(a) * var_map[vid] for vid, a in c.coefficients.items())
        scip.addCons(expr <= c.rhs, name=c.id)
    scip.setMaximize()

    class InspectRule(Branchrule):
        seen = False

        def branchexeclp(self, allowaddcons):
            if self.seen:
                return {"result": SCIP_RESULT.DIDNOTRUN}
            self.seen = True
            cands, cand_sols, cand_fracs, ncands, nprio, nimpl = self.model.getLPBranchCands()
            print(f"branch candidates: ncands={ncands} nprio={nprio} nimpl={nimpl}")

            graph = self.model.getBipartiteGraphRepresentation(suppress_warnings=True)
            print("return_type", type(graph), "len", len(graph))
            for i, part in enumerate(graph):
                print(f"part[{i}] type={type(part)}")
                if hasattr(part, "shape"):
                    print("  shape", part.shape, "dtype", getattr(part, "dtype", None))
                elif isinstance(part, dict):
                    print("  keys", sorted(part.keys()))
                    for key, value in part.items():
                        if hasattr(value, "shape"):
                            print(f"    {key}: shape={value.shape} dtype={getattr(value, 'dtype', None)}")
                        elif isinstance(value, (list, tuple)):
                            print(f"    {key}: len={len(value)} sample={value[:8]}")
                        else:
                            print(f"    {key}: {type(value)} {value}")
                elif isinstance(part, (list, tuple)):
                    print("  len", len(part), "sample", part[:2])
                else:
                    print("  value", part)

            print("candidate column info")
            for cand in cands[: min(nprio, 8)]:
                col = cand.getCol()
                attrs = {
                    "name": cand.name,
                    "col_type": type(col).__name__ if col is not None else None,
                    "lp_pos": col.getLPPos() if col is not None and hasattr(col, "getLPPos") else None,
                    "basis": col.getBasisStatus() if col is not None and hasattr(col, "getBasisStatus") else None,
                }
                print(attrs)
            return {"result": SCIP_RESULT.DIDNOTRUN}

    rule = InspectRule()
    scip.includeBranchrule(rule, "inspect_lp_graph", "Inspect native SCIP LP graph", priority=10_000_000, maxdepth=-1, maxbounddist=1.0)
    scip.optimize()
    if not rule.seen:
        raise SystemExit("No branch callback was reached")


if __name__ == "__main__":
    main()
