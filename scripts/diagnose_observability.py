from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib

import torch

from gom.graph import featurize_problem
from gom.trajectory_dataset import load_branch_samples


def tensor_digest(*tensors: torch.Tensor, decimals: int | None = None) -> str:
    h = hashlib.sha256()
    for tensor in tensors:
        t = tensor.detach().cpu().contiguous()
        if decimals is not None and t.is_floating_point():
            scale = float(10 ** decimals)
            t = torch.round(t * scale) / scale
        h.update(str(t.dtype).encode())
        h.update(str(tuple(t.shape)).encode())
        h.update(t.numpy().tobytes())
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser(description="Check whether observed GOM states collide across different strong-branch labels")
    p.add_argument("trajectories", nargs="+")
    p.add_argument("--samples", type=int, default=24)
    args = p.parse_args()

    samples = load_branch_samples(args.trajectories, max_samples=args.samples)
    if not samples:
        raise SystemExit("no branch samples")

    exact: dict[str, list[str]] = defaultdict(list)
    rounded3: dict[str, list[str]] = defaultdict(list)
    dynamic: dict[str, list[str]] = defaultdict(list)

    for sample in samples:
        graph = featurize_problem(sample.problem, sample.state)
        exact[tensor_digest(graph.x, graph.relation, graph.edge_value, graph.variable_mask)].append(sample.expert_variable)
        rounded3[tensor_digest(graph.x, graph.relation, graph.edge_value, graph.variable_mask, decimals=3)].append(sample.expert_variable)
        # Dynamic state lives in global features 3:9 and variable features 9:15.
        dyn = torch.cat((graph.x[0, 3:9], graph.x[1:1 + len(sample.problem.variables), 9:15].flatten()))
        dynamic[tensor_digest(dyn, decimals=6)].append(sample.expert_variable)

    def report(name: str, groups: dict[str, list[str]]):
        duplicate_groups = [labels for labels in groups.values() if len(labels) > 1]
        conflicts = [labels for labels in duplicate_groups if len(set(labels)) > 1]
        conflicted_samples = sum(len(labels) for labels in conflicts)
        print(
            f"{name}: unique={len(groups)}/{len(samples)} duplicate_groups={len(duplicate_groups)} "
            f"conflict_groups={len(conflicts)} conflicted_samples={conflicted_samples}"
        )
        for labels in conflicts[:5]:
            print("  conflict labels:", labels)

    print(f"samples={len(samples)} problems={len({s.problem.id for s in samples})}")
    report("exact_full_input", exact)
    report("rounded_3dp_full_input", rounded3)
    report("dynamic_state_only", dynamic)


if __name__ == "__main__":
    main()
