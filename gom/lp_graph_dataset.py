from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Sequence

import torch

from .graph import collate_graphs
from .lp_graph import SCIPLPGraphSnapshot, snapshot_to_problem_graph
from .trajectory_dataset import iter_trajectories_jsonl


@dataclass(slots=True)
class LPBranchSample:
    problem_id: str
    problem_type: str
    snapshot: SCIPLPGraphSnapshot
    expert_variable: str
    candidate_scores: dict[str, float]


def load_lp_branch_samples(
    paths: str | Path | Sequence[str | Path],
    *,
    max_samples: int | None = None,
    strict: bool = True,
) -> list[LPBranchSample]:
    samples: list[LPBranchSample] = []
    for trajectory in iter_trajectories_jsonl(paths):
        for step in trajectory.steps:
            snapshot = step.lp_graph
            if snapshot is None:
                continue
            if step.chosen_variable not in snapshot.candidate_columns:
                if strict:
                    raise ValueError(
                        f"{trajectory.problem.id}: expert variable {step.chosen_variable!r} "
                        "is missing from LP candidate column mapping"
                    )
                continue
            candidate_scores = {
                variable_id: float(score)
                for variable_id, score in step.candidate_scores.items()
                if variable_id in snapshot.candidate_columns and math.isfinite(float(score))
            }
            samples.append(
                LPBranchSample(
                    problem_id=trajectory.problem.id,
                    problem_type=trajectory.problem.problem_type,
                    snapshot=snapshot,
                    expert_variable=step.chosen_variable,
                    candidate_scores=candidate_scores,
                )
            )
            if max_samples is not None and len(samples) >= max_samples:
                return samples
    return samples


def make_lp_branch_batch(
    samples: Sequence[LPBranchSample],
    device: str | torch.device = "cpu",
):
    if not samples:
        raise ValueError("samples cannot be empty")

    graphs = [snapshot_to_problem_graph(sample.snapshot, sample.problem_type) for sample in samples]
    batch = collate_graphs(graphs).to(device)
    candidate_mask = torch.zeros_like(batch.variable_mask)
    score_mask = torch.zeros_like(batch.variable_mask)
    score_targets = torch.zeros_like(batch.x[:, :, 0])
    target_indices: list[int] = []

    for bi, sample in enumerate(samples):
        snapshot = sample.snapshot
        n_cols = len(snapshot.col_features)
        for variable_id, col_idx in snapshot.candidate_columns.items():
            if 0 <= col_idx < n_cols:
                candidate_mask[bi, 1 + col_idx] = True
        expert_col = snapshot.candidate_columns[sample.expert_variable]
        if not candidate_mask[bi, 1 + expert_col]:
            raise ValueError("expert LP column is not an active branch candidate")
        target_indices.append(1 + expert_col)

        for variable_id, score in sample.candidate_scores.items():
            col_idx = snapshot.candidate_columns.get(variable_id)
            if col_idx is None or not (0 <= col_idx < n_cols):
                continue
            node_idx = 1 + col_idx
            if candidate_mask[bi, node_idx]:
                score_mask[bi, node_idx] = True
                score_targets[bi, node_idx] = float(score)

    return batch, {
        "variable": torch.tensor(target_indices, dtype=torch.long, device=device),
        "candidate_mask": candidate_mask.to(device),
        "score_mask": score_mask.to(device),
        "score_targets": score_targets.to(device),
    }
