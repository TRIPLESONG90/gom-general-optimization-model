from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Iterator, Sequence

import torch
import torch.nn.functional as F

from .graph import collate_graphs, featurize_problem
from .ir import OptimizationProblem
from .state import SearchState
from .trajectory import SolverTrajectory


@dataclass(slots=True)
class BranchSample:
    problem: OptimizationProblem
    state: SearchState
    expert_variable: str
    chosen_value: float
    score: float
    solver: str


def _normalize_paths(paths: str | Path | Sequence[str | Path]) -> list[Path]:
    if isinstance(paths, (str, Path)):
        return [Path(paths)]
    return [Path(p) for p in paths]


def iter_trajectories_jsonl(
    paths: str | Path | Sequence[str | Path],
) -> Iterator[SolverTrajectory]:
    for path in _normalize_paths(paths):
        with path.open("r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    yield SolverTrajectory.from_dict(json.loads(line))
                except Exception as exc:
                    raise ValueError(
                        f"{path}:{line_no}: invalid trajectory: {exc}"
                    ) from exc


def iter_branch_samples(
    paths: str | Path | Sequence[str | Path],
    *,
    strict: bool = True,
) -> Iterator[BranchSample]:
    for trajectory in iter_trajectories_jsonl(paths):
        variable_ids = {v.id for v in trajectory.problem.variables}
        for step in trajectory.steps:
            if step.chosen_variable not in variable_ids:
                if strict:
                    raise ValueError(
                        f"{trajectory.problem.id}: unknown expert variable "
                        f"{step.chosen_variable!r}"
                    )
                continue

            candidates = {
                var_id
                for var_id, enabled in step.state.branch_candidates.items()
                if enabled
            }
            if candidates and step.chosen_variable not in candidates:
                if strict:
                    raise ValueError(
                        f"{trajectory.problem.id}: expert variable "
                        f"{step.chosen_variable!r} is not an LP branch candidate"
                    )
                continue

            yield BranchSample(
                problem=trajectory.problem,
                state=step.state,
                expert_variable=step.chosen_variable,
                chosen_value=step.chosen_value,
                score=step.score,
                solver=trajectory.solver,
            )


def load_branch_samples(
    paths: str | Path | Sequence[str | Path],
    *,
    max_samples: int | None = None,
    strict: bool = True,
) -> list[BranchSample]:
    samples: list[BranchSample] = []
    for sample in iter_branch_samples(paths, strict=strict):
        samples.append(sample)
        if max_samples is not None and len(samples) >= max_samples:
            break
    return samples


def make_branch_batch(
    samples: Sequence[BranchSample],
    device: str | torch.device = "cpu",
):
    if not samples:
        raise ValueError("samples cannot be empty")

    graphs = [featurize_problem(s.problem, s.state) for s in samples]
    batch = collate_graphs(graphs).to(device)
    candidate_mask = torch.zeros_like(batch.variable_mask)
    targets: list[int] = []

    for batch_index, sample in enumerate(samples):
        variable_ids = [v.id for v in sample.problem.variables]
        expert_index = 1 + variable_ids.index(sample.expert_variable)
        targets.append(expert_index)

        candidates = [
            var_id
            for var_id, enabled in sample.state.branch_candidates.items()
            if enabled and var_id in variable_ids
        ]
        if candidates:
            for var_id in candidates:
                candidate_mask[
                    batch_index, 1 + variable_ids.index(var_id)
                ] = True
        else:
            candidate_mask[batch_index] = batch.variable_mask[batch_index]

    return batch, {
        "variable": torch.tensor(targets, dtype=torch.long, device=device),
        "candidate_mask": candidate_mask.to(device),
    }


def masked_branch_logits(
    variable_logits: torch.Tensor,
    candidate_mask: torch.Tensor,
) -> torch.Tensor:
    if variable_logits.shape != candidate_mask.shape:
        raise ValueError("logit/mask shape mismatch")
    if torch.any(candidate_mask.sum(dim=1) == 0):
        raise ValueError("every sample must have at least one branch candidate")
    return variable_logits.masked_fill(~candidate_mask, -1e9)


def branch_imitation_loss(model, batch, targets):
    output = model(batch)
    logits = masked_branch_logits(
        output["variable_logits"], targets["candidate_mask"]
    )
    loss = F.cross_entropy(logits, targets["variable"])
    return loss, output, logits


def branch_top1_accuracy(logits: torch.Tensor, targets) -> torch.Tensor:
    return (
        logits.argmax(dim=-1) == targets["variable"]
    ).float().mean()
