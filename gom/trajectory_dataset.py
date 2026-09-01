from __future__ import annotations

from dataclasses import dataclass
import json
import math
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
    candidate_scores: dict[str, float]


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

            candidate_scores = {
                var_id: float(score)
                for var_id, score in step.candidate_scores.items()
                if var_id in variable_ids and math.isfinite(float(score))
            }
            if candidate_scores and step.chosen_variable not in candidate_scores:
                if strict:
                    raise ValueError(
                        f"{trajectory.problem.id}: expert variable "
                        f"{step.chosen_variable!r} has no strong-branch score"
                    )
                candidate_scores = {}

            yield BranchSample(
                problem=trajectory.problem,
                state=step.state,
                expert_variable=step.chosen_variable,
                chosen_value=step.chosen_value,
                score=step.score,
                solver=trajectory.solver,
                candidate_scores=candidate_scores,
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
    score_mask = torch.zeros_like(batch.variable_mask)
    score_targets = torch.zeros_like(batch.x[:, :, 0])
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

        for var_id, score in sample.candidate_scores.items():
            if var_id not in variable_ids or not math.isfinite(score):
                continue
            node_index = 1 + variable_ids.index(var_id)
            if candidate_mask[batch_index, node_index]:
                score_mask[batch_index, node_index] = True
                score_targets[batch_index, node_index] = float(score)

    return batch, {
        "variable": torch.tensor(targets, dtype=torch.long, device=device),
        "candidate_mask": candidate_mask.to(device),
        "score_mask": score_mask.to(device),
        "score_targets": score_targets.to(device),
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
    """v0.3 top-1 behavior-cloning loss kept for baseline compatibility."""
    output = model(batch)
    logits = masked_branch_logits(
        output["variable_logits"], targets["candidate_mask"]
    )
    loss = F.cross_entropy(logits, targets["variable"])
    return loss, output, logits


def listwise_strong_branch_loss(
    logits: torch.Tensor,
    score_targets: torch.Tensor,
    score_mask: torch.Tensor,
    *,
    temperature: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Distill a per-state ranking from SCIP strong-branch scores.

    Strong-branch score scales differ substantially across search states, so each
    state's finite candidate scores are standardized before turning them into a
    soft teacher distribution. Rows with fewer than two scored candidates do not
    contribute to the listwise loss.
    """
    if temperature <= 0:
        raise ValueError("temperature must be > 0")
    if logits.shape != score_targets.shape or logits.shape != score_mask.shape:
        raise ValueError("ranking tensor shape mismatch")

    valid_rows = score_mask.sum(dim=1) >= 2
    if not bool(valid_rows.any()):
        return logits.sum() * 0.0, valid_rows

    mask_f = score_mask.to(score_targets.dtype)
    counts = mask_f.sum(dim=1, keepdim=True).clamp_min(1.0)
    means = (score_targets * mask_f).sum(dim=1, keepdim=True) / counts
    centered = (score_targets - means) * mask_f
    variances = (centered.square().sum(dim=1, keepdim=True) / counts).clamp_min(1e-8)
    standardized = centered / variances.sqrt()

    teacher_logits = (standardized / temperature).masked_fill(~score_mask, -1e9)
    teacher_probs = torch.softmax(teacher_logits, dim=-1).detach()
    student_log_probs = torch.log_softmax(logits, dim=-1)
    per_row = -(teacher_probs * student_log_probs).sum(dim=-1)
    return per_row[valid_rows].mean(), valid_rows


def branch_policy_loss(
    model,
    batch,
    targets,
    *,
    ranking_weight: float = 0.75,
    ranking_temperature: float = 1.0,
):
    """Hybrid v0.4 objective: hard expert choice + listwise score distillation."""
    if not 0.0 <= ranking_weight <= 1.0:
        raise ValueError("ranking_weight must be in [0, 1]")

    output = model(batch)
    logits = masked_branch_logits(
        output["variable_logits"], targets["candidate_mask"]
    )
    top1 = F.cross_entropy(logits, targets["variable"])
    ranking, valid_rows = listwise_strong_branch_loss(
        logits,
        targets["score_targets"],
        targets["score_mask"],
        temperature=ranking_temperature,
    )
    effective_weight = ranking_weight if bool(valid_rows.any()) else 0.0
    total = (1.0 - effective_weight) * top1 + effective_weight * ranking
    components = {
        "top1": top1.detach(),
        "ranking": ranking.detach(),
        "ranking_rows": valid_rows.sum().detach(),
    }
    return total, components, output, logits


def branch_top1_accuracy(logits: torch.Tensor, targets) -> torch.Tensor:
    return (
        logits.argmax(dim=-1) == targets["variable"]
    ).float().mean()
