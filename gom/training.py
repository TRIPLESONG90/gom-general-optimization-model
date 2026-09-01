from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
import torch.nn.functional as F

from .generators import LabeledProblem
from .graph import collate_graphs, featurize_problem
from .model import GOMModel


@dataclass
class LossWeights:
    solver: float = 0.25
    action: float = 0.25
    variable: float = 1.0
    value: float = 0.25


def make_batch(samples: Sequence[LabeledProblem], device: str | torch.device = "cpu"):
    graphs = [featurize_problem(s.problem) for s in samples]
    batch = collate_graphs(graphs).to(device)
    var_targets = []
    for sample in samples:
        ids = [v.id for v in sample.problem.variables]
        var_targets.append(1 + ids.index(sample.expert_variable))
    targets = {
        "solver": torch.tensor([s.solver_class for s in samples], dtype=torch.long, device=device),
        "action": torch.tensor([s.action_class for s in samples], dtype=torch.long, device=device),
        "variable": torch.tensor(var_targets, dtype=torch.long, device=device),
        "value": torch.tensor([[s.objective_value / (1.0 + abs(s.objective_value)), 0.0, 0.0] for s in samples], dtype=torch.float32, device=device),
    }
    return batch, targets


def compute_loss(model: GOMModel, batch, targets, weights: LossWeights = LossWeights()):
    out = model(batch)
    losses = {
        "solver": F.cross_entropy(out["solver_logits"], targets["solver"]),
        "action": F.cross_entropy(out["action_logits"], targets["action"]),
        "variable": F.cross_entropy(out["variable_logits"], targets["variable"]),
        "value": F.smooth_l1_loss(out["value"], targets["value"]),
    }
    total = (
        weights.solver * losses["solver"] +
        weights.action * losses["action"] +
        weights.variable * losses["variable"] +
        weights.value * losses["value"]
    )
    return total, losses, out
