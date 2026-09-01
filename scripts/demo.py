from __future__ import annotations

import random
import torch

from gom.generators import generate_knapsack
from gom.graph import collate_graphs, featurize_problem
from gom.model import GOMConfig, GOMModel

problem = generate_knapsack(random.Random(42), n=16)
batch = collate_graphs([featurize_problem(problem)])
model = GOMModel(GOMConfig(d_model=128, n_heads=4, n_layers=2, d_ff=384))
model.eval()
with torch.no_grad():
    out = model(batch)
var_idx = int(out["variable_logits"].argmax(dim=-1)[0]) - 1
print("problem:", problem.id)
print("selected variable:", problem.variables[var_idx].id)
print("solver logits:", out["solver_logits"][0].tolist())
print("action logits:", out["action_logits"][0].tolist())
print("value:", out["value"][0].tolist())
