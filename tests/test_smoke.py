import random
import torch

from gom.generators import generate_knapsack, solve_knapsack_dp, generate_assignment, solve_assignment_exact
from gom.graph import featurize_problem, collate_graphs
from gom.model import GOMConfig, GOMModel
from gom.training import make_batch, compute_loss


def test_ir_graph_model_forward():
    p1 = generate_knapsack(random.Random(1), 12)
    p2 = generate_assignment(random.Random(2), 4)
    batch = collate_graphs([featurize_problem(p1), featurize_problem(p2)])
    model = GOMModel(GOMConfig(d_model=64, n_heads=4, n_layers=2, d_ff=128))
    out = model(batch)
    assert out["solver_logits"].shape == (2, 4)
    assert out["action_logits"].shape == (2, 5)
    assert out["variable_logits"].shape[0] == 2
    assert torch.isfinite(out["value"]).all()


def test_training_loss_backward():
    s1 = solve_knapsack_dp(generate_knapsack(random.Random(3), 10))
    s2 = solve_assignment_exact(generate_assignment(random.Random(4), 4))
    batch, targets = make_batch([s1, s2])
    model = GOMModel(GOMConfig(d_model=64, n_heads=4, n_layers=1, d_ff=128))
    loss, _, _ = compute_loss(model, batch, targets)
    loss.backward()
    assert torch.isfinite(loss)


def test_default_model_is_roughly_30m():
    model = GOMModel()
    n = model.parameter_count()
    assert 25_000_000 <= n <= 45_000_000, n


def test_dynamic_search_state_features():
    from gom.state import SearchState
    p = generate_knapsack(random.Random(5), 8)
    st = SearchState(primal_bound=100, dual_bound=90, gap=0.1, depth=3, nodes=12, elapsed_s=0.5, variable_lp={"x0": 0.5}, variable_fractionality={"x0": 0.5}, branch_candidates={"x0": True})
    g = featurize_problem(p, st)
    assert g.x.shape[1] == 16
    assert g.x[1, 14].item() == 1.0
