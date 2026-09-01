import json
import random

import torch

from gom.generators import generate_knapsack
from gom.graph import featurize_problem
from gom.model import GOMConfig, GOMModel
from gom.solvers.scip_policy import predict_branch_variable
from gom.state import SearchState
from gom.trajectory import BranchStep, SolverTrajectory
from gom.trajectory_dataset import (
    branch_imitation_loss,
    load_branch_samples,
    make_branch_batch,
)


def _trajectory():
    problem = generate_knapsack(random.Random(11), 6)
    state = SearchState(
        primal_bound=float("inf"),
        dual_bound=float("-inf"),
        gap=float("inf"),
        depth=2,
        nodes=4,
        elapsed_s=0.1,
        variable_lp={"x0": 0.4, "x1": 0.6},
        variable_fractionality={"x0": 0.4, "x1": 0.4},
        branch_candidates={"x0": True, "x1": True},
    )
    return SolverTrajectory(
        problem=problem,
        solver="scip-strong-branch",
        steps=[BranchStep(state, "x1", 0.6, 1.25)],
        final_status="optimal",
        final_objective=42.0,
    )


def test_trajectory_roundtrip():
    trajectory = _trajectory()
    restored = SolverTrajectory.from_dict(trajectory.to_dict())
    assert restored.problem.id == trajectory.problem.id
    assert restored.steps[0].chosen_variable == "x1"
    assert restored.steps[0].state.branch_candidates["x0"] is True


def test_jsonl_dataset_and_branch_loss(tmp_path):
    trajectory = _trajectory()
    path = tmp_path / "traj.jsonl"
    path.write_text(json.dumps(trajectory.to_dict()) + "\n", encoding="utf-8")
    samples = load_branch_samples(path)
    assert len(samples) == 1
    batch, targets = make_branch_batch(samples)
    model = GOMModel(GOMConfig(d_model=64, n_heads=4, n_layers=1, d_ff=128))
    loss, _, logits = branch_imitation_loss(model, batch, targets)
    loss.backward()
    assert torch.isfinite(loss)
    assert logits[0, 1].isfinite()
    assert logits[0, 2].isfinite()


def test_nonfinite_solver_bounds_do_not_create_nan():
    trajectory = _trajectory()
    graph = featurize_problem(trajectory.problem, trajectory.steps[0].state)
    assert torch.isfinite(graph.x).all()


def test_policy_prediction_is_restricted_to_candidates():
    trajectory = _trajectory()
    model = GOMModel(GOMConfig(d_model=64, n_heads=4, n_layers=1, d_ff=128))
    chosen, confidence = predict_branch_variable(
        model,
        trajectory.problem,
        trajectory.steps[0].state,
        ["x0", "x1"],
    )
    assert chosen in {"x0", "x1"}
    assert 0.0 <= confidence <= 1.0
