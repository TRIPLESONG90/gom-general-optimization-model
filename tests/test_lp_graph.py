import torch

from gom.graph import BASE_FEATURE_DIM
from gom.lp_graph import SCIPLPGraphSnapshot, snapshot_to_graph_batch, snapshot_to_problem_graph
from gom.lp_graph_dataset import LPBranchSample, make_lp_branch_batch
from gom.model import GOMConfig, GOMModel
from gom.solvers.scip_lp_policy import predict_lp_branch_variable


def _snapshot():
    return SCIPLPGraphSnapshot(
        col_features=[
            [0, 1, 0, 0, -3.0, 1, 1, 0, 0, 0.25, 0.25, 0.0, 0, 1, 0, 0, 1.0, 0.5, 0],
            [0, 1, 0, 0, -2.0, 1, 1, 0, 0, 0.75, 0.25, 0.0, 0, 1, 0, 0, 0.0, 0.2, 0],
        ],
        edge_features=[[0, 0, 2.0], [1, 0, 5.0]],
        row_features=[[0, 1, 2, 0.2, 0.1, 1.0, 0, 0, -0.3, 0, 0, 1, 0, 0]],
        candidate_columns={"x0": 0, "x1": 1},
        global_features=[0.1, 0.2, 0.3],
    )


def test_snapshot_projects_to_gom_graph():
    graph = snapshot_to_problem_graph(_snapshot(), "test_milp")
    assert graph.x.shape == (4, BASE_FEATURE_DIM)  # global + 2 columns + 1 row
    assert graph.variable_mask.tolist() == [False, True, True, False]
    assert graph.edge_value[1, 3] != 0
    assert graph.edge_value[2, 3] != 0


def test_single_snapshot_fast_batch_matches_graph():
    graph = snapshot_to_problem_graph(_snapshot(), "test_milp")
    batch = snapshot_to_graph_batch(_snapshot(), "test_milp")
    assert torch.equal(batch.x[0], graph.x)
    assert torch.equal(batch.node_type[0], graph.node_type)
    assert torch.equal(batch.relation[0], graph.relation)
    assert torch.equal(batch.edge_value[0], graph.edge_value)
    assert torch.equal(batch.variable_mask[0], graph.variable_mask)
    assert not batch.padding_mask.any()


def test_lp_branch_batch_maps_expert_to_column():
    sample = LPBranchSample(
        problem_id="p0",
        problem_type="test_milp",
        snapshot=_snapshot(),
        expert_variable="x1",
        candidate_scores={"x0": 1.0, "x1": 2.0},
    )
    batch, targets = make_lp_branch_batch([sample])
    assert targets["variable"].item() == 2
    assert targets["candidate_mask"][0, 1].item()
    assert targets["candidate_mask"][0, 2].item()


def test_native_lp_prediction_returns_candidate():
    torch.manual_seed(0)
    model = GOMModel(GOMConfig(d_model=32, n_heads=4, n_layers=1, d_ff=64))
    chosen, confidence = predict_lp_branch_variable(
        model,
        _snapshot(),
        problem_type="test_milp",
    )
    assert chosen in {"x0", "x1"}
    assert 0.0 <= confidence <= 1.0
