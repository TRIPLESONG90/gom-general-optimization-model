import torch

from gom.trajectory import BranchStep
from gom.trajectory_dataset import listwise_strong_branch_loss
from gom.state import SearchState


def test_branch_step_candidate_scores_are_backward_compatible():
    old = {
        "state": SearchState(branch_candidates={"x0": True}).to_dict(),
        "chosen_variable": "x0",
        "chosen_value": 0.5,
        "score": 1.2,
    }
    restored = BranchStep.from_dict(old)
    assert restored.candidate_scores == {}

    new = BranchStep(
        SearchState(branch_candidates={"x0": True, "x1": True}),
        "x1",
        0.6,
        3.0,
        {"x0": 1.0, "x1": 3.0},
    )
    roundtrip = BranchStep.from_dict(new.to_dict())
    assert roundtrip.candidate_scores == {"x0": 1.0, "x1": 3.0}


def test_listwise_loss_prefers_logits_with_the_same_ranking():
    scores = torch.tensor([[0.0, 1.0, 3.0, 2.0]], dtype=torch.float32)
    mask = torch.tensor([[False, True, True, True]])
    aligned = torch.tensor([[-10.0, 0.0, 3.0, 2.0]], dtype=torch.float32)
    reversed_logits = torch.tensor([[-10.0, 3.0, 0.0, 1.0]], dtype=torch.float32)

    aligned_loss, valid = listwise_strong_branch_loss(aligned, scores, mask)
    reversed_loss, _ = listwise_strong_branch_loss(reversed_logits, scores, mask)

    assert bool(valid[0])
    assert aligned_loss < reversed_loss


def test_listwise_loss_ignores_rows_without_two_scores():
    logits = torch.tensor([[0.0, 1.0, 2.0]], requires_grad=True)
    scores = torch.tensor([[0.0, 0.0, 2.0]])
    mask = torch.tensor([[False, False, True]])
    loss, valid = listwise_strong_branch_loss(logits, scores, mask)
    assert not bool(valid[0])
    assert loss.item() == 0.0
    loss.backward()
    assert logits.grad is not None
