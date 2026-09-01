from __future__ import annotations

import argparse

import torch

from gom.lp_graph_dataset import load_lp_branch_samples, make_lp_branch_batch
from gom.model import GOMConfig, GOMModel
from gom.trajectory_dataset import branch_policy_loss, branch_top1_accuracy


def evaluate(model, batch, targets, ranking_weight: float):
    model.eval()
    with torch.inference_mode():
        loss, components, _, logits = branch_policy_loss(
            model,
            batch,
            targets,
            ranking_weight=ranking_weight,
        )
        acc = branch_top1_accuracy(logits, targets)
    return float(loss.item()), float(components["ranking"].item()), float(acc.item())


def main():
    p = argparse.ArgumentParser(description="Memorization sanity check using SCIP's native LP graph")
    p.add_argument("trajectories", nargs="+")
    p.add_argument("--samples", type=int, default=24)
    p.add_argument("--steps", type=int, default=300)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--ranking-weight", type=float, default=0.0)
    p.add_argument("--threads", type=int, default=1)
    p.add_argument("--require-top1", type=float, default=0.95)
    args = p.parse_args()

    torch.manual_seed(123)
    if args.threads > 0:
        torch.set_num_threads(args.threads)

    samples = load_lp_branch_samples(args.trajectories, max_samples=args.samples)
    if len(samples) < args.samples:
        raise SystemExit(f"requested {args.samples} LP samples but found {len(samples)}")

    batch, targets = make_lp_branch_batch(samples)
    cfg = GOMConfig(d_model=128, n_heads=4, n_layers=2, d_ff=384)
    model = GOMModel(cfg)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.0)

    initial_loss, initial_rank, initial_acc = evaluate(model, batch, targets, args.ranking_weight)
    print(
        f"samples={len(samples)} parameters={model.parameter_count():,} "
        f"nodes={batch.x.shape[1]} initial_loss={initial_loss:.4f} "
        f"initial_rank={initial_rank:.4f} initial_top1={initial_acc:.4f}"
    )

    checkpoints = {1, 10, 25, 50, 100, args.steps}
    model.train()
    for step in range(1, args.steps + 1):
        loss, _, _, _ = branch_policy_loss(
            model,
            batch,
            targets,
            ranking_weight=args.ranking_weight,
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        if step in checkpoints:
            current_loss, current_rank, current_acc = evaluate(model, batch, targets, args.ranking_weight)
            print(
                f"step={step:04d} loss={current_loss:.4f} rank={current_rank:.4f} "
                f"top1={current_acc:.4f}"
            )
            model.train()

    final_loss, final_rank, final_acc = evaluate(model, batch, targets, args.ranking_weight)
    print(f"final_loss={final_loss:.4f} final_rank={final_rank:.4f} final_top1={final_acc:.4f}")
    if final_acc < args.require_top1:
        raise SystemExit(
            f"native LP graph memorization failed: top1={final_acc:.4f} < {args.require_top1:.4f}"
        )


if __name__ == "__main__":
    main()
