from __future__ import annotations

import argparse
import random

import torch

from gom.model import GOMConfig, GOMModel
from gom.trajectory_dataset import (
    branch_policy_loss,
    branch_top1_accuracy,
    load_branch_samples,
    make_branch_batch,
)


def main():
    p = argparse.ArgumentParser(
        description="Check whether a tiny GOM can memorize a small fixed set of SCIP branch states"
    )
    p.add_argument("trajectories", nargs="+")
    p.add_argument("--samples", type=int, default=24)
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--ranking-weight", type=float, default=0.75)
    p.add_argument("--ranking-temperature", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--device", default="cpu")
    p.add_argument(
        "--require-top1",
        type=float,
        default=None,
        help="optional minimum final memorization accuracy; exits non-zero if missed",
    )
    args = p.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    samples = load_branch_samples(args.trajectories, max_samples=args.samples)
    if not samples:
        raise SystemExit("No branch samples found")

    batch, targets = make_branch_batch(samples, args.device)
    cfg = GOMConfig(d_model=128, n_heads=4, n_layers=2, d_ff=384)
    model = GOMModel(cfg).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.0)

    def evaluate():
        model.eval()
        with torch.inference_mode():
            loss, components, _, logits = branch_policy_loss(
                model,
                batch,
                targets,
                ranking_weight=args.ranking_weight,
                ranking_temperature=args.ranking_temperature,
            )
            accuracy = float(branch_top1_accuracy(logits, targets).item())
        return float(loss.item()), float(components["ranking"].item()), accuracy

    initial_loss, initial_rank, initial_acc = evaluate()
    print(
        f"samples={len(samples)} parameters={model.parameter_count():,} "
        f"initial_loss={initial_loss:.4f} initial_rank={initial_rank:.4f} "
        f"initial_top1={initial_acc:.4f}"
    )

    model.train()
    for step in range(1, args.steps + 1):
        loss, _, _, _ = branch_policy_loss(
            model,
            batch,
            targets,
            ranking_weight=args.ranking_weight,
            ranking_temperature=args.ranking_temperature,
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if step in {1, 10, 25, 50, 100, args.steps}:
            current_loss, current_rank, current_acc = evaluate()
            print(
                f"step={step:04d} loss={current_loss:.4f} "
                f"rank={current_rank:.4f} top1={current_acc:.4f}"
            )
            model.train()

    final_loss, final_rank, final_acc = evaluate()
    print(
        f"final_loss={final_loss:.4f} final_rank={final_rank:.4f} "
        f"final_top1={final_acc:.4f}"
    )
    if args.require_top1 is not None and final_acc + 1e-12 < args.require_top1:
        raise SystemExit(
            f"memorization diagnostic failed: top1={final_acc:.4f} < {args.require_top1:.4f}"
        )


if __name__ == "__main__":
    main()
