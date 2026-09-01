from __future__ import annotations

import argparse
import random

import torch

from gom.model import GOMConfig, GOMModel
from gom.trajectory_dataset import (
    branch_imitation_loss,
    branch_top1_accuracy,
    load_branch_samples,
    make_branch_batch,
)


def chunks(items, size):
    for start in range(0, len(items), size):
        yield items[start:start + size]


def main():
    p = argparse.ArgumentParser(description="Imitate SCIP strong-branching decisions")
    p.add_argument("trajectories", nargs="+", help="JSONL files from collect_scip.py")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-samples", type=int, default=None)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--tiny", action="store_true")
    p.add_argument("--init", default=None, help="Optional GOM checkpoint to fine-tune")
    p.add_argument("--out", default="gom_scip_branch.pt")
    args = p.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    samples = load_branch_samples(args.trajectories, max_samples=args.max_samples)
    if not samples:
        raise SystemExit("No branch samples were found")
    print(f"branch samples: {len(samples):,}")

    if args.init:
        payload = torch.load(args.init, map_location=args.device)
        cfg = GOMConfig(**payload["config"])
        model = GOMModel(cfg).to(args.device)
        model.load_state_dict(payload["model"])
    else:
        cfg = GOMConfig(d_model=128, n_heads=4, n_layers=2, d_ff=384) if args.tiny else GOMConfig()
        model = GOMModel(cfg).to(args.device)

    print(f"parameters: {model.parameter_count():,}")
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)

    for epoch in range(1, args.epochs + 1):
        random.shuffle(samples)
        total_loss = 0.0
        total_correct = 0.0
        total_count = 0
        model.train()
        for batch_samples in chunks(samples, args.batch_size):
            batch, targets = make_branch_batch(batch_samples, args.device)
            loss, _, logits = branch_imitation_loss(model, batch, targets)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            n = len(batch_samples)
            total_loss += float(loss.item()) * n
            total_correct += float(branch_top1_accuracy(logits, targets).item()) * n
            total_count += n
        print(
            f"epoch={epoch:03d} loss={total_loss / total_count:.4f} "
            f"top1={total_correct / total_count:.4f}"
        )

    torch.save({"config": cfg.__dict__, "model": model.state_dict()}, args.out)
    print(f"saved: {args.out}")


if __name__ == "__main__":
    main()
