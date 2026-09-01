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


def chunks(items, size):
    for start in range(0, len(items), size):
        yield items[start:start + size]


def main():
    p = argparse.ArgumentParser(description="Distill SCIP strong-branching decisions and rankings")
    p.add_argument("trajectories", nargs="+", help="JSONL files from collect_scip.py")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-samples", type=int, default=None)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--tiny", action="store_true")
    p.add_argument("--init", default=None, help="Optional GOM checkpoint to fine-tune")
    p.add_argument("--val-fraction", type=float, default=0.1)
    p.add_argument("--ranking-weight", type=float, default=0.0, help="0 keeps the v0.3 top-1 baseline; v0.4 uses >0")
    p.add_argument("--ranking-temperature", type=float, default=1.0)
    p.add_argument("--out", default="gom_scip_branch.pt")
    args = p.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    samples = load_branch_samples(args.trajectories, max_samples=args.max_samples)
    if not samples:
        raise SystemExit("No branch samples were found")
    ranked_samples = sum(len(s.candidate_scores) >= 2 for s in samples)
    print(f"branch samples: {len(samples):,} ranked={ranked_samples:,} ({ranked_samples / len(samples):.1%})")

    # Split by problem id, not by individual branch step. Random step-level
    # splitting leaks near-identical states from the same B&B trajectory.
    problem_ids = sorted({s.problem.id for s in samples})
    random.shuffle(problem_ids)
    n_val_problems = int(round(len(problem_ids) * args.val_fraction))
    if len(problem_ids) > 1 and args.val_fraction > 0:
        n_val_problems = max(1, min(len(problem_ids) - 1, n_val_problems))
    else:
        n_val_problems = 0
    val_ids = set(problem_ids[:n_val_problems])
    val_samples = [s for s in samples if s.problem.id in val_ids]
    train_samples = [s for s in samples if s.problem.id not in val_ids]
    print(
        f"train={len(train_samples):,} ({len(set(s.problem.id for s in train_samples))} problems) "
        f"val={len(val_samples):,} ({len(val_ids)} problems)"
    )

    if args.init:
        payload = torch.load(args.init, map_location=args.device)
        cfg = GOMConfig(**payload["config"])
        model = GOMModel(cfg).to(args.device)
        model.load_state_dict(payload["model"])
    else:
        cfg = GOMConfig(d_model=128, n_heads=4, n_layers=2, d_ff=384) if args.tiny else GOMConfig()
        model = GOMModel(cfg).to(args.device)

    print(f"parameters: {model.parameter_count():,}")
    print(
        f"objective: top1={1.0 - args.ranking_weight:.2f} "
        f"ranking={args.ranking_weight:.2f} temperature={args.ranking_temperature:.2f}"
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)

    for epoch in range(1, args.epochs + 1):
        random.shuffle(train_samples)
        total_loss = 0.0
        total_top1_loss = 0.0
        total_rank_loss = 0.0
        total_rank_rows = 0
        total_correct = 0.0
        total_count = 0
        model.train()
        for batch_samples in chunks(train_samples, args.batch_size):
            batch, targets = make_branch_batch(batch_samples, args.device)
            loss, components, _, logits = branch_policy_loss(
                model,
                batch,
                targets,
                ranking_weight=args.ranking_weight,
                ranking_temperature=args.ranking_temperature,
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            n = len(batch_samples)
            rank_rows = int(components["ranking_rows"].item())
            total_loss += float(loss.item()) * n
            total_top1_loss += float(components["top1"].item()) * n
            total_rank_loss += float(components["ranking"].item()) * rank_rows
            total_rank_rows += rank_rows
            total_correct += float(branch_top1_accuracy(logits, targets).item()) * n
            total_count += n

        train_loss = total_loss / total_count
        train_acc = total_correct / total_count
        train_top1_loss = total_top1_loss / total_count
        train_rank_loss = total_rank_loss / max(1, total_rank_rows)
        val_loss = None
        val_acc = None
        val_rank_loss = None
        if val_samples:
            model.eval()
            vloss = 0.0
            vrank_loss = 0.0
            vrank_rows = 0
            vcorrect = 0.0
            vcount = 0
            with torch.inference_mode():
                for batch_samples in chunks(val_samples, args.batch_size):
                    batch, targets = make_branch_batch(batch_samples, args.device)
                    loss, components, _, logits = branch_policy_loss(
                        model,
                        batch,
                        targets,
                        ranking_weight=args.ranking_weight,
                        ranking_temperature=args.ranking_temperature,
                    )
                    n = len(batch_samples)
                    rank_rows = int(components["ranking_rows"].item())
                    vloss += float(loss.item()) * n
                    vrank_loss += float(components["ranking"].item()) * rank_rows
                    vrank_rows += rank_rows
                    vcorrect += float(branch_top1_accuracy(logits, targets).item()) * n
                    vcount += n
            val_loss = vloss / vcount
            val_acc = vcorrect / vcount
            val_rank_loss = vrank_loss / max(1, vrank_rows)

        suffix = ""
        if val_loss is not None:
            suffix = (
                f" val_loss={val_loss:.4f} val_rank={val_rank_loss:.4f} "
                f"val_top1={val_acc:.4f}"
            )
        print(
            f"epoch={epoch:03d} loss={train_loss:.4f} ce={train_top1_loss:.4f} "
            f"rank={train_rank_loss:.4f} top1={train_acc:.4f}{suffix}"
        )

    torch.save(
        {
            "config": cfg.__dict__,
            "model": model.state_dict(),
            "training": {
                "ranking_weight": args.ranking_weight,
                "ranking_temperature": args.ranking_temperature,
                "seed": args.seed,
            },
        },
        args.out,
    )
    print(f"saved: {args.out}")


if __name__ == "__main__":
    main()
