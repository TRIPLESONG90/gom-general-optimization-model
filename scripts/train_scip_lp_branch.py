from __future__ import annotations

import argparse
import random

import torch

from gom.lp_graph_dataset import load_lp_branch_samples, make_lp_branch_batch
from gom.model import GOMConfig, GOMModel
from gom.trajectory_dataset import branch_policy_loss, branch_top1_accuracy


def chunks(items, size):
    for start in range(0, len(items), size):
        yield items[start:start + size]


def main():
    p = argparse.ArgumentParser(description="Train GOM branching policy from SCIP native LP graph snapshots")
    p.add_argument("trajectories", nargs="+", help="JSONL files from collect_scip.py containing lp_graph snapshots")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-samples", type=int, default=None)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--tiny", action="store_true")
    p.add_argument("--val-fraction", type=float, default=0.1)
    p.add_argument("--ranking-weight", type=float, default=0.0)
    p.add_argument("--ranking-temperature", type=float, default=1.0)
    p.add_argument("--threads", type=int, default=1, help="CPU intra-op threads; 1 is usually faster for tiny branch inference")
    p.add_argument("--out", default="gom_scip_lp_branch.pt")
    args = p.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.device == "cpu" and args.threads > 0:
        torch.set_num_threads(args.threads)

    samples = load_lp_branch_samples(args.trajectories, max_samples=args.max_samples)
    if not samples:
        raise SystemExit("No native LP branch samples were found")
    ranked = sum(len(s.candidate_scores) >= 2 for s in samples)
    print(f"LP branch samples: {len(samples):,} ranked={ranked:,} ({ranked / len(samples):.1%})")

    # Split by problem id to prevent neighboring B&B states from leaking into validation.
    problem_ids = sorted({s.problem_id for s in samples})
    random.shuffle(problem_ids)
    n_val = int(round(len(problem_ids) * args.val_fraction))
    if len(problem_ids) > 1 and args.val_fraction > 0:
        n_val = max(1, min(len(problem_ids) - 1, n_val))
    else:
        n_val = 0
    val_ids = set(problem_ids[:n_val])
    train_samples = [s for s in samples if s.problem_id not in val_ids]
    val_samples = [s for s in samples if s.problem_id in val_ids]
    print(
        f"train={len(train_samples):,} ({len(set(s.problem_id for s in train_samples))} problems) "
        f"val={len(val_samples):,} ({len(val_ids)} problems)"
    )

    cfg = GOMConfig(d_model=128, n_heads=4, n_layers=2, d_ff=384) if args.tiny else GOMConfig()
    model = GOMModel(cfg).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    print(f"parameters: {model.parameter_count():,}")

    def evaluate(data):
        model.eval()
        total_loss = 0.0
        total_correct = 0.0
        total_count = 0
        with torch.inference_mode():
            for sample_batch in chunks(data, args.batch_size):
                batch, targets = make_lp_branch_batch(sample_batch, args.device)
                loss, _, _, logits = branch_policy_loss(
                    model,
                    batch,
                    targets,
                    ranking_weight=args.ranking_weight,
                    ranking_temperature=args.ranking_temperature,
                )
                n = len(sample_batch)
                total_loss += float(loss.item()) * n
                total_correct += float(branch_top1_accuracy(logits, targets).item()) * n
                total_count += n
        if not total_count:
            return None, None
        return total_loss / total_count, total_correct / total_count

    for epoch in range(1, args.epochs + 1):
        random.shuffle(train_samples)
        model.train()
        total_loss = 0.0
        total_correct = 0.0
        total_count = 0
        for sample_batch in chunks(train_samples, args.batch_size):
            batch, targets = make_lp_branch_batch(sample_batch, args.device)
            loss, _, _, logits = branch_policy_loss(
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
            n = len(sample_batch)
            total_loss += float(loss.item()) * n
            total_correct += float(branch_top1_accuracy(logits, targets).item()) * n
            total_count += n

        train_loss = total_loss / total_count
        train_acc = total_correct / total_count
        val_loss, val_acc = evaluate(val_samples)
        suffix = "" if val_loss is None else f" val_loss={val_loss:.4f} val_top1={val_acc:.4f}"
        print(f"epoch={epoch:03d} loss={train_loss:.4f} top1={train_acc:.4f}{suffix}")

    torch.save(
        {
            "config": cfg.__dict__,
            "model": model.state_dict(),
            "input_representation": "scip_native_lp",
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
