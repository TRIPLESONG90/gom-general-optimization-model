from __future__ import annotations

import argparse
import itertools
import torch

from gom.generators import synthetic_stream
from gom.model import GOMConfig, GOMModel
from gom.training import make_batch, compute_loss


def chunks(it, size):
    it = iter(it)
    while True:
        batch = list(itertools.islice(it, size))
        if not batch:
            break
        yield batch


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--tiny", action="store_true", help="Use a tiny model for smoke tests")
    p.add_argument("--out", default="gom_checkpoint.pt")
    args = p.parse_args()
    torch.manual_seed(args.seed)
    cfg = GOMConfig(d_model=128, n_heads=4, n_layers=2, d_ff=384) if args.tiny else GOMConfig()
    model = GOMModel(cfg).to(args.device)
    print(f"parameters: {model.parameter_count():,}")
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    stream = synthetic_stream(args.seed, count=args.steps * args.batch_size)
    for step, samples in enumerate(chunks(stream, args.batch_size), start=1):
        batch, targets = make_batch(samples, args.device)
        total, losses, _ = compute_loss(model, batch, targets)
        opt.zero_grad(set_to_none=True)
        total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step == 1 or step % 10 == 0:
            detail = " ".join(f"{k}={v.item():.4f}" for k, v in losses.items())
            print(f"step={step:04d} total={total.item():.4f} {detail}")
        if step >= args.steps:
            break
    torch.save({"config": cfg.__dict__, "model": model.state_dict()}, args.out)
    print(f"saved: {args.out}")


if __name__ == "__main__":
    main()
