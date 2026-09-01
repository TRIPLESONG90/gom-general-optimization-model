from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import subprocess
import sys

import torch
import yaml


ROOT = Path(__file__).resolve().parents[1]


def _run(cmd: list[str], *, dry_run: bool = False) -> None:
    print("+", " ".join(cmd), flush=True)
    if not dry_run:
        subprocess.run(cmd, cwd=ROOT, check=True)


def _run_parallel(commands: list[list[str]], *, dry_run: bool = False) -> None:
    for cmd in commands:
        print("+", " ".join(cmd), flush=True)
    if dry_run:
        return
    processes = [subprocess.Popen(cmd, cwd=ROOT) for cmd in commands]
    failures = []
    for index, process in enumerate(processes):
        code = process.wait()
        if code != 0:
            failures.append((index, code))
    if failures:
        raise SystemExit(f"collection shard failures: {failures}")


def _device(value: str) -> str:
    if value != "auto":
        return value
    return "cuda" if torch.cuda.is_available() else "cpu"


def _git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return None


def main():
    p = argparse.ArgumentParser(description="Run an end-to-end GOM SCIP experiment")
    p.add_argument("--config", default="experiments/scip_v03.yaml")
    p.add_argument("--profile", default="smoke")
    p.add_argument("--run-dir", default=None)
    p.add_argument("--skip-collect", action="store_true")
    p.add_argument("--skip-train", action="store_true")
    p.add_argument("--skip-benchmark", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    config_path = (ROOT / args.config).resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    try:
        profile = config["profiles"][args.profile]
    except KeyError as exc:
        raise SystemExit(f"unknown profile {args.profile!r} in {config_path}") from exc

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = Path(args.run_dir) if args.run_dir else ROOT / "runs" / f"{config['name']}-{args.profile}-{stamp}"
    run_dir = run_dir.resolve()
    run_dir.mkdir(parents=True, exist_ok=True)

    python = sys.executable
    collect_cfg = profile["collect"]
    train_cfg = profile["train"]
    benchmark_cfg = profile["benchmark"]

    shards = int(collect_cfg.get("shards", 1))
    count_per_shard = int(collect_cfg["count_per_shard"])
    base_seed = int(collect_cfg["seed"])
    trajectory_paths = [run_dir / f"trajectories-{i:03d}.jsonl" for i in range(shards)]
    checkpoint = run_dir / "gom_scip_branch.pt"
    benchmark_path = run_dir / "benchmark.jsonl"
    report_md = run_dir / "REPORT.md"
    report_json = run_dir / "summary.json"

    manifest = {
        "experiment": config["name"],
        "profile": args.profile,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "config": profile,
        "paths": {
            "run_dir": str(run_dir),
            "trajectories": [str(p) for p in trajectory_paths],
            "checkpoint": str(checkpoint),
            "benchmark": str(benchmark_path),
            "report": str(report_md),
        },
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    if not args.skip_collect:
        commands = []
        for shard, path in enumerate(trajectory_paths):
            commands.append([
                python,
                "scripts/collect_scip.py",
                "--count", str(count_per_shard),
                "--seed", str(base_seed + shard),
                "--time-limit", str(collect_cfg["time_limit"]),
                "--max-steps", str(collect_cfg["max_steps"]),
                "--out", str(path),
            ])
        _run_parallel(commands, dry_run=args.dry_run)

    if not args.skip_train:
        cmd = [
            python,
            "scripts/train_scip_branch.py",
            *[str(p) for p in trajectory_paths],
            "--epochs", str(train_cfg["epochs"]),
            "--batch-size", str(train_cfg["batch_size"]),
            "--lr", str(train_cfg["lr"]),
            "--seed", str(train_cfg["seed"]),
            "--val-fraction", str(train_cfg["val_fraction"]),
            "--device", _device(str(train_cfg.get("device", "auto"))),
            "--out", str(checkpoint),
        ]
        if "ranking_weight" in train_cfg:
            cmd += ["--ranking-weight", str(train_cfg["ranking_weight"])]
        if "ranking_temperature" in train_cfg:
            cmd += ["--ranking-temperature", str(train_cfg["ranking_temperature"])]
        if bool(train_cfg.get("tiny", False)):
            cmd.append("--tiny")
        _run(cmd, dry_run=args.dry_run)

    if not args.skip_benchmark:
        _run([
            python,
            "scripts/benchmark_scip.py",
            "--checkpoint", str(checkpoint),
            "--count", str(benchmark_cfg["count"]),
            "--seed", str(benchmark_cfg["seed"]),
            "--time-limit", str(benchmark_cfg["time_limit"]),
            "--device", _device(str(benchmark_cfg.get("device", "cpu"))),
            "--out", str(benchmark_path),
        ], dry_run=args.dry_run)
        _run([
            python,
            "scripts/summarize_benchmark.py",
            str(benchmark_path),
            "--markdown", str(report_md),
            "--json", str(report_json),
        ], dry_run=args.dry_run)

    print(f"run directory: {run_dir}")


if __name__ == "__main__":
    main()
