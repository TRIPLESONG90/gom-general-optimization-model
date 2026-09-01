from __future__ import annotations

import argparse
import json
from pathlib import Path

from gom.benchmark import load_benchmark_rows, render_markdown, summarize_benchmark


def main():
    p = argparse.ArgumentParser(description="Summarize SCIP branching benchmark JSONL")
    p.add_argument("inputs", nargs="+", help="benchmark JSONL files")
    p.add_argument("--baseline", default="default")
    p.add_argument("--markdown", default=None, help="optional Markdown output path")
    p.add_argument("--json", dest="json_out", default=None, help="optional JSON summary path")
    args = p.parse_args()

    rows = load_benchmark_rows(args.inputs)
    summary = summarize_benchmark(rows, baseline_policy=args.baseline)
    report = render_markdown(summary)
    print(report, end="")

    if args.markdown:
        path = Path(args.markdown)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(report, encoding="utf-8")
    if args.json_out:
        path = Path(args.json_out)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
