from __future__ import annotations

from collections import defaultdict
import json
import math
from pathlib import Path
from statistics import mean, median
from typing import Iterable, Sequence


def load_benchmark_rows(paths: str | Path | Sequence[str | Path]) -> list[dict]:
    if isinstance(paths, (str, Path)):
        paths = [paths]
    rows: list[dict] = []
    for raw_path in paths:
        path = Path(raw_path)
        with path.open("r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"{path}:{line_no}: invalid JSON") from exc
                if "problem_id" not in row or "policy" not in row:
                    raise ValueError(f"{path}:{line_no}: missing problem_id/policy")
                rows.append(row)
    return rows


def _finite(value) -> float | None:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def _metric(values: Iterable[float], fn) -> float | None:
    items = list(values)
    return None if not items else float(fn(items))


def compare_runs(candidate: dict, baseline: dict, tol: float = 1e-9) -> int:
    """Compare two time-budgeted solver runs.

    Returns +1 when candidate is better, -1 when baseline is better, and 0 for a tie.
    Primary metric is final optimality gap. If gaps tie, lower wall time wins; if wall
    time also ties, lower node count wins. Missing/non-finite gaps are treated as +inf.
    """
    cg = _finite(candidate.get("gap"))
    bg = _finite(baseline.get("gap"))
    cg = math.inf if cg is None else cg
    bg = math.inf if bg is None else bg
    if abs(cg - bg) > tol:
        return 1 if cg < bg else -1

    ct = _finite(candidate.get("wall_time_s"))
    bt = _finite(baseline.get("wall_time_s"))
    ct = math.inf if ct is None else ct
    bt = math.inf if bt is None else bt
    if abs(ct - bt) > tol:
        return 1 if ct < bt else -1

    cn = int(candidate.get("nodes", 0))
    bn = int(baseline.get("nodes", 0))
    if cn != bn:
        return 1 if cn < bn else -1
    return 0


def summarize_benchmark(rows: Sequence[dict], baseline_policy: str = "default") -> dict:
    grouped: dict[str, list[dict]] = defaultdict(list)
    by_problem: dict[str, dict[str, dict]] = defaultdict(dict)
    for row in rows:
        policy = str(row["policy"])
        grouped[policy].append(row)
        by_problem[str(row["problem_id"])][policy] = row

    policies: dict[str, dict] = {}
    for policy, selected in sorted(grouped.items()):
        gaps = [v for row in selected if (v := _finite(row.get("gap"))) is not None]
        times = [v for row in selected if (v := _finite(row.get("wall_time_s"))) is not None]
        nodes = [float(row.get("nodes", 0)) for row in selected]
        decisions = [float(row.get("gom_decisions", 0) or 0) for row in selected]
        fallbacks = [float(row.get("gom_fallbacks", 0) or 0) for row in selected]
        inference = [float(row.get("gom_inference_ms", 0.0) or 0.0) for row in selected]
        extract = [float(row.get("gom_extract_ms", 0.0) or 0.0) for row in selected]
        tensor = [float(row.get("gom_tensor_ms", 0.0) or 0.0) for row in selected]
        model = [float(row.get("gom_model_ms", 0.0) or 0.0) for row in selected]
        total_decisions = sum(decisions)
        policies[policy] = {
            "runs": len(selected),
            "optimal": sum(str(row.get("status", "")).lower() == "optimal" for row in selected),
            "mean_gap": _metric(gaps, mean),
            "median_gap": _metric(gaps, median),
            "mean_time_s": _metric(times, mean),
            "median_time_s": _metric(times, median),
            "mean_nodes": _metric(nodes, mean),
            "median_nodes": _metric(nodes, median),
            "mean_gom_decisions": _metric(decisions, mean),
            "mean_gom_fallbacks": _metric(fallbacks, mean),
            "mean_gom_inference_ms": _metric(inference, mean),
            "gom_ms_per_decision": None if total_decisions <= 0 else sum(inference) / total_decisions,
            "gom_extract_ms_per_decision": None if total_decisions <= 0 else sum(extract) / total_decisions,
            "gom_tensor_ms_per_decision": None if total_decisions <= 0 else sum(tensor) / total_decisions,
            "gom_model_ms_per_decision": None if total_decisions <= 0 else sum(model) / total_decisions,
        }

    paired: dict[str, dict] = {}
    for policy in sorted(grouped):
        if policy == baseline_policy:
            continue
        wins = ties = losses = 0
        for policies_for_problem in by_problem.values():
            if baseline_policy not in policies_for_problem or policy not in policies_for_problem:
                continue
            cmp = compare_runs(policies_for_problem[policy], policies_for_problem[baseline_policy])
            if cmp > 0:
                wins += 1
            elif cmp < 0:
                losses += 1
            else:
                ties += 1
        total = wins + ties + losses
        paired[policy] = {
            "baseline": baseline_policy,
            "pairs": total,
            "wins": wins,
            "ties": ties,
            "losses": losses,
            "win_rate": None if total == 0 else wins / total,
        }

    return {"baseline": baseline_policy, "policies": policies, "paired": paired}


def _fmt(value, digits: int = 4) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.{digits}f}"


def render_markdown(summary: dict) -> str:
    lines = [
        "# SCIP branching benchmark",
        "",
        "Lower gap/time/nodes is better. Pairwise wins use gap first, then wall time, then nodes.",
        "",
        "| policy | runs | optimal | mean gap | median gap | mean time (s) | median nodes | GOM decisions | fallbacks | inference ms |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for policy, stats in summary["policies"].items():
        lines.append(
            f"| {policy} | {stats['runs']} | {stats['optimal']} | {_fmt(stats['mean_gap'], 6)} | "
            f"{_fmt(stats['median_gap'], 6)} | {_fmt(stats['mean_time_s'], 3)} | "
            f"{_fmt(stats['median_nodes'], 1)} | {_fmt(stats['mean_gom_decisions'], 1)} | "
            f"{_fmt(stats['mean_gom_fallbacks'], 1)} | {_fmt(stats['mean_gom_inference_ms'], 2)} |"
        )

    latency_rows = [
        (policy, stats)
        for policy, stats in summary["policies"].items()
        if stats.get("gom_ms_per_decision") is not None
    ]
    if latency_rows:
        lines += [
            "",
            "## GOM branch latency",
            "",
            "| policy | total ms/decision | extract ms | tensor ms | model ms |",
            "|---|---:|---:|---:|---:|",
        ]
        for policy, stats in latency_rows:
            lines.append(
                f"| {policy} | {_fmt(stats['gom_ms_per_decision'], 3)} | "
                f"{_fmt(stats['gom_extract_ms_per_decision'], 3)} | "
                f"{_fmt(stats['gom_tensor_ms_per_decision'], 3)} | "
                f"{_fmt(stats['gom_model_ms_per_decision'], 3)} |"
            )

    lines += ["", f"## Paired vs `{summary['baseline']}`", "", "| policy | pairs | wins | ties | losses | win rate |", "|---|---:|---:|---:|---:|---:|"]
    for policy, stats in summary["paired"].items():
        rate = "n/a" if stats["win_rate"] is None else f"{stats['win_rate'] * 100:.1f}%"
        lines.append(f"| {policy} | {stats['pairs']} | {stats['wins']} | {stats['ties']} | {stats['losses']} | {rate} |")
    return "\n".join(lines) + "\n"
