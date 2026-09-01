from gom.benchmark import compare_runs, render_markdown, summarize_benchmark


def _row(problem, policy, gap, time_s, nodes, status="timelimit", **extra):
    return {
        "problem_id": problem,
        "problem_type": "multidimensional_knapsack",
        "policy": policy,
        "gap": gap,
        "wall_time_s": time_s,
        "nodes": nodes,
        "status": status,
        **extra,
    }


def test_compare_runs_prefers_gap_then_time_then_nodes():
    baseline = _row("p", "default", 0.1, 10.0, 100)
    assert compare_runs(_row("p", "gom", 0.05, 20.0, 1000), baseline) == 1
    assert compare_runs(_row("p", "gom", 0.1, 9.0, 1000), baseline) == 1
    assert compare_runs(_row("p", "gom", 0.1, 10.0, 90), baseline) == 1
    assert compare_runs(_row("p", "gom", 0.2, 1.0, 1), baseline) == -1


def test_summary_uses_problem_paired_win_rate():
    rows = [
        _row("p1", "default", 0.10, 10.0, 100),
        _row("p1", "gom", 0.05, 10.0, 80, gom_decisions=4),
        _row("p2", "default", 0.00, 8.0, 50, status="optimal"),
        _row("p2", "gom", 0.00, 7.0, 60, status="optimal", gom_decisions=3),
        _row("p3", "default", 0.02, 10.0, 70),
        _row("p3", "gom", 0.03, 10.0, 60, gom_decisions=2),
    ]
    summary = summarize_benchmark(rows)
    paired = summary["paired"]["gom"]
    assert paired["pairs"] == 3
    assert paired["wins"] == 2
    assert paired["losses"] == 1
    assert paired["ties"] == 0
    assert summary["policies"]["gom"]["mean_gom_decisions"] == 3.0

    report = render_markdown(summary)
    assert "SCIP branching benchmark" in report
    assert "66.7%" in report
