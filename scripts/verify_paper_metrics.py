"""Recompute the reported test metrics from the released decision ledger.

Usage: python -m scripts.verify_paper_metrics
"""
from __future__ import annotations

import json
from pathlib import Path

from inference.metrics import aggregate_by_domain, format_report


path = Path(__file__).resolve().parents[1] / "results/paper_test_predictions.jsonl"
rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
assert len(rows) == 332, f"expected 332 test trajectories, got {len(rows)}"
summary = aggregate_by_domain(rows, macro=True)
assert set(summary) == {"Math", "Coding", "Agentic", "overall"}, set(summary)
overall = summary["overall"]
assert (overall["n_safe"], overall["n_unsafe"]) == (169, 163)
assert round(overall["exact_f1"], 2) == 66.44
assert round(overall["far"], 2) == 2.37
assert round(overall["ass_mean"], 3) == 0.592
print(format_report(summary))
