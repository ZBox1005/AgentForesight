"""Shared incremental audit loop for local and API model backends."""

from __future__ import annotations

import time
from collections.abc import Callable

from inference.data import TrajectoryRecord
from inference.prompts import AuditVerdict, build_messages, parse_response


def audit_incremental(
    rec: TrajectoryRecord,
    query: Callable[[list[dict]], str],
) -> tuple[AuditVerdict, float, int, int | None]:
    total_time = 0.0
    n_calls = 0
    last: AuditVerdict | None = None
    for step in range(rec.num_turns):
        messages = build_messages(
            rec.turns, tools=rec.tools, current_step=step,
        )
        start = time.time()
        last = parse_response(query(messages))
        total_time += time.time() - start
        n_calls += 1
        if last.valid and last.pred_step >= 0:
            return last, total_time, n_calls, step
    raw = last.raw_response if last else ""
    return AuditVerdict(-1, "", "", bool(last and last.valid), raw), total_time, n_calls, None
