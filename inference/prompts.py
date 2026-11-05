"""Paper V16 prompt and parser for online auditing.

The prompt is imported from the Stage 2 training package so inference uses the
same text and trajectory formatting. The answer parser mirrors the paper's
final internal evaluator: a parseable SAFE verdict continues the prefix walk;
the first parseable nonnegative step is an alarm.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

from training.prompts import (
    SYSTEM_PROMPT_V16 as SYSTEM_PROMPT,
    build_messages as _build_full_messages,
    build_messages_incremental as _build_incremental_messages,
)
from training.prompts_v15 import (
    build_messages as _build_full_messages_v15,
    build_messages_incremental as _build_incremental_messages_v15,
)


def build_messages(turns: list[dict], tools: list[dict] | None = None,
                   current_step: int | None = None,
                   prompt_version: str = "v16") -> list[dict]:
    if prompt_version == "v15":
        full, incremental = _build_full_messages_v15, _build_incremental_messages_v15
    elif prompt_version == "v16":
        full, incremental = _build_full_messages, _build_incremental_messages
    else:
        raise ValueError(f"unknown prompt version: {prompt_version}")
    if current_step is None:
        return full(turns, tools=tools, task_description=None)
    return incremental(
        turns, current_step=current_step, tools=tools, task_description=None,
    )


@dataclass
class AuditVerdict:
    pred_step: int
    pred_agent: str
    pred_reason: str
    valid: bool
    raw_response: str


_ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL | re.IGNORECASE)
_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


def parse_response(text: str) -> AuditVerdict:
    if not text:
        return AuditVerdict(-1, "", "", False, text or "")
    match = _ANSWER_RE.search(text)
    if not match:
        return AuditVerdict(-1, "", "", False, text)
    payload = _JSON_RE.search(match.group(1))
    if not payload:
        return AuditVerdict(-1, "", "", False, text)
    try:
        obj: dict[str, Any] = json.loads(payload.group(0))
    except (json.JSONDecodeError, ValueError):
        return AuditVerdict(-1, "", "", False, text)
    if "answer" not in obj:
        return AuditVerdict(-1, "", "", False, text)
    answer = obj["answer"]
    agent = str(obj.get("agent") or "")
    reason = str(obj.get("reason") or "")
    if isinstance(answer, str):
        stripped = answer.strip()
        if stripped.upper() == "SAFE":
            return AuditVerdict(-1, agent, reason, True, text)
        if stripped.lstrip("-").isdigit():
            return AuditVerdict(int(stripped), agent, reason, True, text)
    if isinstance(answer, int) and not isinstance(answer, bool):
        return AuditVerdict(answer, agent, reason, True, text)
    return AuditVerdict(-1, agent, reason, False, text)
