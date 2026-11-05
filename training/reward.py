"""V16 Reward — V9 task semantics + simplified V16 schema validation.

Diff from V9 (reward_audit_v9.py):
  - REQUIRED_JSON_FIELDS: drops `evidence_steps` and `suggested_fix`,
                          adds `reason`.
  - Phase 2 SAFE: requires reason=null instead of evidence_steps=[] + suggested_fix=null.
  - Phase 2 UNSAFE: reason as non-empty string of length 15..REASON_MAX_CHARS,
                    AND must mention a step number or agent role (V9-style grounding,
                    transplanted from suggested_fix to reason).
  - Phase 3 vapid <think> soft penalty (-0.3) — UNCHANGED from V9.
  - Phase 4 task reward — V9 SIGMA=1.0, FA=MISS=-1.0, multi-component (W_STEP=0.7, W_AGENT=0.3).
  - Length penalty: NOT in this file. Handled by verl's DAPO RewardManager
    (overlong_buffer_cfg) which uses real token counts from valid_response_length.
    Configured in training/run_grpo.sh via reward.reward_manager.name=dapo and
    +reward.reward_kwargs.overlong_buffer_cfg.{enable,len,penalty_factor,log}.
    Triggers when valid_response_length > (max_resp_len - overlong_buffer.len).
    This file's compute_score is called by DAPORewardManager; the overlong reward
    is added externally to whatever score we return.

Format expected:
  <think>
  ...reasoning text mentioning step numbers and agent names...
  </think>
  <answer>
  {"answer": <step_int> or "SAFE",
   "agent": "<role from trajectory>" or null,
   "reason": "<short grounded sentence mentioning step or agent>" or null}
  </answer>
"""
from __future__ import annotations
import json
import math
import re

# ============================================================
# Constants
# ============================================================

SIGMA = 1.0  # Gaussian width for step proximity (matches V9 / AgenTracer paper)

FORMAT_FAIL_REWARD = -0.5  # outer-format failure
VAPID_THINK_PENALTY = -0.3  # think block lacks step/agent mention

# V16 required JSON fields (was: answer/agent/evidence_steps/suggested_fix)
REQUIRED_JSON_FIELDS = {"answer", "agent", "reason"}

# Multi-component weights (sum to 1.0; both classes have max +1.0) — same as V9
W_STEP = 0.7
W_AGENT = 0.3
AGENT_PARTIAL_CREDIT = 0.5

# Reason field constraints (UNSAFE predictions only)
REASON_MIN_CHARS = 15
REASON_MAX_CHARS = 200  # cap to prevent reason itself from inflating

# (Length penalty handled externally by DAPORewardManager — see module docstring.)


# ============================================================
# Format parsing
# ============================================================

_THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL | re.IGNORECASE)
_ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL | re.IGNORECASE)
_JSON_OBJ_RE = re.compile(r"\{.*\}", re.DOTALL)
_STEP_MENTION_RE = re.compile(r"\bsteps?\s+(\d+)\b", re.IGNORECASE)
_AGENT_MENTION_RE = re.compile(
    r"\b(MathSolver|Verifier|CodeWriter|CodeTester|Manager|search_agent|"
    r"TaskSolver|Validator|Solver|Computer_terminal|"
    r"Verification_Expert|Research_Expert|Geography_Expert|Movie_Expert|"
    r"Statistics_Expert|Sports_Expert|"
    r"[A-Z][a-zA-Z]*_Expert)\b"
)


class ParsedV16:
    __slots__ = ("has_think", "has_answer", "think_text", "answer_text",
                 "json_obj", "json_valid")

    def __init__(self):
        self.has_think = False
        self.has_answer = False
        self.think_text = ""
        self.answer_text = ""
        self.json_obj = None
        self.json_valid = False


def parse_v16(s: str) -> ParsedV16:
    """Parse V16 format string. Lenient — partial parses still populate fields."""
    p = ParsedV16()
    if not s:
        return p
    m_think = _THINK_RE.search(s)
    if m_think:
        p.has_think = True
        p.think_text = m_think.group(1).strip()
    m_ans = _ANSWER_RE.search(s)
    if m_ans:
        p.has_answer = True
        p.answer_text = m_ans.group(1).strip()
        m_json = _JSON_OBJ_RE.search(p.answer_text)
        if m_json:
            try:
                p.json_obj = json.loads(m_json.group(0))
                p.json_valid = True
            except (json.JSONDecodeError, ValueError):
                p.json_obj = None
                p.json_valid = False
    return p


# ============================================================
# Content checks
# ============================================================

def mentions_step_in_text(text: str) -> bool:
    return bool(_STEP_MENTION_RE.search(text or ""))


def mentions_agent_or_role(text: str, agent_name: str = "") -> bool:
    text = text or ""
    if agent_name and agent_name.strip().lower() in text.lower():
        return True
    return bool(_AGENT_MENTION_RE.search(text))


# Length penalty is now handled by verl's DAPORewardManager; see module docstring.


# ============================================================
# Reward computation
# ============================================================

def compute_score(data_source: str, solution_str: str, ground_truth, extra_info: dict = None):
    """V16 multi-component reward with simplified-schema format gate.

    Returns dict with: score, acc, format_valid, r_step, r_agent, r_proximity,
                       abs_step_shift, case, fail_reason.
    Length penalty is added externally by DAPORewardManager.overlong_buffer_cfg.
    """
    return _compute_score_core(data_source, solution_str, ground_truth, extra_info)


def _compute_score_core(data_source: str, solution_str: str, ground_truth, extra_info: dict = None):
    if isinstance(ground_truth, str):
        gt = json.loads(ground_truth)
    else:
        gt = ground_truth

    gt_step = gt.get("mistake_step", -1)
    gt_agent = (gt.get("mistake_agent", "") or "").strip()

    result = {
        "score": FORMAT_FAIL_REWARD,
        "acc": 0.0,
        "format_valid": 0.0,
        "r_step": 0.0,
        "r_agent": 0.0,
        "r_proximity": 0.0,
        "abs_step_shift": -1.0,
        "case": 0,  # 0=format fail, 1=safe correct, 2=FA, 3=miss, 4=detected
        "fail_reason": "",
    }

    # ── Phase 1: structural gate (V16 schema) ──
    parsed = parse_v16(solution_str)
    if not parsed.has_think:
        result["fail_reason"] = "missing_think_block"
        return result
    if not parsed.has_answer:
        result["fail_reason"] = "missing_answer_block"
        return result
    if not parsed.json_valid:
        result["fail_reason"] = "answer_json_invalid"
        return result
    if not REQUIRED_JSON_FIELDS.issubset(parsed.json_obj.keys()):
        missing = REQUIRED_JSON_FIELDS - set(parsed.json_obj.keys())
        result["fail_reason"] = f"missing_fields:{sorted(missing)}"
        return result

    obj = parsed.json_obj
    answer = obj.get("answer")
    is_safe_pred = (
        (isinstance(answer, str) and answer.upper() == "SAFE") or
        (isinstance(answer, int) and answer == -1)
    )

    # ── Phase 2: SAFE/UNSAFE structural consistency (V16 schema) ──
    if is_safe_pred:
        if obj.get("agent") not in (None, "", "null"):
            result["fail_reason"] = "safe_with_nonnull_agent"
            return result
        if obj.get("reason") not in (None, "", "null"):
            result["fail_reason"] = "safe_with_nonnull_reason"
            return result
        pred_step = -1
    else:
        # UNSAFE: must parse step int
        try:
            pred_step = int(answer) if not isinstance(answer, int) else answer
        except (ValueError, TypeError):
            result["fail_reason"] = "answer_not_safe_or_int"
            return result
        if pred_step < 0:
            result["fail_reason"] = "answer_negative_step"
            return result
        # agent must be non-empty string
        agent_pred = obj.get("agent")
        if not (isinstance(agent_pred, str) and agent_pred.strip()):
            result["fail_reason"] = "unsafe_with_null_agent"
            return result
        # reason must be non-empty string in [REASON_MIN_CHARS, REASON_MAX_CHARS] AND grounded
        reason = obj.get("reason")
        if not (isinstance(reason, str) and reason.strip()):
            result["fail_reason"] = "unsafe_with_null_reason"
            return result
        reason_text = reason.strip()
        if len(reason_text) < REASON_MIN_CHARS:
            result["fail_reason"] = f"reason_too_short:{len(reason_text)}"
            return result
        if len(reason_text) > REASON_MAX_CHARS:
            result["fail_reason"] = f"reason_too_long:{len(reason_text)}"
            return result
        # V9-style grounding (transplanted from suggested_fix to reason): mention step OR agent
        has_step = bool(_STEP_MENTION_RE.search(reason_text))
        has_agent = mentions_agent_or_role(reason_text, gt_agent)
        if not (has_step or has_agent):
            result["fail_reason"] = "reason_lacks_grounding"
            return result

    result["format_valid"] = 1.0

    # ── Phase 3: <think> content quality (soft penalty if vapid) — UNCHANGED ──
    think = parsed.think_text
    think_grounded = (mentions_step_in_text(think) or mentions_agent_or_role(think, gt_agent))
    if not think_grounded:
        result["score"] = VAPID_THINK_PENALTY
        result["format_valid"] = 0.0  # so length penalty doesn't double-stack
        result["fail_reason"] = "vapid_think"
        return result

    # ── Phase 4: TASK reward (V9 — symmetric, no domain shaping) ──
    if gt_step == -1:
        # Ground truth SAFE
        if is_safe_pred:
            result["score"] = 1.0
            result["acc"] = 1.0
            result["case"] = 1  # safe correct
        else:
            result["score"] = -1.0
            result["case"] = 2  # FA
        return result

    # Ground truth UNSAFE
    if is_safe_pred:
        result["score"] = -1.0  # missed (symmetric with FA)
        result["case"] = 3
        return result

    # Both detected → multi-component reward
    abs_shift = abs(pred_step - gt_step)
    r_step = math.exp(-(abs_shift ** 2) / (2 * SIGMA ** 2))

    if gt_agent:
        agent_pred = obj["agent"].strip().lower()
        gt_agent_l = gt_agent.lower()
        agent_match = (agent_pred == gt_agent_l) or (gt_agent_l in agent_pred) or (agent_pred in gt_agent_l)
        r_agent = 1.0 if agent_match else AGENT_PARTIAL_CREDIT
    else:
        r_agent = 1.0

    score = W_STEP * r_step + W_AGENT * r_agent
    result["score"] = score
    result["r_step"] = r_step
    result["r_agent"] = r_agent
    result["r_proximity"] = r_step
    result["abs_step_shift"] = float(abs_shift)
    result["case"] = 4  # detected
    if abs_shift == 0 and (not gt_agent or r_agent == 1.0):
        result["acc"] = 1.0
    return result
