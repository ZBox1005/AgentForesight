"""V16 DPO data generator — paper Stage 1 (Failure-Boundary Alignment).

Strict implementation of:

    For every unsafe trajectory (τ, k*, a*) ∈ D_unsafe, we construct two
    boundary-pair prompts that share the same trajectory but differ by exactly
    one turn at the decisive step:
      - Pre-boundary prompt τ_{0:k*-1}  → optimal verdict CONTINUE (SAFE)
      - Post-boundary prompt τ_{0:k*}   → optimal verdict ALARM(k*, a*)

    We sample candidate responses from base-policy rollouts, classify them by
    alignment with the verified label, and use deterministic synthetic
    fallbacks when no valid chosen or rejected rollout is available. We partition
    D_pair = D_BS ∪ D_BE by prompt position and optimize jointly.

Key implementation choices:
  - Source: data/paper_training_inputs/dpo_boundary_prompts.parquet, filtered to
    sample_type ∈ {boundary_safe, boundary_error}. ~1902 prompts.
  - Rollouts: vLLM samples N=4 responses per prompt from base Qwen2.5-7B-Instruct
    using the V16 prompt as-is.
  - Classification (per rollout, against verified label):
      * D_BS prompt (target=SAFE):
          chosen   = parses to SAFE
          rejected = parses to ALARM (any step)
      * D_BE prompt (target=ALARM(k*, a*)):
          chosen   = ALARM with pred_step == k* AND agent matches a* (lenient)
          rejected = SAFE OR wrong step OR wrong agent
      * Format-invalid rollouts: dropped (neither chosen nor rejected).
  - Pair construction: 1 (chosen, rejected) per prompt. If pool is empty,
    fall back to deterministic synthetic responses (logged in meta).
  - Output: data/generated_dpo/dpo_pairs.json + classification report.

Usage:
    CUDA_VISIBLE_DEVICES=0,1 python -m training.generate_dpo_pairs \\
        --model_path Qwen/Qwen2.5-7B-Instruct \\
        --train_parquet data/paper_training_inputs/dpo_boundary_prompts.parquet \\
        --output_dir data/generated_dpo \\
        --n_responses 4 --temperature 0.7

Smoke (first 50 prompts, single GPU):
    CUDA_VISIBLE_DEVICES=0 python -m training.generate_dpo_pairs \\
        --model_path Qwen/Qwen2.5-7B-Instruct \\
        --train_parquet data/paper_training_inputs/dpo_boundary_prompts.parquet \\
        --output_dir data/generated_dpo_smoke \\
        --max_prompts 50 --tensor_parallel_size 1
"""
from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from training.reward import (
    parse_v16, REQUIRED_JSON_FIELDS, compute_score,
)


# ── Synthetic fallbacks (used only when base-policy rollouts produce no valid
#    chosen or no valid rejected for a given prompt — minority case). Identical
#    templates to the older draft so format always passes V11 gate.
# ────────────────────────────────────────────────────────────────────────────

def _trim_to_two_sentences(text: str, max_chars: int = 200) -> str:
    text = " ".join((text or "").split())
    if not text:
        return text
    parts: list[str] = []
    buf = ""
    for ch in text:
        buf += ch
        if ch in ".!?":
            parts.append(buf.strip()); buf = ""
            if len(parts) >= 2: break
    if buf.strip(): parts.append(buf.strip())
    out = " ".join(parts[:2])
    if len(out) > max_chars:
        out = out[:max_chars].rstrip()
        if not out.endswith((".", "!", "?")): out += "."
    return out


def _wrap(think: str, ans: dict) -> str:
    return f"<think>\n{think}\n</think>\n<answer>\n{json.dumps(ans, ensure_ascii=False)}\n</answer>"


def synth_chosen(extra: dict, gt: dict) -> str:
    nt = int(gt.get("num_turns", 1)) or 1
    ms = gt.get("mistake_step", -1)
    if ms == -1:
        think = (
            f"Walked through steps 0 to {nt-1}. Each agent action follows the "
            f"prior context appropriately. No step in this window contains a decisive error."
        )
        return _wrap(think, {"answer": "SAFE", "agent": None, "reason": None})
    agent = (gt.get("mistake_agent") or extra.get("mistake_agent") or "the agent").strip()
    raw = (extra.get("mistake_reason") or "").strip() or (
        f"At step {ms}, {agent} produced an incorrect or unsupported result that "
        f"an online auditor would intervene on."
    )
    reason = _trim_to_two_sentences(raw, 200)
    if not (f"step {ms}" in reason.lower() or agent.lower() in reason.lower()):
        reason = f"At step {ms}, {agent}: {reason}"
        reason = _trim_to_two_sentences(reason, 240)
    think = (
        f"Walked through the visible steps. At step {ms}, {agent} acted in a way "
        f"that constitutes a decisive error: {reason} The visible evidence at step {ms} is sufficient."
    )
    return _wrap(think, {"answer": int(ms), "agent": agent, "reason": reason})


def synth_rejected(extra: dict, gt: dict, rng: random.Random) -> str:
    nt = int(gt.get("num_turns", 1)) or 1
    ms = gt.get("mistake_step", -1)
    if ms >= 0:
        think = (
            f"Walked through steps 0 to {nt-1}. The actions appear reasonable in context, "
            f"and I did not identify a decisive error."
        )
        return _wrap(think, {"answer": "SAFE", "agent": None, "reason": None})
    bad = max(0, nt - 1)
    agent = (extra.get("mistake_agent") or "").strip() or "the agent"
    reason = (
        f"Step {bad} by {agent} appears problematic and likely propagates an error downstream."
    )
    think = (
        f"Reviewing the visible steps, step {bad} by {agent} stands out as questionable; "
        f"flagging it as the decisive error."
    )
    return _wrap(think, {"answer": int(bad), "agent": agent, "reason": reason})


# ── Rollout classification (D_BS or D_BE, per verified label) ───────────────

def classify_rollout(rollout: str, gt: dict, sample_type: str) -> str:
    """Returns 'chosen', 'rejected', or 'invalid'."""
    p = parse_v16(rollout)
    if not (p.has_think and p.has_answer and p.json_valid):
        return "invalid"
    obj = p.json_obj
    if not REQUIRED_JSON_FIELDS.issubset(obj.keys()):
        return "invalid"
    answer = obj.get("answer")
    is_safe = (
        (isinstance(answer, str) and answer.upper() == "SAFE")
        or (isinstance(answer, int) and answer == -1)
    )

    if sample_type == "boundary_safe":
        # gt.mistake_step == -1 (cut hides the error). Verified label = SAFE.
        return "chosen" if is_safe else "rejected"

    if sample_type == "boundary_error":
        # gt.mistake_step == k*. Verified label = ALARM(k*, a*).
        if is_safe:
            return "rejected"  # missed
        try:
            pred_step = int(answer)
        except (ValueError, TypeError):
            return "invalid"
        if pred_step != gt.get("mistake_step", -1):
            return "rejected"  # wrong step
        # Step correct; check agent (lenient).
        agent_pred = (obj.get("agent") or "").strip().lower()
        if not agent_pred:
            return "rejected"
        agent_gt = (gt.get("mistake_agent") or "").strip().lower()
        if not agent_gt:
            return "chosen"  # no GT agent → step match suffices
        if agent_pred == agent_gt or agent_gt in agent_pred or agent_pred in agent_gt:
            return "chosen"
        return "rejected"  # wrong agent

    return "invalid"  # safety net for unexpected sample_type


# ── vLLM rollout helper ─────────────────────────────────────────────────────

def format_messages_to_prompt(messages: list[dict], tokenizer) -> str:
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
    )


def vllm_rollout(prompts_text, n, model_path, max_new_tokens, temperature,
                 tensor_parallel_size, gpu_memory_utilization, seed):
    from vllm import LLM, SamplingParams
    llm = LLM(
        model=model_path,
        tensor_parallel_size=tensor_parallel_size,
        gpu_memory_utilization=gpu_memory_utilization,
        dtype="bfloat16",
        trust_remote_code=True,
        seed=seed,
    )
    sampling = SamplingParams(
        n=n, temperature=temperature, top_p=0.95,
        max_tokens=max_new_tokens, seed=seed,
    )
    return llm.generate(prompts_text, sampling, use_tqdm=True)


# ── Main ────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", default="Qwen/Qwen2.5-7B-Instruct",
                        help="Base policy for rollouts (paper: same model used as RL init).")
    parser.add_argument("--train_parquet", required=True,
                        help="V16 train.parquet from prepare_data_v16.py.")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--n_responses", type=int, default=4,
                        help="Rollouts per prompt (paper does not specify; 4 gives reasonable diversity).")
    parser.add_argument("--temperature", type=float, default=0.7,
                        help="Rollout sampling temperature (diverse but coherent).")
    parser.add_argument("--max_new_tokens", type=int, default=1500,
                        help="Per-rollout max generation length.")
    parser.add_argument("--max_prompts", type=int, default=None,
                        help="If set, only the first N prompts (smoke).")
    parser.add_argument("--tensor_parallel_size", type=int, default=2)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.85)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Paper Stage 1 = D_unsafe boundary pairs only. Filter train.parquet to
    # sample_type ∈ {boundary_safe, boundary_error}.
    print(f"[V16-DPO] Loading {args.train_parquet}")
    df = pd.read_parquet(args.train_parquet)
    is_boundary = df["extra_info"].apply(
        lambda e: e["sample_type"] in ("boundary_safe", "boundary_error")
    )
    df = df[is_boundary].reset_index(drop=True)
    if args.max_prompts is not None:
        df = df.head(args.max_prompts).reset_index(drop=True)
    type_counts = Counter(e["sample_type"] for e in df["extra_info"])
    print(f"[V16-DPO] Boundary prompts: {len(df)}  ({dict(type_counts)})")

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)

    print("[V16-DPO] Building chat-template prompts")
    prompts_text = []
    for _, row in tqdm(df.iterrows(), total=len(df)):
        msgs = [dict(m) for m in row["prompt"]]
        prompts_text.append(format_messages_to_prompt(msgs, tokenizer))

    print(f"[V16-DPO] vLLM rollout: model={args.model_path}  n={args.n_responses}  T={args.temperature}")
    print(f"[V16-DPO] Total generations: {len(prompts_text) * args.n_responses}")
    outputs = vllm_rollout(
        prompts_text, n=args.n_responses, model_path=args.model_path,
        max_new_tokens=args.max_new_tokens, temperature=args.temperature,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization, seed=args.seed,
    )

    print("[V16-DPO] Classifying rollouts + building pairs")
    pairs = []
    classification_log: Counter = Counter()
    pair_source: Counter = Counter()
    for i, (_, row) in enumerate(tqdm(list(df.iterrows()), total=len(df))):
        extra = dict(row["extra_info"])
        gt = json.loads(row["reward_model"]["ground_truth"])
        sample_type = extra["sample_type"]
        rollouts = [o.text for o in outputs[i].outputs]

        chosen_pool, rejected_pool = [], []
        for r in rollouts:
            cls = classify_rollout(r, gt, sample_type)
            classification_log[(sample_type, cls)] += 1
            if cls == "chosen":
                chosen_pool.append(r)
            elif cls == "rejected":
                rejected_pool.append(r)
            # 'invalid' rollouts are dropped

        # Pick one chosen + one rejected; fall back to synthetic when pool empty.
        if chosen_pool:
            chosen = rng.choice(chosen_pool)
            chosen_src = "rollout"
        else:
            chosen = synth_chosen(extra, gt)
            chosen_src = "synth"
        if rejected_pool:
            rejected = rng.choice(rejected_pool)
            rejected_src = "rollout"
        else:
            rejected = synth_rejected(extra, gt, rng)
            rejected_src = "synth"
        if chosen.strip() == rejected.strip():
            # Should be impossible (different sample_type or different content);
            # log and skip.
            classification_log[("DROPPED_chosen_eq_rejected", sample_type)] += 1
            continue

        pair_source[(chosen_src, rejected_src)] += 1
        meta = dict(extra)
        meta["chosen_source"] = chosen_src
        meta["rejected_source"] = rejected_src
        meta["rollout_chosen_count"] = len(chosen_pool)
        meta["rollout_rejected_count"] = len(rejected_pool)
        meta["rollout_invalid_count"] = args.n_responses - len(chosen_pool) - len(rejected_pool)
        pairs.append({
            "prompt": [dict(m) for m in row["prompt"]],
            "chosen": chosen,
            "rejected": rejected,
            "meta": meta,
        })

    # Validate every pair under V11 reward (chosen should score high, rejected low).
    # This is a sanity check — pairs that fail are kept (paper takes base-policy as-is)
    # but counted.
    print("[V16-DPO] V11-reward sanity check")
    cs_list, rs_list, anomalies = [], [], Counter()
    for p in pairs:
        gt_dict = {
            "mistake_step": p["meta"]["mistake_step"],
            "num_turns": p["meta"]["num_turns"],
            "mistake_agent": p["meta"].get("mistake_agent", ""),
        }
        cs = compute_score("agentforesight_audit", p["chosen"], gt_dict, p["meta"]).get("score", 0.0)
        rs = compute_score("agentforesight_audit", p["rejected"], gt_dict, p["meta"]).get("score", 0.0)
        cs_list.append(cs); rs_list.append(rs)
        if cs < 0.5: anomalies["chosen_low"] += 1
        if rs > 0.5: anomalies["rejected_high"] += 1
        if cs <= rs: anomalies["chosen_not_better"] += 1

    out_path = out / "dpo_pairs.json"
    with open(out_path, "w") as f:
        json.dump(pairs, f, ensure_ascii=False, indent=2)

    # Diversity report: how many unique chosen/rejected
    unique_chosen = len(set(p["chosen"] for p in pairs))
    unique_rejected = len(set(p["rejected"] for p in pairs))

    print(f"\n[V16-DPO] Wrote {len(pairs)} pairs → {out_path}")
    print(f"[V16-DPO] Per-sample-type rollout classification (sample_type, class → count):")
    for (st, cls), n in sorted(classification_log.items()):
        print(f"  {st:<18} {cls:<8} = {n}")
    print(f"[V16-DPO] Pair source (chosen, rejected → count):")
    for (cs_src, rs_src), n in sorted(pair_source.items()):
        print(f"  chosen={cs_src:<8} rejected={rs_src:<8} = {n}")
    if cs_list:
        print(f"[V16-DPO] V11 reward sanity:")
        print(f"  chosen   score: mean={sum(cs_list)/len(cs_list):+.3f}  min={min(cs_list):+.3f}")
        print(f"  rejected score: mean={sum(rs_list)/len(rs_list):+.3f}  max={max(rs_list):+.3f}")
        print(f"  anomalies: {dict(anomalies) if anomalies else 'none'}")
    print(f"[V16-DPO] Diversity: unique chosen={unique_chosen}/{len(pairs)}  unique rejected={unique_rejected}/{len(pairs)}")


if __name__ == "__main__":
    main()
