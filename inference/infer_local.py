"""Local-model online auditing inference on AFTraj-2K.

For every safe trajectory the auditor is queried once on the full trajectory
(binary SAFE / UNSAFE classification). For every unsafe trajectory the
auditor is queried at each prefix in temporal order; the first non-SAFE
verdict is recorded as the predicted decisive step.

Per-sample results are streamed to ``per_sample.jsonl`` and a summary is
written to ``results.json``.

Example::

    python -m inference.infer_local \\
        --model-path <hf_repo_or_local_path> \\
        --data-dir   ./data \\
        --output-dir ./outputs/af7b
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from inference.data import TrajectoryRecord, load_aftraj
from inference.metrics import aggregate_by_domain, format_report, per_sample_metrics
from inference.prompts import AuditVerdict, build_messages, parse_response

def load_model(model_path: str, device: str = "auto") -> tuple:
    print(f"Loading auditor from {model_path} ...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map=device,
    )
    model.eval()
    print(f"  loaded; device={model.device}")
    return model, tokenizer

def generate(model, tokenizer, messages: list[dict],
             max_new_tokens: int, temperature: float) -> str:
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer([text], return_tensors="pt").to(model.device)

    gen_kwargs = dict(
        max_new_tokens=max_new_tokens,
        do_sample=temperature > 0,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id,
    )
    if temperature > 0:
        gen_kwargs["temperature"] = temperature
        gen_kwargs["top_p"] = 0.95

    with torch.no_grad():
        out = model.generate(inputs.input_ids, **gen_kwargs)

    new_ids = out[0][inputs.input_ids.shape[1]:]
    return tokenizer.decode(new_ids, skip_special_tokens=False)

def audit_full(model, tokenizer, rec: TrajectoryRecord,
               max_new_tokens: int, temperature: float) -> tuple[AuditVerdict, float, int]:
    msgs = build_messages(rec.turns, tools=rec.tools, current_step=None)
    t0 = time.time()
    raw = generate(model, tokenizer, msgs, max_new_tokens, temperature)
    return parse_response(raw), time.time() - t0, 1

def audit_incremental(model, tokenizer, rec: TrajectoryRecord,
                      max_new_tokens: int, temperature: float) -> tuple[AuditVerdict, float, int, int]:
    total_time = 0.0
    n_calls = 0
    last: AuditVerdict | None = None
    detection_step = rec.num_turns - 1                                         
    for k in range(rec.num_turns):
        msgs = build_messages(rec.turns, tools=rec.tools, current_step=k)
        t0 = time.time()
        raw = generate(model, tokenizer, msgs, max_new_tokens, temperature)
        total_time += time.time() - t0
        n_calls += 1
        last = parse_response(raw)
        if last.valid and last.pred_step >= 0:
            detection_step = k
            return last, total_time, n_calls, detection_step
    return last or AuditVerdict(-1, "", "", False, ""), total_time, n_calls, detection_step

def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-path",     required=True,       help="HF repo id or local path of the auditor.")
    p.add_argument("--data-dir",       default="./data",    help="Directory holding aftraj_safe.parquet and aftraj_unsafe.parquet.")
    p.add_argument("--output-dir",     default="./outputs", help="Where to write per_sample.jsonl + results.json.")
    p.add_argument("--device",         default="auto")
    p.add_argument("--max-new-tokens", type=int,   default=2048)
    p.add_argument("--temperature",    type=float, default=0.0)
    p.add_argument("--max-trajs",      type=int,   default=None, help="Optional cap (smoke test).")
    p.add_argument("--domains",        type=str,   default=None, help="Comma-separated domain whitelist.")
    p.add_argument("--paper-test-split", action="store_true",
                   help="Restrict to the held-out test split used in the paper's main table.")
    p.add_argument("--macro-domain",   action="store_true",
                   help="Aggregate metrics by the paper's 3-way macro buckets (Math/Coding/Agentic).")
    args = p.parse_args()

    domains = [d.strip() for d in args.domains.split(",")] if args.domains else None

    records = load_aftraj(args.data_dir, domains=domains, limit=args.max_trajs,
                          paper_test_split=args.paper_test_split)
    print(f"Loaded {len(records)} trajectories from {args.data_dir}")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = out_dir / "per_sample.jsonl"
    json_path  = out_dir / "results.json"

    model, tokenizer = load_model(args.model_path, args.device)

    per_sample: list[dict] = []
    with open(jsonl_path, "w", encoding="utf-8") as f:
        config = {
            "model_path":     args.model_path,
            "data_dir":       str(Path(args.data_dir).resolve()),
            "max_new_tokens": args.max_new_tokens,
            "temperature":    args.temperature,
            "domains":        domains,
        }
        f.write(json.dumps({"_record_type": "config", **config}, ensure_ascii=False) + "\n")
        f.flush()

        for rec in tqdm(records, desc="audit"):
            if rec.label == "safe":
                verdict, gen_s, n_calls = audit_full(
                    model, tokenizer, rec, args.max_new_tokens, args.temperature
                )
                detection_step = -1
            else:
                verdict, gen_s, n_calls, detection_step = audit_incremental(
                    model, tokenizer, rec, args.max_new_tokens, args.temperature
                )

            row = {
                "conv_id":        rec.conv_id,
                "domain":         rec.domain,
                "label":          rec.label,
                "gt_step":        rec.mistake_step,
                "pred_step":      verdict.pred_step,
                "pred_agent":     verdict.pred_agent,
                "pred_reason":    verdict.pred_reason,
                "format_valid":   verdict.valid,
                "detection_step": detection_step,
                "num_turns":      rec.num_turns,
                "num_calls":      n_calls,
                "gen_time_s":     round(gen_s, 2),
                "raw_response":   verdict.raw_response[:3000],
                **per_sample_metrics(verdict.pred_step, rec.mistake_step),
            }
            per_sample.append(row)
            f.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
            f.flush()

    by_domain = aggregate_by_domain(per_sample, macro=args.macro_domain)
    print("\n" + format_report(by_domain))

    json_path.write_text(
        json.dumps({"by_domain": by_domain, "per_sample": per_sample}, indent=2, default=str),
        encoding="utf-8",
    )
    print(f"\nResults written to {json_path}")

if __name__ == "__main__":
    main()
