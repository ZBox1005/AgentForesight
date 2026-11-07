"""Local-model online auditing inference on AFTraj-2K.

For every trajectory the auditor is queried at each prefix in temporal order;
the first parseable non-SAFE verdict stops the walk. If no alarm is issued,
the prediction is SAFE.

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
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from inference.data import TrajectoryRecord, load_aftraj
from inference.metrics import aggregate_by_domain, format_report, per_sample_metrics
from inference.prompts import AuditVerdict
from inference.runner import audit_incremental as run_incremental

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
        pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
    )
    if temperature > 0:
        gen_kwargs["temperature"] = temperature
        gen_kwargs["top_p"] = 0.95

    with torch.no_grad():
        out = model.generate(inputs.input_ids, **gen_kwargs)

    new_ids = out[0][inputs.input_ids.shape[1]:]
    return tokenizer.decode(new_ids, skip_special_tokens=True)

def audit_incremental(model, tokenizer, rec: TrajectoryRecord,
                      max_new_tokens: int, temperature: float) -> tuple[AuditVerdict, float, int, int | None]:
    return run_incremental(
        rec,
        lambda messages: generate(model, tokenizer, messages, max_new_tokens, temperature),
    )

def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-path",     required=True,       help="HF repo id or local path of the auditor.")
    p.add_argument("--data-dir",       default="./data",    help="Directory holding aftraj_safe.parquet and aftraj_unsafe.parquet.")
    p.add_argument("--output-dir",     default="./outputs", help="Where to write per_sample.jsonl + results.json.")
    p.add_argument("--device",         default="auto")
    p.add_argument("--max-new-tokens", type=int,   default=1500)
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
            verdict, gen_s, n_calls, detection_step = audit_incremental(
                model, tokenizer, rec, args.max_new_tokens, args.temperature,
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
