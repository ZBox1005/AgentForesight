"""Build Stage 1 and Stage 2 inputs from the public AFTraj release.

The AFTraj parquets contain the curated trajectories and labels. The public
split manifest identifies the paper evaluation IDs. This script expands the
remaining trajectories into BPPO boundary prompts and GRPO rows.
"""

from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd
from datasets import Dataset

from training.prompts import SYSTEM_PROMPT, build_prompt, build_prompt_incremental


TOOLSAFETY_PREFIXES = (
    "safe_vendor_", "safe_log_", "safe_record_", "safe_health_",
    "safe_customer_", "safe_quarterly_", "safe_market_",
    "indirect_injection_", "wrong_count_", "premature_payment_",
    "credential_lookup_", "stale_db_",
)


def source_domain(conv_id: str) -> str:
    for name in ("math", "coding", "hotpotqa", "gaia"):
        if conv_id.startswith(name + "_"):
            return name
    if conv_id.startswith(TOOLSAFETY_PREFIXES):
        return "toolsafety"
    return "expert_team"


def family_id(conv_id: str, domain: str) -> str:
    if domain in {"toolsafety", "expert_team"}:
        match = re.search(r"_idx(\d+)$", conv_id)
        if not match:
            raise ValueError(f"Missing task index in {conv_id}")
        return f"{domain}_{int(match.group(1)):04d}"
    return re.sub(r"_(safe|unsafe.*|injected)$", "", conv_id)


def load_records(data_dir: Path) -> tuple[list[dict], list[dict]]:
    safe_df = pd.read_parquet(data_dir / "aftraj_safe.parquet")
    unsafe_df = pd.read_parquet(data_dir / "aftraj_unsafe.parquet")
    records = []
    for df, label in ((safe_df, "safe"), (unsafe_df, "unsafe")):
        for row in df.to_dict("records"):
            conv_id = str(row["conv_id"])
            domain = source_domain(conv_id)
            record = {
                "conv_id": conv_id,
                "family_id": family_id(conv_id, domain),
                "domain": domain,
                "record_type": "safe" if label == "safe" else str(row.get("unsafe_source") or "unsafe"),
                "uid": conv_id,
                "label": label,
                "turns": list(row["turns"]),
                "tools": list(row["tools"]),
                "num_turns": int(row["num_turns"]),
                "mistake_step": -1 if label == "safe" else int(row["mistake_step"]),
                "mistake_agent": "" if label == "safe" else str(row.get("mistake_agent") or ""),
                "mistake_reason": "" if label == "safe" else str(row.get("mistake_reason") or ""),
            }
            records.append(record)

    manifest = json.loads((data_dir / "splits_test.json").read_text())
    test_ids = set(manifest["test_safe"]) | set(manifest["test_unsafe"])
    all_ids = [r["conv_id"] for r in records]
    if len(all_ids) != len(set(all_ids)):
        raise ValueError("Duplicate conv_id in AFTraj parquets")
    if not test_ids.issubset(all_ids):
        raise ValueError("Test manifest contains IDs absent from AFTraj parquets")
    train = [r for r in records if r["conv_id"] not in test_ids]
    test = [r for r in records if r["conv_id"] in test_ids]
    if len(train) != 1940 or len(test) != 332:
        raise ValueError(f"Expected 1940 train / 332 test; got {len(train)} / {len(test)}")
    return train, test


def to_verl(record: dict, split: str, index: int, sample_type: str,
            cut_k: int | None = None) -> dict:
    mistake_step = record["mistake_step"]
    if sample_type == "full_safe":
        user_content = build_prompt(record["turns"], record["tools"])
        gt_step, num_turns, label = -1, record["num_turns"], "safe"
    elif sample_type == "full_error":
        user_content = build_prompt(record["turns"], record["tools"])
        gt_step, num_turns, label = mistake_step, record["num_turns"], "unsafe"
    elif sample_type == "boundary_safe":
        if mistake_step <= 0:
            raise ValueError(f"No pre-boundary prefix for {record['conv_id']}")
        user_content = build_prompt_incremental(record["turns"], mistake_step - 1, record["tools"])
        gt_step, num_turns, label = -1, mistake_step, "boundary_safe"
    elif sample_type == "boundary_error":
        user_content = build_prompt_incremental(record["turns"], mistake_step, record["tools"])
        gt_step, num_turns, label = mistake_step, mistake_step + 1, "boundary_error"
    elif sample_type == "safe_prefix":
        if cut_k is None:
            raise ValueError("safe_prefix needs cut_k")
        user_content = build_prompt_incremental(record["turns"], cut_k, record["tools"])
        gt_step, num_turns, label = -1, cut_k + 1, "safe_prefix"
    else:
        raise ValueError(sample_type)

    extra = {
        "split": split, "index": index, "conv_id": record["conv_id"],
        "family_id": record["family_id"], "label": label,
        "sample_type": sample_type, "mistake_step": gt_step,
        "num_turns": num_turns, "original_mistake_step": mistake_step,
        "original_num_turns": record["num_turns"],
        "domain": record["domain"], "record_type": record["record_type"],
        "uid": record["uid"], "mistake_agent": record["mistake_agent"],
        "mistake_reason": record["mistake_reason"], "cut_k": cut_k,
    }
    ground_truth = json.dumps({
        "mistake_step": gt_step, "num_turns": num_turns,
        "mistake_agent": record["mistake_agent"] if gt_step >= 0 else "",
    })
    return {
        "data_source": "agentforesight_audit",
        "prompt": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
        "ability": "audit",
        "reward_model": {"style": "rule", "ground_truth": ground_truth},
        "extra_info": extra,
    }


def build_rows(train: list[dict], test: list[dict], seed: int):
    rng = random.Random(seed)
    k_by_domain: dict[str, list[int]] = defaultdict(list)
    for record in train:
        if record["mistake_step"] >= 0:
            k_by_domain[record["domain"]].append(record["mistake_step"])

    grpo_rows, boundary_rows = [], []
    for record in train:
        if record["label"] == "unsafe":
            grpo_rows.append(to_verl(record, "train", len(grpo_rows), "full_error"))
            if record["mistake_step"] > 0:
                for sample_type in ("boundary_safe", "boundary_error"):
                    boundary_rows.append(to_verl(record, "train", len(boundary_rows), sample_type))
        else:
            want_prefix = rng.random() < 0.5
            if want_prefix and record["num_turns"] > 2:
                ks = k_by_domain.get(record["domain"], [])
                cut_k = rng.choice(ks) if ks else rng.randint(1, record["num_turns"] - 1)
                cut_k = max(1, min(cut_k, record["num_turns"] - 1))
                grpo_rows.append(to_verl(record, "train", len(grpo_rows), "safe_prefix", cut_k))
            else:
                grpo_rows.append(to_verl(record, "train", len(grpo_rows), "full_safe"))

    validation_rows = [
        to_verl(record, "test", index,
                "full_safe" if record["label"] == "safe" else "full_error")
        for index, record in enumerate(test)
    ]
    if len(grpo_rows) != 1940 or len(boundary_rows) != 1902 or len(validation_rows) != 332:
        raise ValueError("Unexpected training-input row counts")
    return grpo_rows, boundary_rows, validation_rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True,
                        help="Directory containing public AFTraj parquets and splits_test.json")
    parser.add_argument("--output-dir", type=Path, default=Path("data/paper_training_inputs"))
    parser.add_argument("--safe-prefix-seed", type=int, default=42)
    args = parser.parse_args()

    train, test = load_records(args.data_dir)
    grpo, boundary, validation = build_rows(train, test, args.safe_prefix_seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for filename, rows in (
        ("grpo_train.parquet", grpo),
        ("bppo_boundary_prompts.parquet", boundary),
        ("grpo_validation.parquet", validation),
    ):
        Dataset.from_list(rows).to_parquet(str(args.output_dir / filename))
        print(f"{filename}: {len(rows)} rows; sample types: "
              f"{dict(Counter(r['extra_info']['sample_type'] for r in rows))}")


if __name__ == "__main__":
    main()
