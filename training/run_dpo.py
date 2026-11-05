"""Stage 1 boundary preference optimization for AgentForesight-7B.

The implementation uses an explicit DPO loss and automatic model device
placement. Run `bash training/run_dpo.sh` with the paper's processed pairs.
"""

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    get_cosine_schedule_with_warmup,
)
from tqdm import tqdm


# ============================================================
# Dataset
# ============================================================

class DPODataset(Dataset):
    """DPO dataset: prompt + chosen + rejected."""

    def __init__(self, data_path: str, tokenizer, max_prompt_len: int = 6144,
                 max_response_len: int = 3072):
        with open(data_path) as f:
            self.pairs = json.load(f)

        self.tokenizer = tokenizer
        self.max_prompt_len = max_prompt_len
        self.max_response_len = max_response_len

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        pair = self.pairs[idx]
        return {
            "prompt": pair["prompt"],       # list of message dicts
            "chosen": pair["chosen"],       # response string
            "rejected": pair["rejected"],   # response string
            "meta": pair.get("meta", {}),
        }


def collate_dpo(batch, tokenizer, max_length=9216):
    """Tokenize and collate a batch of DPO pairs.

    Returns dict with:
        chosen_input_ids, chosen_labels, chosen_attention_mask
        rejected_input_ids, rejected_labels, rejected_attention_mask
    """
    result = {}

    for key in ["chosen", "rejected"]:
        all_input_ids = []
        all_labels = []

        for item in batch:
            # Build full conversation: prompt + response
            messages = item["prompt"] + [{"role": "assistant", "content": item[key]}]
            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False,
            )

            # Also build prompt-only to find where response starts
            prompt_text = tokenizer.apply_chat_template(
                item["prompt"], tokenize=False, add_generation_prompt=True,
            )

            full_ids = tokenizer.encode(text, max_length=max_length, truncation=True)
            prompt_ids = tokenizer.encode(prompt_text, max_length=max_length, truncation=True)

            # Labels: -100 for prompt tokens, actual ids for response tokens
            labels = [-100] * len(prompt_ids) + full_ids[len(prompt_ids):]
            # Pad labels to match full_ids length
            if len(labels) < len(full_ids):
                labels = labels + [-100] * (len(full_ids) - len(labels))
            labels = labels[:len(full_ids)]

            all_input_ids.append(full_ids)
            all_labels.append(labels)

        # Pad to same length
        max_len = max(len(ids) for ids in all_input_ids)
        pad_id = tokenizer.eos_token_id

        padded_ids = []
        padded_labels = []
        padded_mask = []

        for ids, labs in zip(all_input_ids, all_labels):
            pad_len = max_len - len(ids)
            padded_ids.append(ids + [pad_id] * pad_len)
            padded_labels.append(labs + [-100] * pad_len)
            padded_mask.append([1] * len(ids) + [0] * pad_len)

        result[f"{key}_input_ids"] = torch.tensor(padded_ids)
        result[f"{key}_labels"] = torch.tensor(padded_labels)
        result[f"{key}_attention_mask"] = torch.tensor(padded_mask)

    return result


# ============================================================
# DPO Loss
# ============================================================

def compute_log_probs(model, input_ids, labels, attention_mask):
    """Compute per-token log probabilities for response tokens only."""
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = outputs.logits

    # Shift for causal LM
    shift_logits = logits[:, :-1, :]
    shift_labels = labels[:, 1:]

    # Per-token log probs
    log_probs = F.log_softmax(shift_logits, dim=-1)
    token_log_probs = log_probs.gather(2, shift_labels.unsqueeze(2).clamp(min=0)).squeeze(2)

    # Mask: only count response tokens (labels != -100)
    mask = (shift_labels != -100).float()
    # Use SUM (not average) — standard DPO formulation: log π(y|x) = Σ_t log π(y_t|...)
    # Average would dilute the signal by ~200x (response length), making β×ratio ≈ 0
    seq_log_probs = (token_log_probs * mask).sum(dim=1)

    return seq_log_probs


def dpo_loss(policy_chosen_logps, policy_rejected_logps,
             ref_chosen_logps, ref_rejected_logps, beta=0.1):
    """Standard DPO loss.

    L = -log σ(β × ((log π(y_w|x) - log π_ref(y_w|x)) - (log π(y_l|x) - log π_ref(y_l|x))))
    """
    chosen_rewards = beta * (policy_chosen_logps - ref_chosen_logps)
    rejected_rewards = beta * (policy_rejected_logps - ref_rejected_logps)

    loss = -F.logsigmoid(chosen_rewards - rejected_rewards).mean()

    # Metrics
    with torch.no_grad():
        reward_margin = (chosen_rewards - rejected_rewards).mean().item()
        accuracy = ((chosen_rewards > rejected_rewards).float().mean().item())

    return loss, reward_margin, accuracy


# ============================================================
# Training loop
# ============================================================

def train_dpo(args):
    """Main DPO training loop."""
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    if getattr(args, "disable_thinking", False):
        # Force Qwen3 chat_template to always render with enable_thinking=False,
        # regardless of how apply_chat_template is called. This works by prepending
        # a Jinja `{% set enable_thinking = false %}` statement that overrides any
        # value passed in. Non-destructive — preserves all other template logic.
        orig = tokenizer.chat_template or ""
        marker = "{%- set enable_thinking = false -%}\n"
        if orig and "enable_thinking" in orig and marker not in orig:
            tokenizer.chat_template = marker + orig
            print("[disable_thinking] Prepended `{% set enable_thinking = false %}` to chat_template.")
        else:
            print("[disable_thinking] chat_template already patched or does not use enable_thinking — no-op.")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load policy model
    print(f"Loading policy model from {args.model_path}...")
    policy_model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="eager",
    )
    policy_model.gradient_checkpointing_enable()

    # Load reference model (frozen)
    print(f"Loading reference model from {args.model_path}...")
    ref_model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    ref_model.eval()
    for param in ref_model.parameters():
        param.requires_grad = False

    # Dataset
    dataset = DPODataset(args.dpo_data, tokenizer)
    print(f"DPO dataset: {len(dataset)} pairs")

    def collate_fn(batch):
        return collate_dpo(batch, tokenizer, max_length=args.max_length)

    dataloader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True,
        collate_fn=collate_fn, drop_last=True,
    )

    # Optimizer — 8-bit Adam to reduce memory (96GB fp32 → ~48GB)
    try:
        import bitsandbytes as bnb
        optimizer = bnb.optim.AdamW8bit(
            policy_model.parameters(), lr=args.lr, weight_decay=0.01,
        )
        print("Using 8-bit Adam (bitsandbytes)")
    except ImportError:
        optimizer = torch.optim.AdamW(
            policy_model.parameters(), lr=args.lr, weight_decay=0.01,
        )
        print("WARNING: bitsandbytes not available, using fp32 AdamW (may OOM)")

    total_steps = (len(dataloader) * args.epochs) // args.grad_accum
    warmup_steps = min(50, total_steps // 10)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps,
    )

    print(f"\n=== DPO Training Config ===")
    print(f"  Pairs: {len(dataset)}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Steps/epoch: {len(dataloader)}")
    print(f"  Total steps: {total_steps}")
    print(f"  LR: {args.lr}")
    print(f"  Beta: {args.beta}")
    print(f"  Grad accum: {args.grad_accum}")
    print(f"  Effective batch: {args.batch_size * args.grad_accum}")
    print(f"  Optimizer steps: {total_steps}")
    print(f"  Warmup: {warmup_steps}")

    # Training
    global_step = 0
    optimizer.zero_grad()
    for epoch in range(args.epochs):
        policy_model.train()
        epoch_loss = 0
        epoch_acc = 0
        epoch_margin = 0

        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{args.epochs}")
        for batch in pbar:
            device = next(policy_model.parameters()).device

            # Move to device
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.to(device)

            # Compute policy log probs
            policy_chosen_logps = compute_log_probs(
                policy_model,
                batch["chosen_input_ids"],
                batch["chosen_labels"],
                batch["chosen_attention_mask"],
            )
            policy_rejected_logps = compute_log_probs(
                policy_model,
                batch["rejected_input_ids"],
                batch["rejected_labels"],
                batch["rejected_attention_mask"],
            )

            # Compute reference log probs
            with torch.no_grad():
                ref_chosen_logps = compute_log_probs(
                    ref_model,
                    batch["chosen_input_ids"],
                    batch["chosen_labels"],
                    batch["chosen_attention_mask"],
                )
                ref_rejected_logps = compute_log_probs(
                    ref_model,
                    batch["rejected_input_ids"],
                    batch["rejected_labels"],
                    batch["rejected_attention_mask"],
                )

            # DPO loss
            loss, margin, acc = dpo_loss(
                policy_chosen_logps, policy_rejected_logps,
                ref_chosen_logps, ref_rejected_logps,
                beta=args.beta,
            )

            # Backward with gradient accumulation
            scaled_loss = loss / args.grad_accum
            scaled_loss.backward()

            if (global_step + 1) % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(policy_model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            epoch_loss += loss.item()
            epoch_acc += acc
            epoch_margin += margin
            global_step += 1

            pbar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "acc": f"{acc:.2f}",
                "margin": f"{margin:.3f}",
                "lr": f"{scheduler.get_last_lr()[0]:.2e}",
            })

        n_batches = len(dataloader)
        print(f"Epoch {epoch+1}: loss={epoch_loss/n_batches:.4f} "
              f"acc={epoch_acc/n_batches:.2f} margin={epoch_margin/n_batches:.3f}")

        # Per-epoch checkpoint (added for V16: monitor val + allow rollback)
        epoch_dir = Path(args.output_dir) / f"epoch_{epoch+1}"
        epoch_dir.mkdir(parents=True, exist_ok=True)
        print(f"  Saving epoch-{epoch+1} ckpt → {epoch_dir}")
        policy_model.save_pretrained(epoch_dir)
        tokenizer.save_pretrained(epoch_dir)
        # Sidecar metrics file for quick monitoring without loading model
        with open(epoch_dir / "epoch_metrics.json", "w") as f:
            json.dump({
                "epoch": epoch + 1,
                "loss": epoch_loss / n_batches,
                "acc": epoch_acc / n_batches,
                "margin": epoch_margin / n_batches,
            }, f, indent=2)

    # Final symlink (compat with existing eval scripts that expect output_dir/ as the model)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nSaving final DPO model to {out_dir}...")
    policy_model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)
    print("Done!")

    # Save training config
    config = vars(args)
    config["total_steps"] = total_steps
    config["dataset_size"] = len(dataset)
    with open(out_dir / "dpo_config.json", "w") as f:
        json.dump(config, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description="AgentForesight Stage 1 DPO training")
    parser.add_argument("--model_path", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--dpo_data", type=str, default="data/paper_training_inputs/dpo_pairs.json")
    parser.add_argument("--output_dir", type=str, default="checkpoints/v16_dpo")
    parser.add_argument("--beta", type=float, default=0.1, help="DPO temperature")
    parser.add_argument("--lr", type=float, default=5e-7)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=1, help="Global batch size")
    parser.add_argument("--grad_accum", type=int, default=16, help="Gradient accumulation steps")
    parser.add_argument("--max_length", type=int, default=8192, help="Max total sequence length")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--disable_thinking", action="store_true",
        help="Patch tokenizer's chat_template to remove Qwen3 native '<think>\\n' prefix. "
             "Use for V10-Qwen-nothink variant. Has no effect on Llama/other non-Qwen3 models.",
    )
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    train_dpo(args)


if __name__ == "__main__":
    main()
