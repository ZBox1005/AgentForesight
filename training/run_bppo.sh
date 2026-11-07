#!/usr/bin/env bash
# Stage 1 boundary preference optimization with the original hyperparameters.
set -euo pipefail
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_DIR}"
MODEL_PATH="${MODEL_PATH:-Qwen/Qwen2.5-7B-Instruct}"
PAIR_DATA="${PAIR_DATA:-data/paper_training_inputs/bppo_pairs.json}"
OUTPUT_DIR="${OUTPUT_DIR:-checkpoints/bppo}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python -m training.run_bppo \
  --model_path "${MODEL_PATH}" \
  --pair_data "${PAIR_DATA}" \
  --output_dir "${OUTPUT_DIR}" \
  --beta 0.1 --lr 5e-7 --epochs 3 \
  --batch_size 1 --grad_accum 16 --max_length 8192 --seed 42 \
  "$@"
