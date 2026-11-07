# AgentForesight training recipe

This directory contains the training implementation for AgentForesight-7B. Stage 1 trains a boundary preference policy with BPPO; Stage 2 starts from that checkpoint and trains with GRPO.

## Prepare inputs from the public dataset

Download the public [AFTraj dataset](https://huggingface.co/datasets/ZBox008003/AFTraj). The included `splits_test.json` identifies the paper evaluation trajectories. Generate the Stage 1 boundary prompts and Stage 2 inputs with:

```bash
python -m training.prepare_public_data \
  --data-dir /path/to/downloaded/AFTraj \
  --output-dir data/paper_training_inputs
```

The command writes 1,902 boundary prompts, 1,940 GRPO training rows, and 332 evaluation rows. Generate Stage 1 chosen/rejected pairs from the boundary prompts with the base model:

```bash
python -m training.generate_bppo_pairs \
  --model_path Qwen/Qwen2.5-7B-Instruct \
  --train_parquet data/paper_training_inputs/bppo_boundary_prompts.parquet \
  --output_dir data/paper_training_inputs \
  --n_responses 4 --temperature 0.7
```

This produces `data/paper_training_inputs/bppo_pairs.json`, the default input of `run_bppo.sh`. The prompt builder, response classification, and fallback rules are included in `training/`.

Generated files:

| File | Rows | Use |
| --- | ---: | --- |
| `data/paper_training_inputs/bppo_boundary_prompts.parquet` | 1,902 | Stage 1 pair generation |
| `data/paper_training_inputs/bppo_pairs.json` | 1,902 | Stage 1 training pairs |
| `data/paper_training_inputs/grpo_train.parquet` | 1,940 | Stage 2 training |
| `data/paper_training_inputs/grpo_validation.parquet` | 332 | Stage 2 periodic evaluation; same IDs as the paper test split |

The 332-row split is used as `data.val_files` every five training steps and for the reported paper evaluation.

## Environment and commands

The recorded training environment used `torch==2.9.0`, `transformers==4.57.6`, `bitsandbytes==0.49.2`, `vllm==0.12.0`, and `pyarrow==23.0.1`. The source `verl` checkout was commit `227acc767e694d118980e0c10f98c44444c37af7`. The base model identifier was `Qwen/Qwen2.5-7B-Instruct`.

```bash
MODEL_PATH=Qwen/Qwen2.5-7B-Instruct bash training/run_bppo.sh
git clone https://github.com/volcengine/verl.git /path/to/verl
git -C /path/to/verl checkout 227acc767e694d118980e0c10f98c44444c37af7
VERL_DIR=/path/to/verl bash training/run_grpo.sh
```

The BPPO script uses `device_map="auto"` and an 8-bit Adam optimizer when `bitsandbytes` is installed. Stage 2 uses 8 rollouts per prompt, learning rate `1e-6`, KL coefficient `0.001`, a batch of 32, and the DAPO reward manager. The released checkpoint is the merged Stage 2 step 480 model. The Stage 2 script writes checkpoints every 50 steps.

`generate_bppo_pairs.py` samples base-model responses and classifies them into chosen/rejected pools; deterministic responses fill empty pools. The generated pairs record each response's source. The script requires `vllm`.
