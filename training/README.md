# AgentForesight training recipe

This directory contains the training implementation used for the reported V16 AgentForesight-7B model. Stage 1 trains a boundary preference policy with DPO; Stage 2 starts from that checkpoint and trains with GRPO and the V11 reward.

## Original inputs and provenance

The local original run used 1,902 chosen/rejected DPO pairs, 1,940 GRPO training rows, and 332 rows for periodic Stage 2 evaluation. The DPO generator consumed 1,902 boundary prompt rows. Those exact processed files are under source and privacy review and are **not included in this public code release**. Contact the authors if you need them before publication. The public [AFTraj dataset](https://huggingface.co/datasets/ZBox008003/AFTraj) has the 2,272 trajectories in the paper's train/test manifests. Its export normalizes tool-call IDs, so rebuilding prompts from that export is semantically similar but not byte-identical to the original inputs.

Expected input paths when the processed files are available:

| File | Rows | Use |
| --- | ---: | --- |
| `data/paper_training_inputs/dpo_boundary_prompts.parquet` | 1,902 | Optional Stage 1 pair generation |
| `data/paper_training_inputs/dpo_pairs.json` | 1,902 | Exact Stage 1 training pairs |
| `data/paper_training_inputs/grpo_train.parquet` | 1,940 | Stage 2 training |
| `data/paper_training_inputs/grpo_validation.parquet` | 332 | Stage 2 periodic evaluation; same IDs as the paper test split |

The 332-row split was used as `data.val_files` every five training steps and to evaluate the reported checkpoint. It should not be described as an untouched holdout. Running this recipe is stochastic and has not been independently rerun from scratch for this release.

## Environment and commands

The original environment recorded `torch==2.9.0`, `transformers==4.57.6`, `bitsandbytes==0.49.2`, `vllm==0.12.0`, and `pyarrow==23.0.1`. The source `verl` checkout was commit `227acc767e694d118980e0c10f98c44444c37af7`. A local modification to `verl/trainer/ppo/ray_trainer.py` wrote a step counter to `/tmp`; the V11 reward does not read it. We have not established whether the exact base-model revision was pinned; the identifier was `Qwen/Qwen2.5-7B-Instruct`.

```bash
MODEL_PATH=Qwen/Qwen2.5-7B-Instruct bash training/run_dpo.sh
git clone https://github.com/volcengine/verl.git /path/to/verl
git -C /path/to/verl checkout 227acc767e694d118980e0c10f98c44444c37af7
VERL_DIR=/path/to/verl bash training/run_grpo.sh
```

The DPO script uses `device_map="auto"` and an 8-bit Adam optimizer when `bitsandbytes` is installed. It does not wrap the model in FSDP. Stage 2 uses 8 rollouts per prompt, learning rate `1e-6`, KL coefficient `0.001`, a batch of 32, and the DAPO reward manager. The final paper checkpoint is the merged Stage 2 step 480 model. The Stage 2 script writes checkpoints every 50 steps.

`generate_dpo_pairs.py` shows the original pair construction: base-model rollouts supply a chosen or rejected response when available; deterministic synthetic responses fill missing pools. The saved DPO pairs include the source of each response. The script requires the boundary prompt parquet and `vllm`.
