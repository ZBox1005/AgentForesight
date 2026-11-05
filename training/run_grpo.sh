#!/usr/bin/env bash
# Stage 2 GRPO recipe used for AgentForesight-7B. Run from this repository.
# VERL_DIR must be a checkout of verl at 227acc767e694d118980e0c10f98c44444c37af7.
set -euo pipefail
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
: "${VERL_DIR:?Set VERL_DIR to the pinned verl checkout}"
TRAIN_DATA="${TRAIN_DATA:-${PROJECT_DIR}/data/paper_training_inputs/grpo_train.parquet}"
VAL_DATA="${VAL_DATA:-${PROJECT_DIR}/data/paper_training_inputs/grpo_validation.parquet}"
REWARD_FN="${PROJECT_DIR}/training/reward.py"
MODEL_PATH="${MODEL_PATH:-${PROJECT_DIR}/checkpoints/v16_dpo}"
CKPT_DIR="${CKPT_DIR:-${PROJECT_DIR}/checkpoints/v16_1_grpo}"
GROUP_SIZE=8
TRAIN_BATCH_SIZE=32
TOTAL_EPOCHS=8
LR=1e-6
N_GPUS=2
export VLLM_ATTENTION_BACKEND=FLASH_ATTN
export MKL_SERVICE_FORCE_INTEL=1
export MKL_THREADING_LAYER=GNU
mkdir -p "${CKPT_DIR}"
cd "${VERL_DIR}"

python -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=${TRAIN_DATA} \
    data.val_files=${VAL_DATA} \
    data.train_batch_size=${TRAIN_BATCH_SIZE} \
    data.val_batch_size=100 \
    data.max_prompt_length=8192 \
    data.max_response_length=4096 \
    data.filter_overlong_prompts=True \
    data.truncation=error \
    data.return_raw_chat=True \
    actor_rollout_ref.model.path=${MODEL_PATH} \
    actor_rollout_ref.actor.optim.lr=${LR} \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=32 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    'actor_rollout_ref.actor.checkpoint.save_contents=[model]' \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.65 \
    actor_rollout_ref.rollout.n=${GROUP_SIZE} \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.top_p=1.0 \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.1 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.rollout.enable_chunked_prefill=True \
    actor_rollout_ref.rollout.max_num_batched_tokens=32768 \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=True \
    actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes=4096 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.ref.fsdp_config.param_offload=False \
    algorithm.use_kl_in_reward=False \
    reward.custom_reward_function.path=${REWARD_FN} \
    reward.custom_reward_function.name=compute_score \
    reward.reward_manager.name=dapo \
    +reward.reward_kwargs.max_resp_len=4096 \
    +reward.reward_kwargs.overlong_buffer_cfg.enable=True \
    +reward.reward_kwargs.overlong_buffer_cfg.len=3500 \
    +reward.reward_kwargs.overlong_buffer_cfg.penalty_factor=1.0 \
    +reward.reward_kwargs.overlong_buffer_cfg.log=True \
    trainer.critic_warmup=0 \
    'trainer.logger=[console,wandb]' \
    trainer.project_name=agentforesight \
    trainer.experiment_name=grpo_qwen25_7b_audit_v16_1_from_dpo \
    trainer.default_local_dir=${CKPT_DIR} \
    trainer.n_gpus_per_node=${N_GPUS} \
    trainer.nnodes=1 \
    trainer.save_freq=50 \
    trainer.test_freq=5 \
    trainer.total_epochs=${TOTAL_EPOCHS} \
    trainer.val_before_train=True \
    trainer.max_actor_ckpt_to_keep=999 \
    trainer.resume_mode=auto \
    "$@"
