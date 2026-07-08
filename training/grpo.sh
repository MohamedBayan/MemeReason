#!/usr/bin/env bash
# Supervised GRPO post-training (paper Sec. 4.4).
#
# Optimizes the composite reward
#   R = a_fmt*R_fmt + a_lbl*R_lbl + a_exp*R_exp + a_met*R_met [+ a_think*R_think]
# on top of an SFT warm-up checkpoint (or the raw backbone for the cold-start
# baseline). Reward functions are implemented in training/rewards.py.
#
# Usage:
#   training/grpo.sh <hateful|armeme> <init_checkpoint> [--no-think-reward]
#
# The default reward set includes R_think with the paper weights
# (0.35 0.35 0.08 0.12 0.10). --no-think-reward reproduces the ablation
# without thinking-length regularization (0.5 0.4 0.05 0.05).
set -euo pipefail

task=${1:?usage: grpo.sh <hateful|armeme> <init_checkpoint> [--no-think-reward]}
model=${2:?missing initialization checkpoint (SFT warm-up, or the base model for cold start)}
think_reward=true
[[ "${3:-}" == "--no-think-reward" ]] && think_reward=false

DATA_DIR=${DATA_DIR:-data/formatted/$task}
OUTPUT_DIR=${OUTPUT_DIR:-checkpoints/grpo/$task}
GPUS=${GPUS:-0,1,2,3}
nproc=$(awk -F, '{print NF}' <<<"$GPUS")
repo_root=$(cd "$(dirname "$0")/.." && pwd)

if $think_reward; then
    reward_funcs=(format_check label_accuracy explanation_length explanation_meteor think_length)
    reward_weights=(0.35 0.35 0.08 0.12 0.10)
else
    reward_funcs=(format_check label_accuracy explanation_length explanation_meteor)
    reward_weights=(0.5 0.4 0.05 0.05)
fi

CUDA_VISIBLE_DEVICES=$GPUS \
NPROC_PER_NODE=$nproc \
MASTER_PORT=${MASTER_PORT:-29501} \
PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True' \
MAX_PIXELS=1003520 \
MEME_TASK=$task \
swift rlhf \
    --rlhf_type grpo \
    --model "$model" \
    --train_type full \
    --dataset "$DATA_DIR/grpo/train.jsonl" \
    --val_dataset "$DATA_DIR/grpo/dev.jsonl" \
    --external_plugins "$repo_root/training/rewards.py" \
    --reward_funcs "${reward_funcs[@]}" \
    --reward_weights "${reward_weights[@]}" \
    --load_from_cache_file true \
    --torch_dtype bfloat16 \
    --num_train_epochs 5 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --learning_rate 1e-6 \
    --gradient_accumulation_steps 1 \
    --warmup_ratio 0.05 \
    --max_length 6000 \
    --max_completion_length 2048 \
    --num_generations 16 \
    --generation_batch_size 16 \
    --temperature 1.0 \
    --top_p 0.85 \
    --repetition_penalty 1.05 \
    --use_vllm true \
    --vllm_gpu_memory_utilization 0.4 \
    --vllm_tensor_parallel_size 2 \
    --vllm_max_model_len 6000 \
    --sleep_level 1 \
    --offload_model true \
    --offload_optimizer true \
    --deepspeed zero3 \
    --overlong_filter true \
    --log_completions true \
    --dataloader_num_workers 4 \
    --save_strategy epoch \
    --eval_strategy epoch \
    --save_total_limit 2 \
    --logging_steps 5 \
    --report_to tensorboard \
    --output_dir "$OUTPUT_DIR"
