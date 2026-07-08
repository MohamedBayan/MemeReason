#!/usr/bin/env bash
# Self-supervised GRPO on unlabeled memes (paper Sec. 4.5).
#
# Continues training the best supervised GRPO checkpoint on 2,000
# disagreement-sampled unlabeled memes, replacing the gold-label reward with
# a majority-vote consensus pseudo-reward:
#   R = 0.30*R_fmt + 0.20*R_mv + 0.20*R_exp + 0.30*R_think
#
# Build the unlabeled training set first with
# data_prep/build_unlabeled_training_set.py (see inference/multi_temperature.sh
# for generating the multi-temperature predictions it consumes).
#
# Usage:
#   training/grpo_self_supervised.sh <hateful|armeme> <supervised_grpo_checkpoint>
set -euo pipefail

task=${1:?usage: grpo_self_supervised.sh <hateful|armeme> <supervised_grpo_checkpoint>}
model=${2:?missing supervised GRPO checkpoint}

TRAIN_FILE=${TRAIN_FILE:-data/formatted/$task/unlabeled/train_2000.jsonl}
OUTPUT_DIR=${OUTPUT_DIR:-checkpoints/grpo_self_supervised/$task}
GPUS=${GPUS:-0,1,2,3}
nproc=$(awk -F, '{print NF}' <<<"$GPUS")
repo_root=$(cd "$(dirname "$0")/.." && pwd)

CUDA_VISIBLE_DEVICES=$GPUS \
NPROC_PER_NODE=$nproc \
MASTER_PORT=${MASTER_PORT:-29507} \
PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True' \
MAX_PIXELS=1003520 \
MEME_TASK=$task \
MAJORITY_VOTE_DEBUG=${MAJORITY_VOTE_DEBUG:-0} \
swift rlhf \
    --rlhf_type grpo \
    --model "$model" \
    --train_type full \
    --dataset "$TRAIN_FILE" \
    --external_plugins "$repo_root/training/rewards.py" \
    --reward_funcs format_check majority_vote_label explanation_length think_length \
    --reward_weights 0.30 0.20 0.20 0.30 \
    --load_from_cache_file true \
    --torch_dtype bfloat16 \
    --num_train_epochs 4 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --learning_rate 2e-7 \
    --kl_coef 0.15 \
    --gradient_accumulation_steps 4 \
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
    --save_total_limit 2 \
    --logging_steps 5 \
    --report_to tensorboard \
    --output_dir "$OUTPUT_DIR"
