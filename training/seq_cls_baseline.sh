#!/usr/bin/env bash
# Multimodal sequence-classification baselines (paper Sec. 5.4).
#
# Fine-tunes an MLLM with a classification head (LoRA) instead of generative
# training. These baselines predict a class index only and produce no
# explanations. Data comes from data_prep/prepare_seq_cls_data.py.
#
# Usage:
#   training/seq_cls_baseline.sh <hateful|armeme> [model]
#
# Models used in the paper: Qwen/Qwen3-VL-8B-Instruct, google/gemma-3-12b-it,
# and Qwen/Qwen3-VL-8B-Thinking (ArMeme only).
set -euo pipefail

task=${1:?usage: seq_cls_baseline.sh <hateful|armeme> [model]}
model=${2:-Qwen/Qwen3-VL-8B-Instruct}

DATA_DIR=${DATA_DIR:-data/formatted/$task/seq_cls}
OUTPUT_DIR=${OUTPUT_DIR:-checkpoints/seq_cls/$task/$(basename "$model")}
GPUS=${GPUS:-0}

case "$task" in
    hateful) num_labels=2 ;;
    armeme)  num_labels=4 ;;
    *) echo "unknown task: $task" >&2; exit 1 ;;
esac

CUDA_VISIBLE_DEVICES=$GPUS \
MAX_PIXELS=1003520 \
swift sft \
    --model "$model" \
    --task_type seq_cls \
    --num_labels "$num_labels" \
    --use_chat_template true \
    --use_hf true \
    --dataset "$DATA_DIR/train.jsonl" \
    --val_dataset "$DATA_DIR/dev.jsonl" \
    --torch_dtype bfloat16 \
    --train_type lora \
    --lora_rank 16 \
    --lora_alpha 32 \
    --target_modules all-linear \
    --num_train_epochs 4 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 16 \
    --gradient_accumulation_steps 1 \
    --learning_rate 1e-4 \
    --warmup_ratio 0.05 \
    --max_length 2048 \
    --eval_steps 100 \
    --save_steps 100 \
    --save_total_limit 5 \
    --logging_steps 5 \
    --dataloader_num_workers 4 \
    --output_dir "$OUTPUT_DIR"
