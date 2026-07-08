#!/usr/bin/env bash
# SFT warm-up for Qwen3-VL-8B-Thinking (paper Sec. 4.3).
#
# Three variants, trained in order (cotd starts from the cls-fg-exp checkpoint):
#   cls-exp     labels + explanations
#   cls-fg-exp  labels + fine-grained annotations + explanations
#   cotd        distilled chain-of-thought on top of the cls-fg-exp checkpoint
#
# Usage:
#   training/sft_warmup.sh <hateful|armeme> <cls-exp|cls-fg-exp|cotd> [base_model]
#
# base_model defaults to Qwen/Qwen3-VL-8B-Thinking for cls-exp / cls-fg-exp.
# For cotd you must pass the best cls-fg-exp checkpoint directory.
#
# Expects the formatted data produced by data_prep/prepare_training_data.py
# under data/formatted/. Override DATA_DIR / OUTPUT_DIR / GPUS as needed.
set -euo pipefail

task=${1:?usage: sft_warmup.sh <hateful|armeme> <cls-exp|cls-fg-exp|cotd> [base_model]}
variant=${2:?missing variant: cls-exp | cls-fg-exp | cotd}
model=${3:-Qwen/Qwen3-VL-8B-Thinking}

DATA_DIR=${DATA_DIR:-data/formatted/$task}
OUTPUT_DIR=${OUTPUT_DIR:-checkpoints/sft/$task/$variant}
GPUS=${GPUS:-0,1,2,3}
nproc=$(awk -F, '{print NF}' <<<"$GPUS")

# Hyperparameters as used for the paper runs; see Appendix A for details.
case "$task" in
    hateful) lr=1e-6; epochs=3; shuffle=false ;;
    armeme)  lr=1e-5; epochs=5; shuffle=true ;;
    *) echo "unknown task: $task" >&2; exit 1 ;;
esac

case "$variant" in
    cls-exp)
        datasets=("$DATA_DIR/classification_only/train.jsonl" "$DATA_DIR/with_explanation/train.jsonl")
        val_dataset=$DATA_DIR/classification_only/dev.jsonl
        ;;
    cls-fg-exp)
        datasets=("$DATA_DIR/fine_grained/train.jsonl" "$DATA_DIR/classification_only/train.jsonl" "$DATA_DIR/with_explanation/train.jsonl")
        val_dataset=$DATA_DIR/classification_only/dev.jsonl
        ;;
    cotd)
        if [[ "$model" == "Qwen/Qwen3-VL-8B-Thinking" ]]; then
            echo "cotd continues training from the cls-fg-exp checkpoint; pass it as the third argument" >&2
            exit 1
        fi
        datasets=("$DATA_DIR/thinking/train.jsonl")
        val_dataset=$DATA_DIR/thinking/dev.jsonl
        lr=1e-6; epochs=7; shuffle=true
        ;;
    *) echo "unknown variant: $variant" >&2; exit 1 ;;
esac

# ignore_empty_think masks the placeholder <think></think> tags out of the
# loss for the variants that carry no reasoning trace.
loss_scale_args=(--loss_scale ignore_empty_think)
if [[ "$variant" == "cotd" ]]; then
    loss_scale_args=()
fi

CUDA_VISIBLE_DEVICES=$GPUS \
NPROC_PER_NODE=$nproc \
MASTER_PORT=${MASTER_PORT:-29501} \
PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True' \
MAX_PIXELS=1003520 \
swift sft \
    --model "$model" \
    --train_type full \
    --dataset "${datasets[@]}" \
    --val_dataset "$val_dataset" \
    --torch_dtype bfloat16 \
    --num_train_epochs "$epochs" \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --learning_rate "$lr" \
    "${loss_scale_args[@]}" \
    --gradient_accumulation_steps 1 \
    --warmup_ratio 0.05 \
    --max_length 4096 \
    --dataset_shuffle "$shuffle" \
    --deepspeed zero3 \
    --dataloader_num_workers 4 \
    --save_strategy epoch \
    --eval_strategy epoch \
    --save_total_limit 2 \
    --logging_steps 5 \
    --report_to tensorboard \
    --output_dir "$OUTPUT_DIR"
