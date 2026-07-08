#!/usr/bin/env bash
# Inference with a fine-tuned checkpoint (SFT, GRPO, or self-supervised GRPO).
#
# Usage:
#   inference/finetuned.sh <checkpoint_dir> <test_jsonl> <output_jsonl>
#
# The thinking-format models are evaluated on the CoT test file:
#   data/formatted/<task>/cot_explanation/test.jsonl
# Pass ADAPTERS=<lora_dir> to evaluate a LoRA adapter (e.g. seq-cls baselines,
# with test file data/formatted/<task>/seq_cls/test.jsonl).
set -euo pipefail

checkpoint=${1:?usage: finetuned.sh <checkpoint_dir> <test_jsonl> <output_jsonl>}
test_file=${2:?missing test JSONL}
output=${3:?missing output path}

mkdir -p "$(dirname "$output")"

extra_args=()
if [[ -n "${ADAPTERS:-}" ]]; then
    extra_args+=(--adapters "$ADAPTERS")
fi

MAX_PIXELS=1003520 \
swift infer \
    --model "$checkpoint" \
    --infer_backend vllm \
    --val_dataset "$test_file" \
    --vllm_gpu_memory_utilization 0.95 \
    --max_new_tokens 4096 \
    --temperature 0 \
    --max_batch_size 16 \
    --use_hf true \
    --load_data_args false \
    "${extra_args[@]}" \
    --result_path "$output"
