#!/usr/bin/env bash
# Second fine-grained annotator: Llama-4-Scout via swift infer (paper Sec. 3.4).
# Input files come from annotation/prepare_annotator_input.py.
#
# Usage:
#   annotation/annotate_techniques_llama.sh <input_jsonl> <output_jsonl> [model]
set -euo pipefail

input=${1:?usage: annotate_techniques_llama.sh <input_jsonl> <output_jsonl> [model]}
output=${2:?missing output path}
model=${3:-meta-llama/Llama-4-Scout-17B-16E-Instruct}

mkdir -p "$(dirname "$output")"

MAX_PIXELS=1003520 \
swift infer \
    --model "$model" \
    --infer_backend vllm \
    --val_dataset "$input" \
    --vllm_gpu_memory_utilization 0.9 \
    --max_new_tokens 2048 \
    --temperature 0 \
    --max_batch_size 8 \
    --use_hf true \
    --result_path "$output"
