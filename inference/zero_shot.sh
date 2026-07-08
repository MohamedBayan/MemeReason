#!/usr/bin/env bash
# Zero-shot / CoT inference with an off-the-shelf MLLM (paper Sec. 5.2).
#
# The prompt is baked into the evaluation file, so switching between direct
# prompting and CoT prompting is just a matter of the test file:
#   direct  data/formatted/<task>/with_explanation/test.jsonl
#   CoT     data/formatted/<task>/cot_explanation/test.jsonl
#
# Usage:
#   inference/zero_shot.sh <model> <test_jsonl> <output_jsonl> [backend]
#
# backend defaults to vllm; use pt for models without vLLM support
# (e.g. Llama-4-Scout, Phi-3.5-vision).
set -euo pipefail

model=${1:?usage: zero_shot.sh <model> <test_jsonl> <output_jsonl> [vllm|pt]}
test_file=${2:?missing test JSONL}
output=${3:?missing output path}
backend=${4:-vllm}

mkdir -p "$(dirname "$output")"

backend_args=()
if [[ "$backend" == "vllm" ]]; then
    backend_args=(--vllm_gpu_memory_utilization 0.95 --max_batch_size 16)
else
    backend_args=(--max_batch_size 1)
fi

MAX_PIXELS=1003520 \
swift infer \
    --model "$model" \
    --infer_backend "$backend" \
    --val_dataset "$test_file" \
    --max_new_tokens 2048 \
    --temperature 0 \
    --use_hf true \
    "${backend_args[@]}" \
    --result_path "$output"
