#!/usr/bin/env bash
# Run the full zero-shot baseline sweep of the paper (Table 8) for one task:
# every open-weight model, with and without CoT prompting. GPT-4.1 is not
# included here; it runs through the Azure OpenAI batch API (see annotation/).
#
# Usage:
#   inference/run_all_zero_shot.sh <hateful|armeme>
set -euo pipefail

task=${1:?usage: run_all_zero_shot.sh <hateful|armeme>}
DATA_DIR=${DATA_DIR:-data/formatted/$task}
RESULTS_DIR=${RESULTS_DIR:-results/zero_shot/$task}
script_dir=$(cd "$(dirname "$0")" && pwd)

# model, backend
models=(
    "Qwen/Qwen3-VL-8B-Instruct         vllm"
    "Qwen/Qwen3-VL-8B-Thinking         vllm"
    "google/gemma-3-12b-it             vllm"
    "meta-llama/Llama-3.2-11B-Vision-Instruct  vllm"
    "meta-llama/Llama-4-Scout-17B-16E-Instruct pt"
    "moonshotai/Kimi-VL-A3B-Instruct   vllm"
    "moonshotai/Kimi-VL-A3B-Thinking   vllm"
    "microsoft/Phi-3.5-vision-instruct pt"
    "QCRI/Fanar-2-Oryx-IVU             vllm"
)

for entry in "${models[@]}"; do
    read -r model backend <<<"$entry"
    name=$(basename "$model")
    for variant in with_explanation cot_explanation; do
        output=$RESULTS_DIR/${name}-${variant}.jsonl
        if [[ -f "$output" ]]; then
            echo "skipping $output (exists)"
            continue
        fi
        "$script_dir/zero_shot.sh" "$model" \
            "$DATA_DIR/$variant/test.jsonl" "$output" "$backend"
    done
done
