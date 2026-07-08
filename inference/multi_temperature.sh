#!/usr/bin/env bash
# Multi-temperature inference over the unlabeled pool (paper Sec. 4.5).
#
# Runs the model at temperatures {0, 0.2, 0.4, 0.8, 1.0} over the unlabeled
# memes. The resulting five prediction files feed
# data_prep/build_unlabeled_training_set.py, which keeps only samples with
# partial agreement (3/5 or 4/5) - the disagreement-based selection used for
# self-supervised GRPO.
#
# Usage:
#   inference/multi_temperature.sh <checkpoint_dir> <pool_jsonl> <output_dir>
set -euo pipefail

checkpoint=${1:?usage: multi_temperature.sh <checkpoint_dir> <pool_jsonl> <output_dir>}
pool=${2:?missing unlabeled pool JSONL}
output_dir=${3:?missing output directory}

mkdir -p "$output_dir"

for temp in 0 0.2 0.4 0.8 1.0; do
    output=$output_dir/temp${temp}.jsonl
    if [[ -f "$output" ]]; then
        echo "skipping temperature $temp (exists)"
        continue
    fi
    MAX_PIXELS=1003520 \
    swift infer \
        --model "$checkpoint" \
        --infer_backend vllm \
        --val_dataset "$pool" \
        --vllm_gpu_memory_utilization 0.95 \
        --max_new_tokens 4096 \
        --temperature "$temp" \
        --max_batch_size 16 \
        --use_hf true \
        --load_data_args false \
        --result_path "$output"
done
