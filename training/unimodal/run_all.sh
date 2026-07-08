#!/usr/bin/env bash
# Train the full unimodal baseline matrix of the paper (Table 7) for one task.
#
# Usage:
#   training/unimodal/run_all.sh <hateful|armeme> <raw_data_dir>
#
# raw_data_dir must contain {train,dev,test}.jsonl with text, class_label and
# img_path fields (the raw dataset, not the chat-formatted files).
set -euo pipefail

task=${1:?usage: run_all.sh <hateful|armeme> <raw_data_dir>}
data_dir=${2:?missing raw data directory}
script_dir=$(cd "$(dirname "$0")" && pwd)

text_models=(
    bert-base-multilingual-cased
    xlm-roberta-base
)
image_models=(
    google/vit-base-patch16-224
    microsoft/beit-base-patch16-224
    facebook/convnext-large-224
    facebook/dinov2-large
    microsoft/resnet-101
    microsoft/swin-large-patch4-window7-224
)
if [[ "$task" == "hateful" ]]; then
    text_models+=(distilbert-base-uncased)
else
    text_models+=(aubmindlab/bert-base-arabertv2 ahmedabdelali/bert-base-qarib)
fi

for model in "${text_models[@]}"; do
    name=$(basename "$model")
    python "$script_dir/train_text_classifier.py" \
        --data_dir "$data_dir" \
        --model_name "$model" \
        --output_dir "checkpoints/unimodal/text/$task/$name" \
        --result_dir "results/unimodal/text/$task/$name"
done

for model in "${image_models[@]}"; do
    name=$(basename "$model")
    python "$script_dir/train_image_classifier.py" \
        --data_dir "$data_dir" \
        --model_name "$model" \
        --output_dir "checkpoints/unimodal/image/$task/$name" \
        --result_dir "results/unimodal/image/$task/$name"
done
