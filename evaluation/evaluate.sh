#!/usr/bin/env bash
# Score every prediction file in a directory with compute_metrics.py.
#
# Usage:
#   evaluation/evaluate.sh <results_dir> <scores_dir> [pattern] [extra args...]
#
# Examples:
#   evaluation/evaluate.sh results/hateful scores/hateful '*.jsonl' --has_explanation
#   evaluation/evaluate.sh results/armeme scores/armeme '*grpo*' --has_explanation --is_arabic
set -euo pipefail

results_dir=${1:?usage: evaluate.sh <results_dir> <scores_dir> [pattern] [extra args...]}
scores_dir=${2:?missing scores output directory}
pattern=${3:-*.jsonl}
shift $(( $# > 3 ? 3 : $# ))

script_dir=$(cd "$(dirname "$0")" && pwd)
shopt -s nullglob
files=("$results_dir"/$pattern)
if [[ ${#files[@]} -eq 0 ]]; then
    echo "no files matching '$pattern' in $results_dir" >&2
    exit 1
fi

for file in "${files[@]}"; do
    name=$(basename "$file" .jsonl)
    echo "scoring $name"
    python "$script_dir/compute_metrics.py" \
        --data "$file" \
        --out_dir "$scores_dir/$name" \
        "$@"
done
