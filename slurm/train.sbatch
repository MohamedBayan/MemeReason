#!/bin/bash
# SLURM wrapper: submit any repo script as a GPU job, e.g.
#   sbatch slurm/train.sbatch training/grpo.sh hateful checkpoints/sft/hateful/cotd/checkpoint-XXXX
#
# Adjust the partition/account/QoS to your cluster. The paper's training runs
# used 4x NVIDIA H200 (141GB); single-GPU jobs (seq-cls, unimodal baselines)
# fit on one A100 80GB.
#SBATCH -J memereason
#SBATCH -p gpu-H200
#SBATCH -A H200
#SBATCH -q h200_qos
#SBATCH --gres=gpu:H200_141GB:4
#SBATCH -c 32
#SBATCH --mem=30G
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err

set -euo pipefail
mkdir -p logs

module load cuda12.2/toolkit/12.2.1 2>/dev/null || true
source "$HOME/anaconda3/bin/activate" "${CONDA_ENV:-qswift}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

bash "$@"
