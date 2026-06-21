#!/bin/bash
# Kelvin2-specific execution script.
# This file is committed intentionally so experiment runs are reproducible.
# It assumes the repo is cloned to:
#   /mnt/scratch2/users/$USER/repos/model-generalization
# It assumes the project venv exists at:
#   /mnt/scratch2/users/$USER/venvs/mcq-generalization
# It assumes Hugging Face cache/token/model files live under:
#   /mnt/scratch2/users/$USER/hf
# Do not put secrets or tokens in this script.
# HF authentication should be done with:
#   hf auth login
#
# MMLU half of 19_redo_abcd_llama8b.sh, split into its own job — the
# original job died mid-ARC-Challenge (job 9209873, 2h limit) and never
# reached MMLU at all, so this starts fresh (no checkpoint to resume).
# See 23_resume_abcd_llama8b_arc.sh for the ARC-Challenge half and the
# 2026-06-21 abcd timeout post-mortem.
#
# No empirical MMLU rate exists yet for this model; budgeted from its
# ARC-Challenge rate (~11.4 min/50q) times the ~1.4x ARC-to-MMLU slowdown
# observed for Qwen-7B (the other model on this same GPU class) at this
# model's smaller scale, plus generous margin.
#
#SBATCH --job-name=mcqgen_fresh_abcd_llama8b_mmlu
#SBATCH --output=logs/mcqgen_fresh_abcd_llama8b_mmlu_%j.out
#SBATCH --error=logs/mcqgen_fresh_abcd_llama8b_mmlu_%j.err
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --partition=k2-gpu-a100mig
#SBATCH --gres=gpu:3g.40gb:1

set -euo pipefail

REPO_ROOT="/mnt/scratch2/users/$USER/repos/model-generalization"
cd "$REPO_ROOT"

mkdir -p logs

SCRATCH="/mnt/scratch2/users/$USER"
VENV_DIR="$SCRATCH/venvs/mcq-generalization"

export HF_HOME="$SCRATCH/hf"
export HF_HUB_CACHE="$HF_HOME/hub"
export MODEL_ROOT="$SCRATCH/models"
export RESULTS_DIR="$SCRATCH/results/mcq-generalization"

# module command is currently unavailable in batch job shells on this
# cluster (verified 2026-06-21 via diagnostic job — venv activation alone
# already resolves to the correct interpreter, so this degrades gracefully
# rather than aborting the whole job if the cluster module system is down).
if command -v module >/dev/null 2>&1; then
    module load python3/3.10.5/gcc-9.3.0
fi

source "$VENV_DIR/bin/activate"

echo "Job ID:  $SLURM_JOB_ID"
echo "Node:    $SLURMD_NODENAME"
echo "GPU:     ${CUDA_VISIBLE_DEVICES:-none}"
echo "Python:  $(python --version)"
echo "Repo:    $REPO_ROOT"

python scripts/run_experiment.py --config config/llama8b_mmlu_redo_abcd.yaml --yes

echo "llama8b abcd MMLU complete."
