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
# GPU resource note:
# Llama 3.1 70B needs 2x Kelvin2 A100s (160GB total); device=auto in the config handles multi-GPU splitting via accelerate automatically.
#
# MMLU half of 22_redo_abcd_llama70b.sh, split into its own job — the
# original job never reached MMLU at all (died mid-ARC-Challenge), so this
# starts fresh (no checkpoint to resume). See
# 25_resume_abcd_llama70b_arc.sh for the ARC-Challenge half and the
# 2026-06-21 abcd timeout post-mortem.
#
# Budgeted from this model's ARC-Challenge rate (~23.9 min/50q): unlike the
# smaller Qwen-7B/32B (which were ~1.4-2x slower on MMLU than ARC), Qwen-72B
# -- the other model on matched 2x-A100 hardware -- showed almost no
# ARC-vs-MMLU gap (~20.1 vs ~22 min/50q), so Llama-70B's MMLU rate is
# assumed close to its ARC rate, with margin for the assumption being an
# extrapolation rather than a direct measurement.
#
#SBATCH --job-name=mcqgen_fresh_abcd_llama70b_mmlu
#SBATCH --output=logs/mcqgen_fresh_abcd_llama70b_mmlu_%j.out
#SBATCH --error=logs/mcqgen_fresh_abcd_llama70b_mmlu_%j.err
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=160G
#SBATCH --partition=k2-gpu-a100
#SBATCH --gres=gpu:a100:2

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

module load python3/3.10.5/gcc-9.3.0

source "$VENV_DIR/bin/activate"

echo "Job ID:  $SLURM_JOB_ID"
echo "Node:    $SLURMD_NODENAME"
echo "GPU:     ${CUDA_VISIBLE_DEVICES:-none}"
echo "Python:  $(python --version)"
echo "Repo:    $REPO_ROOT"

python scripts/run_experiment.py --config config/llama70b_mmlu_redo_abcd.yaml --yes

echo "llama70b abcd MMLU complete."
