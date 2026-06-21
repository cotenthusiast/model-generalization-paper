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
# Resume of 20_redo_abcd_qwen32b.sh's MMLU half (job 9209874 hit its 4h time
# limit at 150/1000 questions). This model's ARC-Challenge half already
# completed in full (1000/1000, run 20260621_012827) so there is no
# corresponding ARC resume job — only MMLU needs resuming. Split out per
# the 2026-06-21 abcd timeout post-mortem.
#
# --run-id reuses the exact checkpoint at
# checkpoints/20260621_044033/abcd__Qwen_Qwen2.5-32B-Instruct__mmlu.json
# (150/1000 done) so this resumes rather than restarts.
#
# Also bundles the PriDe ARC-Challenge calibration-prior recompute for this
# model (~200 cheap score_options calls, no generation) onto this job,
# reusing the model load even though this job's main work is MMLU — see
# scripts/recompute_pride_calibration.py and the pride.py/pride_debias.py
# fix for the hardcoded-OPTION_LETTERS bug.
#
# Observed rate from the original attempt: ~6.77 min/50 questions on MMLU.
# Remaining 850 questions ≈ 1.9h; budgeted with generous margin.
#
#SBATCH --job-name=mcqgen_resume_abcd_qwen32b_mmlu
#SBATCH --output=logs/mcqgen_resume_abcd_qwen32b_mmlu_%j.out
#SBATCH --error=logs/mcqgen_resume_abcd_qwen32b_mmlu_%j.err
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --partition=k2-gpu-a100
#SBATCH --gres=gpu:a100:1

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

python scripts/run_experiment.py --config config/qwen32b_mmlu_redo_abcd.yaml --run-id 20260621_044033 --yes

python scripts/recompute_pride_calibration.py \
  --config config/qwen32b_arc_expensive.yaml \
  --model-name Qwen/Qwen2.5-32B-Instruct \
  --apply

echo "qwen32b abcd MMLU resume + PriDe calibration recompute complete."
