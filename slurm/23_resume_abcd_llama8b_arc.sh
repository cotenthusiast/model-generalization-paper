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
# Resume of 19_redo_abcd_llama8b.sh's ARC-Challenge half (job 9209873 hit
# its 2h time limit at 500/1000 questions). Split out of the original
# two-benchmark job per the 2026-06-21 abcd timeout post-mortem: ARC and
# MMLU now run as separate SLURM jobs so a stall on one never strands
# completed work on the other (see git log for the finish_reason and
# hardcoded-OPTION_LETTERS fixes from the same investigation).
#
# --run-id reuses the exact checkpoint at
# checkpoints/20260621_011720/abcd__meta-llama_Llama-3.1-8B-Instruct__arc_challenge.json
# (500/1000 done) so this resumes rather than restarts.
#
# Also bundles the PriDe ARC-Challenge calibration-prior recompute for this
# model (~200 cheap score_options calls, no generation) onto this job since
# it already loads the model for ARC-Challenge work — see
# scripts/recompute_pride_calibration.py and the pride.py/pride_debias.py
# fix for the hardcoded-OPTION_LETTERS bug.
#
# Observed rate from the original attempt: ~11.4 min/50 questions. Remaining
# 500 questions ≈ 1.9h; budgeted with generous margin.
#
#SBATCH --job-name=mcqgen_resume_abcd_llama8b_arc
#SBATCH --output=logs/mcqgen_resume_abcd_llama8b_arc_%j.out
#SBATCH --error=logs/mcqgen_resume_abcd_llama8b_arc_%j.err
#SBATCH --time=03:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --partition=k2-gpu-a100mig
#SBATCH --gres=gpu:3g.40gb:1

set -euo pipefail

# BASH_SOURCE/dirname resolves incorrectly in Kelvin2's SLURM execution environment.
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

python scripts/run_experiment.py --config config/llama8b_arc_challenge_redo_abcd.yaml --run-id 20260621_011720 --yes

python scripts/recompute_pride_calibration.py \
  --config config/llama8b_arc_expensive.yaml \
  --model-name meta-llama/Llama-3.1-8B-Instruct \
  --apply

echo "llama8b abcd ARC-Challenge resume + PriDe calibration recompute complete."
