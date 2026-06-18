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
# Patches the prompt_builder "phantom D option" bug for Qwen 72B's
# ARC-Challenge data (see prompt_builder fix + scripts/prepare_patch_checkpoints.py
# for background). All 4 combos are already complete (1000/1000) — each call
# regenerates only the 3 contaminated questions, 997 untouched rows carried over:
#   1. run_id 20260602_010405 (baseline/calibration/additional_option)
#   2. run_id 20260602_125909 (cyclic/pride/two_prompt)
#   3. run_id 20260608_162341 (text_extraction)
#   4. run_id 20260608_180141 (abcd)
#
# PREREQUISITE (run once on the login node before submitting any of the 5
# per-model patch jobs — cheap, CPU-only, no GPU/SLURM needed):
#   python scripts/prepare_patch_checkpoints.py --apply
#
# GPU resource note:
# This script uses 2x Kelvin2 A100s (2x80GB = 160GB total), matching the
# original qwen72b_arc_*.yaml runs. device=auto in the config handles
# multi-GPU splitting via accelerate automatically.
# Verify available partitions with:
#   sinfo -o "%P %D %G %m %l %N"
#
#SBATCH --job-name=mcqgen_patch_qwen72b
#SBATCH --output=logs/patch_qwen72b_%j.out
#SBATCH --error=logs/patch_qwen72b_%j.err
#SBATCH --time=03:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=160G
#SBATCH --partition=k2-gpu-a100
#SBATCH --gres=gpu:a100:2

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

python scripts/run_experiment.py --run-id 20260602_010405 --config config/qwen72b_arc_core.yaml --yes
python scripts/run_experiment.py --run-id 20260602_125909 --config config/qwen72b_arc_expensive.yaml --yes
python scripts/run_experiment.py --run-id 20260608_162341 --config config/qwen72b_arc_text_extraction.yaml --yes
python scripts/run_experiment.py --run-id 20260608_180141 --config config/qwen72b_arc_abcd.yaml --yes

echo "Qwen 72B ARC-Challenge patch complete."
