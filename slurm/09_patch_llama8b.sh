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
# Patches the prompt_builder "phantom D option" bug for Llama 8B's
# ARC-Challenge data (see prompt_builder fix + scripts/prepare_patch_checkpoints.py
# for background). Three sequential run_experiment.py calls, one per affected
# run_id, so the model loads once per call but the whole patch is one SLURM
# submission instead of one per method:
#   1. run_id 20260529_180747 (baseline/calibration/additional_option):
#      regenerates the 3 contaminated questions, 997 untouched rows carried over.
#   2. run_id 20260529_180748 (cyclic/pride/two_prompt): same, 3 questions only.
#   3. run_id 20260617_162624 (abcd/text_extraction): this combo never finished
#      (850/1000) — regenerates all 150 missing questions fresh, which
#      naturally includes the 3 contaminated ones generated correctly this time.
#
# PREREQUISITE (run once on the login node before submitting any of the 5
# per-model patch jobs — cheap, CPU-only, no GPU/SLURM needed):
#   python scripts/prepare_patch_checkpoints.py --apply
#
# GPU resource note:
# Llama 8B fits comfortably in an A100 MIG 3g.40gb slice.
# Verify available partitions with:
#   sinfo -o "%P %D %G %m %l %N"
#
#SBATCH --job-name=mcqgen_patch_llama8b
#SBATCH --output=logs/patch_llama8b_%j.out
#SBATCH --error=logs/patch_llama8b_%j.err
#SBATCH --time=02:00:00
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

python scripts/run_experiment.py --run-id 20260529_180747 --config config/llama8b_arc_core.yaml --yes
python scripts/run_experiment.py --run-id 20260529_180748 --config config/llama8b_arc_expensive.yaml --yes
python scripts/run_experiment.py --run-id 20260617_162624 --config config/rerun_arc_abcd_text_extraction_llama8b.yaml --yes

echo "Llama 8B ARC-Challenge patch complete."
