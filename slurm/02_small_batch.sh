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
# This script uses Kelvin2 A100 MIG 3g.40gb for small/medium local-model runs such as Qwen 7B.
# Verify available partitions with:
#   sinfo -o "%P %D %G %m %l %N"
#
#SBATCH --job-name=mcqgen_small
#SBATCH --output=logs/small_%j.out
#SBATCH --error=logs/small_%j.err
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --partition=k2-gpu-a100mig
#SBATCH --gres=gpu:3g.40gb:1
# Full A100 alternative (for larger models):
# #SBATCH --partition=k2-gpu-a100
# #SBATCH --gres=gpu:a100:1
# H100 alternative (higher memory):
# #SBATCH --partition=k2-gpu-h100
# #SBATCH --gres=gpu:h100:1

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

python scripts/run_experiment.py --config config/small_batch.yaml --yes

echo "Small batch run complete."
