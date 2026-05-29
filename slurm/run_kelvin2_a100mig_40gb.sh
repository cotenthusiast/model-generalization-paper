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
# Usage:
#   CONFIG=config/small_batch.yaml sbatch slurm/run_kelvin2_a100mig_40gb.sh
#
#SBATCH --job-name=mcqgen_a100mig
#SBATCH --output=logs/a100mig_%j.out
#SBATCH --error=logs/a100mig_%j.err
#SBATCH --time=12:00:00
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

if [[ -z "${CONFIG:-}" ]]; then
    echo "ERROR: CONFIG environment variable is not set." >&2
    echo "Usage: CONFIG=config/small_batch.yaml sbatch slurm/run_kelvin2_a100mig_40gb.sh" >&2
    exit 1
fi

echo "Job ID:  $SLURM_JOB_ID"
echo "Node:    $SLURMD_NODENAME"
echo "GPU:     ${CUDA_VISIBLE_DEVICES:-none}"
echo "Python:  $(python --version)"
echo "Repo:    $REPO_ROOT"
echo "Config:  $CONFIG"

python scripts/run_experiment.py --config "$CONFIG" --yes

echo "Run complete."
