#!/bin/bash
#SBATCH --job-name=twoprompt_full
#SBATCH --output=logs/full_%j.out
#SBATCH --error=logs/full_%j.err
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=80G
#SBATCH --gres=gpu:1
# TODO: set --partition to the correct GPU partition name on Kelvin2.
# Run `sinfo` on the login node to see available partitions.
# e.g. #SBATCH --partition=k2-gpu
#
# For larger models (32B, 70B, 72B) you will need multiple GPUs.
# Uncomment and adjust:
# #SBATCH --gres=gpu:4

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

mkdir -p logs

module load apps/python3/3.12.4/gcc-14.1.0

source .venv/bin/activate

export HF_HOME="/mnt/scratch2/users/$USER/hf"
export HF_HUB_CACHE="/mnt/scratch2/users/$USER/hf/hub"

echo "Job ID:  $SLURM_JOB_ID"
echo "Node:    $SLURMD_NODENAME"
echo "GPU:     $CUDA_VISIBLE_DEVICES"
echo "Python:  $(python --version)"
echo "Repo:    $REPO_ROOT"

python scripts/run_experiment.py --config config/default.yaml --yes

echo "Full run complete."
