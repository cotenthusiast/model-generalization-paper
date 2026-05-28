#!/bin/bash
#SBATCH --job-name=twoprompt_full
#SBATCH --output=logs/full_%j.out
#SBATCH --error=logs/full_%j.err
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=80G
#SBATCH --partition=k2-gpu-a100
#SBATCH --gres=gpu:a100:1
# TODO: verify partition and GRES with `sinfo` on the Kelvin2 login node before submitting.
#
# H100 alternative (higher memory, useful for 32B/70B/72B models):
# #SBATCH --partition=k2-gpu-h100
# #SBATCH --gres=gpu:h100:1
#
# For 70B/72B models, multiple GPUs will be needed. Adjust --gres accordingly,
# e.g. --gres=gpu:a100:4, and ensure device: auto is set in config.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

mkdir -p logs

SCRATCH="/mnt/scratch2/users/$USER"
VENV_DIR="$SCRATCH/venvs/mcq-generalization"

export HF_HOME="$SCRATCH/hf"
export HF_HUB_CACHE="$HF_HOME/hub"
export MODEL_ROOT="$SCRATCH/models"
export RESULTS_DIR="$SCRATCH/results/mcq-generalization"

module load apps/python3/3.12.4/gcc-14.1.0

source "$VENV_DIR/bin/activate"

echo "Job ID:  $SLURM_JOB_ID"
echo "Node:    $SLURMD_NODENAME"
echo "GPU:     ${CUDA_VISIBLE_DEVICES:-none}"
echo "Python:  $(python --version)"
echo "Repo:    $REPO_ROOT"

python scripts/run_experiment.py --config config/default.yaml --yes

echo "Full run complete."
