#!/bin/bash
#SBATCH --job-name=twoprompt_small
#SBATCH --output=logs/small_%j.out
#SBATCH --error=logs/small_%j.err
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --partition=k2-gpu-a100mig
#SBATCH --gres=gpu:2g.20gb:1
# TODO: verify partition and GRES with `sinfo` on the Kelvin2 login node before submitting.

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

python scripts/run_experiment.py --config config/small_batch.yaml --yes

echo "Small batch run complete."
