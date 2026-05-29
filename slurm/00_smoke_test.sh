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
#SBATCH --job-name=mcqgen_smoke
#SBATCH --output=logs/smoke_%j.out
#SBATCH --error=logs/smoke_%j.err
#SBATCH --time=00:10:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
# No GPU needed — verifies environment and dry-runs the dummy pipeline only.

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

echo "Python: $(python --version)"
echo "Repo:   $REPO_ROOT"
echo "Venv:   $VENV_DIR"

python -c "import modelgen; print('modelgen import OK')"
python -c "from modelgen.backends.dummy import DummyBackend; print('DummyBackend import OK')"

# Preflight estimate — no model calls
python scripts/run_experiment.py --config config/dummy.yaml --dry-run

# Real dummy execution — runs all runners with DummyBackend (5 questions, no GPU needed)
python scripts/run_experiment.py --config config/dummy.yaml --yes

echo "Smoke test passed."
