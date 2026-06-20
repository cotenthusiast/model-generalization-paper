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
# Full redo of additional_option and cyclic for this model, both
# benchmarks, using the score_options()-based redesign of both methods
# (see config/llama70b_arc_challenge_redo_aoi_cyclic.yaml /
# config/llama70b_mmlu_redo_aoi_cyclic.yaml for details and citations).
# Two sequential run_experiment.py calls so the model loads once per
# call but the whole redo is one SLURM submission instead of two.
#
# Old majority-vote cyclic run data is archived under
# runs_archive/cyclic_majority_vote_pre_eq1_redesign_20260621/ and is not
# touched by this job.
#
# GPU resource note:
# Llama 3.1 70B needs 2x Kelvin2 A100s (160GB total); device=auto in the config handles multi-GPU splitting via accelerate automatically.
# Verify available partitions with:
#   sinfo -o "%P %D %G %m %l %N"
#
#SBATCH --job-name=mcqgen_redo_llama70b
#SBATCH --output=logs/mcqgen_redo_llama70b_%j.out
#SBATCH --error=logs/mcqgen_redo_llama70b_%j.err
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
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

python scripts/run_experiment.py --config config/llama70b_arc_challenge_redo_aoi_cyclic.yaml --yes
python scripts/run_experiment.py --config config/llama70b_mmlu_redo_aoi_cyclic.yaml --yes

echo "llama70b additional_option/cyclic redo complete."
