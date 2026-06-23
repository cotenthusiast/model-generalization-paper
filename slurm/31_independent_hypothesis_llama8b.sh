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
# Fresh run of independent_hypothesis for this model, both benchmarks (see
# config/llama8b_mmlu_independent_hypothesis.yaml /
# config/llama8b_arc_independent_hypothesis.yaml for details). Each question
# makes one generate() call per option (3-4, evaluated independently), so
# this is ~4x the generation volume of a single-call method like abcd or
# text_extraction at the same question count — time budget sized
# accordingly. Two sequential run_experiment.py calls so the model loads
# once per call but the whole run is one SLURM submission instead of two.
# Run tiny_llama8b_independent_hypothesis.yaml on its own first to confirm
# the prompt/parsing path before submitting this.
#
# GPU resource note:
# Llama 3.1 8B fits comfortably in an A100 MIG 3g.40gb slice.
# Verify available partitions with:
#   sinfo -o "%P %D %G %m %l %N"
#
#SBATCH --job-name=mcqgen_independent_hypothesis_llama8b
#SBATCH --output=logs/independent_hypothesis_llama8b_%j.out
#SBATCH --error=logs/independent_hypothesis_llama8b_%j.err
#SBATCH --time=08:00:00
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

# module command is currently unavailable in batch job shells on this
# cluster (verified 2026-06-21 via diagnostic job — venv activation alone
# already resolves to the correct interpreter, so this degrades gracefully
# rather than aborting the whole job if the cluster module system is down).
if command -v module >/dev/null 2>&1; then
    module load python3/3.10.5/gcc-9.3.0
fi

source "$VENV_DIR/bin/activate"

echo "Job ID:  $SLURM_JOB_ID"
echo "Node:    $SLURMD_NODENAME"
echo "GPU:     ${CUDA_VISIBLE_DEVICES:-none}"
echo "Python:  $(python --version)"
echo "Repo:    $REPO_ROOT"

python scripts/run_experiment.py --config config/llama8b_mmlu_independent_hypothesis.yaml --yes
python scripts/run_experiment.py --config config/llama8b_arc_independent_hypothesis.yaml --yes

echo "llama8b independent_hypothesis run complete."
