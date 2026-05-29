#!/usr/bin/env bash
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
# Source this script after SSH-ing into Kelvin2:
#   source scripts/env_kelvin2.sh

export SCRATCH="/mnt/scratch2/users/$USER"
export REPO_DIR="$SCRATCH/repos/model-generalization"
export VENV_DIR="$SCRATCH/venvs/mcq-generalization"
export HF_HOME="$SCRATCH/hf"
export HF_HUB_CACHE="$HF_HOME/hub"
export MODEL_ROOT="$SCRATCH/models"
export RESULTS_DIR="$SCRATCH/results/mcq-generalization"
export PYTHONPATH="$REPO_DIR/src"

mkdir -p "$SCRATCH" "$REPO_DIR" "$VENV_DIR" "$HF_HOME" "$HF_HUB_CACHE" "$MODEL_ROOT" "$RESULTS_DIR"

module load python3/3.10.5/gcc-9.3.0

source "$VENV_DIR/bin/activate"

cd "$REPO_DIR"
