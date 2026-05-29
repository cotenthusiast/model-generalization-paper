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
# Download HuggingFace model weights to scratch storage.
#
# Run only after verifying Kelvin2 download policy. Prefer an interactive
# compute job or data-transfer node if available. Do not repeatedly download
# large models on the login node.
#
# Usage:
#   bash slurm/01_download_models.sh
#
# Prerequisites:
#   hf auth login   (run once interactively before this script)
#   Meta licence accepted on huggingface.co (required for Llama models)
#
# Models are downloaded into HF_HUB_CACHE on scratch. Configs reference models
# by their HuggingFace hub ID (e.g. Qwen/Qwen2.5-7B-Instruct) so HF_HOME must
# be set consistently in every SLURM job that loads them.

set -euo pipefail

# BASH_SOURCE/dirname resolves incorrectly in Kelvin2's SLURM execution environment.
REPO_ROOT="/mnt/scratch2/users/$USER/repos/model-generalization"
cd "$REPO_ROOT"

SCRATCH="/mnt/scratch2/users/$USER"
VENV_DIR="$SCRATCH/venvs/mcq-generalization"

export HF_HOME="$SCRATCH/hf"
export HF_HUB_CACHE="$HF_HOME/hub"
export MODEL_ROOT="$SCRATCH/models"
export RESULTS_DIR="$SCRATCH/results/mcq-generalization"

mkdir -p "$HF_HOME"

module load python3/3.10.5/gcc-9.3.0

source "$VENV_DIR/bin/activate"

echo "HF cache: $HF_HOME"

# --- Tiny model first — verify everything works before downloading large weights ---

echo "--- Qwen/Qwen2.5-0.5B-Instruct (~1GB, smoke-test model) ---"
hf download Qwen/Qwen2.5-0.5B-Instruct

# --- Uncomment models below once the tiny download and smoke test succeed ---

# echo "--- Qwen/Qwen2.5-7B-Instruct (~15GB) ---"
# hf download Qwen/Qwen2.5-7B-Instruct

# echo "--- meta-llama/Llama-3.1-8B-Instruct (~16GB) ---"
# hf download meta-llama/Llama-3.1-8B-Instruct

echo "Download complete."
echo "HF cache size: $(du -sh "$HF_HOME" | cut -f1)"
