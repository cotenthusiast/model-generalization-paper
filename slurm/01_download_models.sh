#!/bin/bash
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
#   huggingface-cli login   (run once interactively before this script)
#   Meta licence accepted on huggingface.co (required for Llama models)

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

SCRATCH="/mnt/scratch2/users/$USER"
VENV_DIR="$SCRATCH/venvs/mcq-generalization"

export HF_HOME="$SCRATCH/hf"
export HF_HUB_CACHE="$HF_HOME/hub"
export MODEL_ROOT="$SCRATCH/models"

mkdir -p "$HF_HOME" "$MODEL_ROOT"

module load apps/python3/3.12.4/gcc-14.1.0

source "$VENV_DIR/bin/activate"

echo "Downloading to: $MODEL_ROOT"
echo "HF cache:       $HF_HOME"

# --- Tiny model first — verify everything works before downloading large weights ---

echo "--- Qwen/Qwen2.5-0.5B-Instruct (~1GB, smoke-test model) ---"
huggingface-cli download Qwen/Qwen2.5-0.5B-Instruct \
    --local-dir "$MODEL_ROOT/Qwen2.5-0.5B-Instruct"

# --- Uncomment models below once the tiny download succeeds ---

# echo "--- Qwen/Qwen2.5-7B-Instruct (~15GB) ---"
# huggingface-cli download Qwen/Qwen2.5-7B-Instruct \
#     --local-dir "$MODEL_ROOT/Qwen2.5-7B-Instruct"

# echo "--- meta-llama/Llama-3.1-8B-Instruct (~16GB) ---"
# huggingface-cli download meta-llama/Llama-3.1-8B-Instruct \
#     --local-dir "$MODEL_ROOT/Llama-3.1-8B-Instruct"

echo "Download complete."
echo "Scratch usage: $(du -sh "$SCRATCH" | cut -f1)"
