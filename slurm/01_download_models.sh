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
#
# Models are downloaded into HF_HUB_CACHE on scratch. Configs reference models
# by their HuggingFace hub ID (e.g. Qwen/Qwen2.5-7B-Instruct) so HF_HOME must
# be set consistently in every SLURM job that loads them.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

SCRATCH="/mnt/scratch2/users/$USER"
VENV_DIR="$SCRATCH/venvs/mcq-generalization"

export HF_HOME="$SCRATCH/hf"
export HF_HUB_CACHE="$HF_HOME/hub"

mkdir -p "$HF_HOME"

module load apps/python3/3.12.4/gcc-14.1.0

source "$VENV_DIR/bin/activate"

echo "HF cache: $HF_HOME"

# --- Tiny model first — verify everything works before downloading large weights ---

echo "--- Qwen/Qwen2.5-0.5B-Instruct (~1GB, smoke-test model) ---"
huggingface-cli download Qwen/Qwen2.5-0.5B-Instruct

# --- Uncomment models below once the tiny download and smoke test succeed ---

# echo "--- Qwen/Qwen2.5-7B-Instruct (~15GB) ---"
# huggingface-cli download Qwen/Qwen2.5-7B-Instruct

# echo "--- meta-llama/Llama-3.1-8B-Instruct (~16GB) ---"
# huggingface-cli download meta-llama/Llama-3.1-8B-Instruct

echo "Download complete."
echo "HF cache size: $(du -sh "$HF_HOME" | cut -f1)"
