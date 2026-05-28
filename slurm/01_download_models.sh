#!/bin/bash
# Download HuggingFace model weights to scratch storage.
#
# Run this DIRECTLY on the login node (not via sbatch) since compute nodes
# may not have outbound internet access:
#
#   bash slurm/01_download_models.sh
#
# Requires: huggingface-cli login (run once interactively before this script)
# Models are saved to HF_HOME so all subsequent SLURM jobs find them automatically.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

module load apps/python3/3.12.4/gcc-14.1.0

source .venv/bin/activate

export HF_HOME="/mnt/scratch2/users/$USER/hf"
export HF_HUB_CACHE="/mnt/scratch2/users/$USER/hf/hub"

mkdir -p "$HF_HOME"

echo "Downloading to: $HF_HOME"

# Qwen 7B (~15GB)
echo "--- Qwen/Qwen2.5-7B-Instruct ---"
huggingface-cli download Qwen/Qwen2.5-7B-Instruct

# Llama 3.1 8B (~16GB) — requires accepted Meta licence on huggingface.co
echo "--- meta-llama/Llama-3.1-8B-Instruct ---"
huggingface-cli download meta-llama/Llama-3.1-8B-Instruct

echo "All models downloaded."
echo "Total cache size: $(du -sh "$HF_HOME" | cut -f1)"
