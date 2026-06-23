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
# GPU resource note:
# Llama 3.1 70B needs 2x Kelvin2 A100s (160GB total); device=auto in the config handles multi-GPU splitting via accelerate automatically.
#
# Resume of 22_redo_abcd_llama70b.sh's ARC-Challenge half (job 9209877 hit
# its 8h time limit at 950/1000 questions -- it was NOT hung; it made
# steady progress throughout, just needed slightly more than 8h to finish
# 1000 questions at ~23.9 min/50q). Split out per the 2026-06-21 abcd
# timeout post-mortem so a near-complete run is never stranded again.
#
# --run-id reuses the exact checkpoint at
# checkpoints/20260621_014921/abcd__meta-llama_Llama-3.1-70B-Instruct__arc_challenge.json
# (950/1000 done) so this resumes rather than restarts.
#
# Also bundles the PriDe ARC-Challenge calibration-prior recompute for this
# model (~200 cheap score_options calls, no generation) onto this job since
# it already loads the model for ARC-Challenge work — see
# scripts/recompute_pride_calibration.py and the pride.py/pride_debias.py
# fix for the hardcoded-OPTION_LETTERS bug.
#
# Remaining 50 questions at ~23.9 min/50q ≈ 24 min; budgeted with large
# margin since model load alone takes a noticeable share of a short job.
#
# Bumped 02:00:00 -> 04:00:00 after Qwen-32B's resume job (9211248) finished
# its abcd work fine but only cleared the bundled PriDe recompute's model
# reload with ~6 minutes of margin left on a 4h budget: the per-question
# rate measured from the original truncated run undershot the resumed
# run's actual rate by ~2x, and the recompute step's own model reload
# (separate process, several minutes for a 32B model) was never budgeted
# at all. Both risks are worse for a 70B model on 2 GPUs, so doubling the
# budget here rather than re-deriving a tighter number from numbers that
# already proved unreliable once.
#
#SBATCH --job-name=mcqgen_resume_abcd_llama70b_arc
#SBATCH --output=logs/mcqgen_resume_abcd_llama70b_arc_%j.out
#SBATCH --error=logs/mcqgen_resume_abcd_llama70b_arc_%j.err
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=160G
#SBATCH --partition=k2-gpu-a100
#SBATCH --gres=gpu:a100:2

set -euo pipefail

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

python scripts/run_experiment.py --config config/llama70b_arc_challenge_redo_abcd.yaml --run-id 20260621_014921 --yes

python scripts/recompute_pride_calibration.py \
  --config config/llama70b_arc_expensive.yaml \
  --model-name meta-llama/Llama-3.1-70B-Instruct \
  --apply

echo "llama70b abcd ARC-Challenge resume + PriDe calibration recompute complete."
