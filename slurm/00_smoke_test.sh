#!/bin/bash
#SBATCH --job-name=twoprompt_smoke
#SBATCH --output=logs/smoke_%j.out
#SBATCH --error=logs/smoke_%j.err
#SBATCH --time=00:10:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
# No GPU needed — this just verifies the environment and dry-runs the pipeline.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

mkdir -p logs

module load apps/python3/3.12.4/gcc-14.1.0

source .venv/bin/activate

echo "Python: $(python --version)"
echo "Repo:   $REPO_ROOT"

# Verify core imports
python -c "import twoprompt; print('twoprompt import OK')"
python -c "from twoprompt.backends.dummy import DummyBackend; print('DummyBackend import OK')"

# Dry run with dummy backend — no model weights or API calls
python scripts/run_experiment.py --config config/dummy.yaml --dry-run

echo "Smoke test passed."
