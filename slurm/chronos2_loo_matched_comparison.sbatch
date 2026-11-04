#!/bin/bash
#SBATCH --job-name=chronos2_loo_matched_comparison
#SBATCH --partition=normal
#SBATCH --qos=high_p
#SBATCH --time=00:20:00
#SBATCH --mem=8GB
#SBATCH --cpus-per-task=1
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail
REPO_ROOT=${AIREADI_ROOT:?set AIREADI_ROOT to your aireadi-collab checkout}
ml load math py-pytorch/2.9.1_py314 py-transformers/4.57.3_py314 py-einops/0.8.1_py314 py-pandas/2.3.3_py314 py-pyarrow/22.0.0_py314
source "$REPO_ROOT/venv-chronos/bin/activate"

python "$REPO_ROOT/code/analysis/chronos2_loo_matched_comparison.py"
