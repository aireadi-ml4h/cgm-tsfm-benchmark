#!/bin/bash
#SBATCH --job-name=chronos2_iclcov_test
#SBATCH --partition=gpu
#SBATCH --time=01:00:00
#SBATCH --mem=32GB
#SBATCH --cpus-per-task=4
#SBATCH -G 1
#SBATCH --output=logs/%x_%j.out

set -euo pipefail
REPO_ROOT=${AIREADI_ROOT:?set AIREADI_ROOT to your cgm-tsfm-benchmark checkout}
export HF_HOME=$AIREADI_SCRATCH/hf_cache
mkdir -p "$HF_HOME"

export CHRONOS2_ICLCOV_SPLIT=test
: "${CHRONOS2_ICLCOV_HISTORY_DAYS:?set CHRONOS2_ICLCOV_HISTORY_DAYS to 1, 3, or 7 before submitting}"
export CHRONOS2_ICLCOV_RESULTS_DIR="$REPO_ROOT/chronos2_personalization_results/test/incontext_cov_${CHRONOS2_ICLCOV_HISTORY_DAYS}d"

ml load math py-pytorch/2.9.1_py314 py-transformers/4.57.3_py314 py-einops/0.8.1_py314 py-pandas/2.3.3_py314 py-pyarrow/22.0.0_py314
source "$REPO_ROOT/venv-chronos/bin/activate"

python "$REPO_ROOT/code/chronos2/chronos2_incontext_covariates_personalization.py"
