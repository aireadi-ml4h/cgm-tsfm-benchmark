#!/bin/bash
#SBATCH --job-name=timesfm_icl_pers_test
#SBATCH --partition=normal
#SBATCH --qos=high_p
#SBATCH --time=02:00:00
#SBATCH --mem=32GB
#SBATCH --cpus-per-task=4
#SBATCH -G 1
#SBATCH --output=logs/%x_%j.out

set -euo pipefail
REPO_ROOT=${AIREADI_ROOT:?set AIREADI_ROOT to your cgm-tsfm-benchmark checkout}
export HF_HOME=$AIREADI_SCRATCH/hf_cache
mkdir -p "$HF_HOME"

export TIMESFM_ICL_SPLIT=test
: "${TIMESFM_ICL_HISTORY_DAYS:?set TIMESFM_ICL_HISTORY_DAYS to 1, 3, or 7 before submitting}"
export TIMESFM_ICL_RESULTS_DIR="$REPO_ROOT/timesfm_personalization_results/test/incontext_${TIMESFM_ICL_HISTORY_DAYS}d"

ml load math py-pytorch/2.9.1_py314 py-huggingface-hub/0.36.0_py314 py-safetensors/0.7.0_py314 py-numpy/2.3.5_py314 py-pandas/2.3.3_py314 py-pyarrow/22.0.0_py314
source "$REPO_ROOT/venv-timesfm/bin/activate"

python "$REPO_ROOT/code/timesfm/timesfm_incontext_personalization.py"
