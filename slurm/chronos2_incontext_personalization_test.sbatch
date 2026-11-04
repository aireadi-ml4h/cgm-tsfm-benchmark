#!/bin/bash
#SBATCH --job-name=chronos2_icl_pers_test
#SBATCH --partition=normal
#SBATCH --qos=high_p
#SBATCH --time=02:00:00
#SBATCH --mem=32GB
#SBATCH --cpus-per-task=4
#SBATCH -G 1
#SBATCH --output=logs/%x_%j.out

# Real full run on the FIXED script (2026-09-07 gap-confound bug fixed --
# see code/CHRONOS2_INCONTEXT_PERSONALIZATION.md). Mirrors
# timesfm_incontext_personalization_test.sbatch's clean per-k-value template,
# per a collaborator's own writeup (external-eval/chronos2_icl_findings.md) for
# invocation/runtime expectations -- those aren't affected by the bug, only
# the script's internal context-building logic was wrong.
set -euo pipefail
REPO_ROOT=${AIREADI_ROOT:?set AIREADI_ROOT to your aireadi-collab checkout}
export HF_HOME=$AIREADI_SCRATCH/hf_cache
mkdir -p "$HF_HOME"

export CHRONOS2_ICL_SPLIT=test
: "${CHRONOS2_ICL_HISTORY_DAYS:?set CHRONOS2_ICL_HISTORY_DAYS to 1, 3, or 7 before submitting}"
export CHRONOS2_ICL_RESULTS_DIR="$REPO_ROOT/chronos2_personalization_results/test/icl_p${CHRONOS2_ICL_HISTORY_DAYS}d"

ml load math py-pytorch/2.9.1_py314 py-transformers/4.57.3_py314 py-einops/0.8.1_py314 py-pandas/2.3.3_py314 py-pyarrow/22.0.0_py314
source "$REPO_ROOT/venv-chronos/bin/activate"

python "$REPO_ROOT/code/chronos2/chronos2_incontext_personalization.py"
