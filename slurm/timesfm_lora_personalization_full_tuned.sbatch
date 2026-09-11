#!/bin/bash
#SBATCH --job-name=timesfm_lora_pers_tuned
#SBATCH --partition=normal
#SBATCH --qos=high_p
#SBATCH --time=12:00:00
#SBATCH --mem=32GB
#SBATCH --cpus-per-task=4
#SBATCH -G 1
#SBATCH --output=/scratch/users/%u/logs/%x_%j.out

set -euo pipefail
REPO_ROOT=${AIREADI_ROOT:?set AIREADI_ROOT to your aireadi-collab checkout}
export HF_HOME=$SCRATCH/hf_cache
mkdir -p "$HF_HOME"

ml load math py-pytorch/2.9.1_py314 py-numpy/2.3.5_py314 py-pandas/2.3.3_py314 py-pyarrow/22.0.0_py314
source "$REPO_ROOT/venv-timesfm/bin/activate"

: "${TIMESFM_LORA_SPLIT:?set TIMESFM_LORA_SPLIT to val or test}"
export TIMESFM_LORA_SPLIT

# TUNED CONFIG, 2026-09-09: real bracket sweep (8 configs x k=1d/7d, 50
# participants, cluster-by-participant t-test vs untuned control lr=1e-4/
# steps=100) found NUM_STEPS=25 (lr unchanged at 1e-4) significantly BETTER
# at k=1d (mean_diff=2.66 mg/dL, p=0.0002, best of 6 candidates tested) and
# statistically TIED with control at k=7d (p=0.08, not worse) -- and ~4x
# faster per fit (3.4s vs 13.3s/participant). Writes to a NEW "_tuned"
# suffixed directory; old untuned lora_{1,3,7}d results are UNTOUCHED.
export TIMESFM_LORA_NUM_STEPS=25

for days in 1 3 7; do
  echo "=== history_days=$days split=$TIMESFM_LORA_SPLIT (TUNED: steps=25) ===" >&2
  export TIMESFM_LORA_HISTORY_DAYS=$days
  export TIMESFM_LORA_RESULTS_DIR="$REPO_ROOT/timesfm_personalization_results/$TIMESFM_LORA_SPLIT/lora_${days}d_tuned"
  python "$REPO_ROOT/code/timesfm/timesfm_lora_personalization.py"
done
