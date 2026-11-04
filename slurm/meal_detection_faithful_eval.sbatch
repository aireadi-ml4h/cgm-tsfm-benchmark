#!/bin/bash
#SBATCH --job-name=meal_detection_faithful_eval
#SBATCH --partition=normal
#SBATCH --qos=high_p
#SBATCH --time=02:00:00
#SBATCH --mem=32GB
#SBATCH --cpus-per-task=4
#SBATCH -G 1
#SBATCH --output=logs/%x_%j.out

# Re-runs ONLY the event-level evaluation of the source-protocol arm trained
# by slurm/meal_detection_faithful_arm.sbatch (job 42433651). That job's
# training was correct, but its eval scored the DEFAULT checkpoint, because
# generate_meal_probability.load_model() hardcoded CHECKPOINT_PATH -- fixed
# 2026-09-07 to honour MEAL_DETECTION_OUTPUT_DIR. Training is not repeated;
# faithful_arm/checkpoint.pt is reused as-is.
set -euo pipefail
REPO_ROOT=${AIREADI_ROOT:?set AIREADI_ROOT to your cgm-tsfm-benchmark checkout}
ml load math py-pytorch/2.9.1_py314 py-huggingface-hub/0.36.0_py314 py-safetensors/0.7.0_py314 py-numpy/2.3.5_py314 py-pandas/2.3.3_py314 py-pyarrow/22.0.0_py314
ml load math py-scikit-learn/1.8.0_py314
source "$REPO_ROOT/venv-timesfm/bin/activate"

BIO=$(find "$AIREADI_SCRATCH/cgmacros/1.0.0/extracted" -maxdepth 3 -name bio.csv | head -1)
export CGMACROS_DIR="$(dirname "$BIO")"
export CGMACROS_BIO_CSV="$BIO"
export MEAL_DETECTION_OUTPUT_DIR="$REPO_ROOT/model_input/meal_detection/faithful_arm"
export MEAL_DETECTION_STRIDE=1   # affects dataset build only; eval stride is fixed at 1

python - <<'PY'
import os, torch
p = os.environ["MEAL_DETECTION_OUTPUT_DIR"] + "/checkpoint.pt"
d = os.path.dirname(p)
import sys; sys.path.insert(0, "${AIREADI_ROOT}/code/meal_detection")
from generate_meal_probability import CHECKPOINT_PATH
assert CHECKPOINT_PATH == p, f"WIRING CHECK FAILED: load_model would read {CHECKPOINT_PATH}, not {p}"
print("wiring check OK ->", CHECKPOINT_PATH)
PY

echo "=== event-level eval at the paper's operating point (theta=0.36, rho=0.77) ==="
python "$REPO_ROOT/code/meal_detection/evaluate_event_level.py"
echo "=== rho curve at fixed theta=0.36 ==="
python "$REPO_ROOT/code/meal_detection/evaluate_event_level_curve.py"
echo "MEAL_DETECTION_FAITHFUL_EVAL_OK"
