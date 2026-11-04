#!/bin/bash
#SBATCH --job-name=meal_detection_faithful
#SBATCH --partition=normal
#SBATCH --qos=high_p
#SBATCH --time=12:00:00
#SBATCH --mem=32GB
#SBATCH --cpus-per-task=4
#SBATCH -G 1
#SBATCH --output=logs/%x_%j.out

# SOURCE-PROTOCOL REPRODUCTION ARM -- DIAGNOSTIC ONLY.
#
# Trains the meal detector at SSM-CGM's own published Training Setup
# (arXiv:2510.04386 Appendix A.2): "Sampling 5-minute CGM; window length
# L=72; stride s=1; CNN channels cout=16; 2 BiLSTM hidden h=64 per direction;
# dropout 0.4; LayerNorm; Adam (lr 10-4, weight decay 10-4); OneCycleLR;
# batch size 16; 10 epochs; positive class weight alpha=5".
#
# WHY: of the four ways the reported detector deviates from that setup
# (stride 6 vs 1, batch 64 vs 16, <=40 epochs w/ early stopping vs a fixed
# 10, lr 3e-4 vs 1e-4), only the learning rate was ever measured -- and that
# measurement is confounded, because stride 6 + batch 64 yields ~24x fewer
# optimizer steps per epoch and a higher lr is the expected compensation for
# a shorter budget. So "our precision is 55.6% vs their reported 72.1%" may
# be a training-budget artifact rather than the unpublished-split effect it
# has been attributed to. This arm measures that.
#
# DIAGNOSTIC ONLY -- writes to its OWN directory. It does NOT overwrite
# model_input/meal_detection/{checkpoint.pt,metrics.json,event_level_pr_curve.csv}
# and does NOT regenerate predmeal_prob/. The reported covariate channel is
# unchanged; every downstream arm that consumes it stays valid. Promoting
# this checkpoint would mean re-running every predmeal_prob consumer, which
# is a separate decision, not something this job takes.
#
# ONE KNOWN RESIDUAL DEVIATION, and it favours the faithful arm: train.py
# checkpoints the best epoch by val AUPRC, whereas the source trains a fixed
# 10 epochs and takes the final model. Early stopping is disabled here
# (patience 99 > 10 epochs) so all 10 epochs run, but the saved checkpoint is
# best-of-10, not final-of-10. Since best-of-10 >= final-of-10 on val AUPRC,
# an underperforming result here cannot be blamed on that.
set -euo pipefail
REPO_ROOT=${AIREADI_ROOT:?set AIREADI_ROOT to your aireadi-collab checkout}
# Same module set as slurm/meal_detection_train.sbatch -- see its comment for
# why py-jax must NOT be loaded (cuDNN conflict kills nn.LSTM).
ml load math py-pytorch/2.9.1_py314 py-huggingface-hub/0.36.0_py314 py-safetensors/0.7.0_py314 py-numpy/2.3.5_py314 py-pandas/2.3.3_py314 py-pyarrow/22.0.0_py314
ml load math py-scikit-learn/1.8.0_py314
source "$REPO_ROOT/venv-timesfm/bin/activate"

# Our OWN copy (slurm/cgmacros_download.sbatch). The path the original run
# used, /path/to/work/..., is mode 0700 and unreadable here.
# Auto-detect the extracted root: the zip's top-level folder name is not
# guaranteed, so locate the directory that actually contains bio.csv.
CGM_ROOT="$AIREADI_SCRATCH/cgmacros/1.0.0/extracted"
BIO=$(find "$CGM_ROOT" -maxdepth 3 -name bio.csv | head -1)
test -n "$BIO" || { echo "bio.csv not found under $CGM_ROOT -- did cgmacros_download.sbatch run?" >&2; exit 1; }
export CGMACROS_DIR="$(dirname "$BIO")"
export CGMACROS_BIO_CSV="$BIO"
echo "CGMACROS_DIR=$CGMACROS_DIR"

OUT="$REPO_ROOT/model_input/meal_detection/faithful_arm"
mkdir -p "$OUT"
export MEAL_DETECTION_OUTPUT_DIR="$OUT"

# The published Training Setup, verbatim.
export MEAL_DETECTION_STRIDE=1
export MEAL_DETECTION_BATCH_SIZE=16
export MEAL_DETECTION_EPOCHS=10
export MEAL_DETECTION_LR=1e-4
export MEAL_DETECTION_C_OUT=16
export MEAL_DETECTION_HIDDEN=64
export MEAL_DETECTION_DROPOUT=0.4
export MEAL_DETECTION_PATIENCE=99   # disable early stopping; run all 10 epochs

echo "=== 1/3 train (source protocol) ==="
python "$REPO_ROOT/code/meal_detection/train.py"

echo "=== 2/3 event-level eval at the paper's operating point (theta=0.36, rho=0.77) ==="
python "$REPO_ROOT/code/meal_detection/evaluate_event_level.py"

echo "=== 3/3 rho curve at fixed theta=0.36 ==="
python "$REPO_ROOT/code/meal_detection/evaluate_event_level_curve.py"

echo "MEAL_DETECTION_FAITHFUL_ARM_OK"
