#!/bin/bash
#SBATCH --job-name=cgmlsm_newbase_test
#SBATCH --partition=normal
#SBATCH --qos=high_p
#SBATCH --time=02:00:00
#SBATCH --mem=32GB
#SBATCH --cpus-per-task=4
#SBATCH -G 1
#SBATCH --output=logs/%x_%j.out

# Scores the NEW frozen base (2 epochs, lr=1e-2, from the T1D-excluded
# GlucoFM-Bench pretrain) on the TEST windows -- the reportable number, to sit
# alongside the old settled checkpoint's 13.520/28.214/49.291.
#
# INDEPENDENT OF THE PERSONALIZATION WORK. The recipe was selected on val
# windows and is now bracketed on every axis (lr=2e-2 destroys the model,
# lr=5e-3 is worse, 1 epoch is worse, 3-4 epochs are worse), so nothing about
# personalization can change it. This should have been running in parallel.
#
# ALL FOUR SEEDS, not just seed 42. Seed 42 was hardcoded until the stability
# check, and it turned out to be the LUCKIEST of four (val 2h: 24.0 vs 25.8,
# 29.4, 30.1). Reporting a single seed's test number would carry that luck into
# the headline figure. The honest report is the mean and range across seeds,
# with seed 42 identified as the optimistic end.
set -euo pipefail
REPO_ROOT=${AIREADI_ROOT:?set AIREADI_ROOT to your cgm-tsfm-benchmark checkout}
SCRATCH_BASE="$AIREADI_SCRATCH/aireadi_cgmlsm"
RESULTS_ROOT="$REPO_ROOT/cgmlsm_results/newbase_test"

ml load math py-pytorch/2.4.1_py312
source "$REPO_ROOT/venv-cgmlsm/bin/activate"
mkdir -p "$RESULTS_ROOT"

declare -A CKPTS=(
  [seed42]="$SCRATCH_BASE/finetune_sweep4_ep2_lr1e-2/final"
  [seed7]="$SCRATCH_BASE/finetune_seedcheck_ep2_lr1e-2_seed7/final"
  [seed123]="$SCRATCH_BASE/finetune_seedcheck_ep2_lr1e-2_seed123/final"
  [seed2024]="$SCRATCH_BASE/finetune_seedcheck_ep2_lr1e-2_seed2024/final"
)

for TAG in seed42 seed7 seed123 seed2024; do
  CKPT="${CKPTS[$TAG]}"
  OUT="$RESULTS_ROOT/ep2_lr1e-2_$TAG"
  test -d "$CKPT" || { echo "missing checkpoint $CKPT" >&2; exit 1; }
  if [ -f "$OUT/summary.csv" ]; then echo "===== skip $TAG ====="; continue; fi
  echo "===== $TAG on TEST windows ====="
  CGMLSM_CHECKPOINT_DIR="$CKPT" CGMLSM_EVAL_OUTPUT_DIR="$OUT" \
    python "$REPO_ROOT/code/cgmlsm/cgmlsm_eval.py"
done

echo "=== new base on TEST windows, per seed ==="
for d in "$RESULTS_ROOT"/*/; do
  echo "--- $(basename "$d")"; cut -d, -f1,2,3,4,5 "$d/summary.csv" 2>/dev/null
done
echo "CGMLSM_NEWBASE_TEST_OK"
