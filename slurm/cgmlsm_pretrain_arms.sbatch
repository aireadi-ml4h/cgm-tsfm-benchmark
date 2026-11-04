#!/bin/bash
# CGM-LSM GlucoFM-Bench pretraining: full corpus vs. T1D-excluded.
#
# Both arms run code/cgmlsm/cgmlsm_pretrain_glucofmbench.py with identical resources.
# Consolidated from two near-identical scripts.
#
#   full     20 epochs over the full GlucoFM-Bench corpus (script defaults)
#   no_t1d   the same, on the T1D-excluded corpus -- this is the pretrain that
#            feeds the fine-tuned CGM-LSM reported in the paper
#
# 20 epochs over 11,074 train chunks, with three eval sets
# (held_out / temporal / internal) reported per epoch.
#
# Output goes to $CGMLSM_WORK_ROOT (scratch), not this repo.
#
#SBATCH --job-name=cgmlsm_pretrain
#SBATCH --partition=normal
#SBATCH --qos=high_p
#SBATCH --time=01:00:00
#SBATCH --mem=32GB
#SBATCH --cpus-per-task=4
#SBATCH -G 1
#SBATCH --array=0-1
#SBATCH --output=logs/%x_%A_%a.out
set -euo pipefail
REPO_ROOT=${AIREADI_ROOT:?set AIREADI_ROOT to your cgm-tsfm-benchmark checkout}
CGMLSM_WORK_ROOT=${CGMLSM_WORK_ROOT:-$AIREADI_SCRATCH/aireadi_cgmlsm}

# arm | input dir under model_input/ | output dir under $CGMLSM_WORK_ROOT
ARMS=(
  "full|-|-"
  "no_t1d|glucofmbench_no_t1d|pretrain_glucofmbench_no_t1d_v1"
)
IFS='|' read -r ARM_NAME IN_SUBDIR OUT_SUBDIR <<< "${ARMS[${SLURM_ARRAY_TASK_ID:-0}]}"
[ "$IN_SUBDIR" != "-" ] && export GLUCOFMBENCH_INPUT_DIR_OVERRIDE="$REPO_ROOT/model_input/$IN_SUBDIR"
[ "$OUT_SUBDIR" != "-" ] && export CGMLSM_PRETRAIN_OUTPUT_DIR_OVERRIDE="$CGMLSM_WORK_ROOT/$OUT_SUBDIR"
echo "arm=$ARM_NAME in=$IN_SUBDIR out=$OUT_SUBDIR"

ml load math py-pytorch/2.4.1_py312
source "$REPO_ROOT/venv-cgmlsm/bin/activate"
python "$REPO_ROOT/code/cgmlsm/cgmlsm_pretrain_glucofmbench.py"
