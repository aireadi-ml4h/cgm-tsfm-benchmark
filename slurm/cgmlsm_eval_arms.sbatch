#!/bin/bash
# CGM-LSM evaluation arms.
#
# All five arms run the same entry point (code/cgmlsm/cgmlsm_eval.py) with identical
# resources; they differ only in which trained checkpoint is scored and where
# the predictions go. Consolidated into one array job from five near-identical
# scripts (plus a generic --export runner used by the now-removed sweep).
#
# The two training schemes reported in the paper:
#   from_scratch            random init, 20 epochs on the AI-READI train split
#   pretrain_no_t1d_*       20 epochs on the T1D-excluded GlucoFM-Bench corpus,
#                           then AI-READI fine-tuning (the reported recipe:
#                           1 epoch at lr 5e-6)
# pretrain_full_* is the same pipeline on the full GlucoFM-Bench corpus
# (T1D included), kept as the comparison that motivated excluding T1D.
#
# Checkpoints are read from $CGMLSM_WORK_ROOT (scratch space written by the
# training/fine-tuning jobs), not from this repo.
#
# Submit all arms:   sbatch cgmlsm_eval_arms.sbatch
# Submit one arm:    sbatch --array=4 cgmlsm_eval_arms.sbatch
#
#SBATCH --job-name=cgmlsm_eval
#SBATCH --partition=normal
#SBATCH --qos=high_p
#SBATCH --time=01:00:00
#SBATCH --mem=32GB
#SBATCH --cpus-per-task=4
#SBATCH -G 1
#SBATCH --array=0-4
#SBATCH --output=logs/%x_%A_%a.out
set -euo pipefail

REPO_ROOT=${AIREADI_ROOT:?set AIREADI_ROOT to your aireadi-collab checkout}
CGMLSM_WORK_ROOT=${CGMLSM_WORK_ROOT:-/path/to/work/aireadi_cgmlsm}

# arm_name | checkpoint dir under $CGMLSM_WORK_ROOT | output dir under $REPO_ROOT/cgmlsm_results
# "-" means leave at cgmlsm_eval.py's own default.
ARMS=(
  "from_scratch|-|-"
  "pretrain_full_zeroshot|pretrain_glucofmbench_v2/final|pretrain_full/zeroshot"
  "pretrain_full_finetuned|finetune_glucofmbench_to_aireadi_v1/final|pretrain_full/finetuned"
  "pretrain_no_t1d_zeroshot|pretrain_glucofmbench_no_t1d_v1/final|pretrain_no_t1d/zeroshot"
  "pretrain_no_t1d_finetuned|finetune_sweep2_ep1_lr5e-6/final|pretrain_no_t1d/finetuned_final"
)

IFS='|' read -r ARM_NAME CKPT_SUBDIR OUT_SUBDIR <<< "${ARMS[${SLURM_ARRAY_TASK_ID:-0}]}"

[ "$CKPT_SUBDIR" != "-" ] && export CGMLSM_CHECKPOINT_DIR="$CGMLSM_WORK_ROOT/$CKPT_SUBDIR"
[ "$OUT_SUBDIR" != "-" ] && export CGMLSM_EVAL_OUTPUT_DIR="$REPO_ROOT/cgmlsm_results/$OUT_SUBDIR"

echo "arm=$ARM_NAME ckpt=${CKPT_SUBDIR} out=${OUT_SUBDIR}"

ml load math py-pytorch/2.4.1_py312
source "$REPO_ROOT/venv-cgmlsm/bin/activate"
python "$REPO_ROOT/code/cgmlsm/cgmlsm_eval.py"
