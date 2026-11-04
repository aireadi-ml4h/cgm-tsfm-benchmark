#!/bin/bash
# CGM-LSM AI-READI fine-tuning arms (after GlucoFM-Bench pretraining).
#
# Both arms run the same entry point (code/cgmlsm/cgmlsm_finetune_aireadi.py) with
# identical resources, differing only in which pretrained checkpoint they start
# from and the epoch/learning-rate schedule. Consolidated from three scripts
# (two configs plus a generic --export runner used by the now-removed sweep).
#
#   pretrain_full     defaults baked into cgmlsm_finetune_aireadi.py: starts
#                     from the full-corpus (T1D-included) pretrain, 10 epochs
#   no_t1d_final      the recipe reported in the paper: starts from the
#                     T1D-excluded pretrain, 1 epoch at lr 5e-6
#
# Runtime is minutes, not the hour requested: ~356 steps/epoch over 22,783
# AI-READI train chunks, and the comparable from-scratch run did 7,120 steps
# in ~239 s on the same GPU. The 1 h walltime is deliberate headroom.
#
# Checkpoints are read from and written to $CGMLSM_WORK_ROOT (scratch), not
# this repo.
#
# Submit both arms:  sbatch cgmlsm_finetune_arms.sbatch
# Submit one arm:    sbatch --array=1 cgmlsm_finetune_arms.sbatch
#
#SBATCH --job-name=cgmlsm_finetune
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
CGMLSM_WORK_ROOT=${CGMLSM_WORK_ROOT:-/path/to/work/aireadi_cgmlsm}

# arm_name | pretrained ckpt subdir | output subdir | epochs | learning rate
# "-" means leave at cgmlsm_finetune_aireadi.py's own default.
ARMS=(
  "pretrain_full|-|-|-|-"
  "no_t1d_final|pretrain_glucofmbench_no_t1d_v1/final|finetune_sweep2_ep1_lr5e-6|1|5e-6"
)

IFS='|' read -r ARM_NAME CKPT_SUBDIR OUT_SUBDIR EPOCHS LR <<< "${ARMS[${SLURM_ARRAY_TASK_ID:-0}]}"

[ "$CKPT_SUBDIR" != "-" ] && export CGMLSM_PRETRAINED_CHECKPOINT_DIR="$CGMLSM_WORK_ROOT/$CKPT_SUBDIR"
[ "$OUT_SUBDIR" != "-" ] && export CGMLSM_FINETUNE_OUTPUT_DIR_OVERRIDE="$CGMLSM_WORK_ROOT/$OUT_SUBDIR"
[ "$EPOCHS" != "-" ] && export CGMLSM_NUM_EPOCHS="$EPOCHS"
[ "$LR" != "-" ] && export CGMLSM_LEARNING_RATE="$LR"

echo "arm=$ARM_NAME ckpt=${CKPT_SUBDIR} out=${OUT_SUBDIR} epochs=${EPOCHS} lr=${LR}"

ml load math py-pytorch/2.4.1_py312
source "$REPO_ROOT/venv-cgmlsm/bin/activate"
python "$REPO_ROOT/code/cgmlsm/cgmlsm_finetune_aireadi.py"
