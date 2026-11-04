#!/bin/bash
# SSM-CGM training arms (all splits except paper mode).
#
# All five arms run the same entry point (code/ssmcgm/ssmcgm_train_real.py)
# with identical resources, differing only in the split they train on, whether
# covariates are supplied, and the decoder horizon. Consolidated from five
# near-identical scripts.
#
# ssmcgm_train_paper.sbatch is deliberately NOT folded in here: it requests
# 200 GB rather than 96 GB, and a Slurm array cannot vary --mem per task, so
# merging it would make all five arms over-request memory on a shared cluster.
#
# The three split constructions (see the appendix):
#   disjoint       participant-disjoint, the main cross-model comparison
#   proportional   participants may appear in more than one partition; windows
#                  allocated proportionally -- a robustness check on the
#                  covariate result
# The *_nocov arms are the covariate ablation: same split, covariates removed.
#
# horizon=24 (2 h) is the decoder length used for the 2-hour cross-model
# comparison; the default 12 (1 h) matches the published configuration.
# Decoder length is fixed at training time, so each horizon needs its own run.
#
# The disjoint arm resumes from a prior checkpoint (last.ckpt) by default,
# matching how it was actually run after a walltime cutoff. Clear
# SSMCGM_RESUME_CKPT to train from scratch.
#
# Submit all arms:   sbatch ssmcgm_train_arms.sbatch
# Submit one arm:    sbatch --array=2 ssmcgm_train_arms.sbatch
#
#SBATCH --job-name=ssmcgm_train
#SBATCH --partition=gpu
#SBATCH --qos=high_p
#SBATCH --time=6-00:00:00
#SBATCH --mem=96GB
#SBATCH --cpus-per-task=4
#SBATCH -G 1
#SBATCH --array=0-4
#SBATCH --output=logs/%x_%A_%a.out
set -euo pipefail

REPO_ROOT=${AIREADI_ROOT:?set AIREADI_ROOT to your aireadi-collab checkout}

# arm_name | split csv stem | output subdir | extra env (space-separated KEY=VAL)
# "-" means leave at ssmcgm_train_real.py's own default.
ARMS=(
  "disjoint|-|-|SSMCGM_RESUME_CKPT=$REPO_ROOT/ssmcgm_train_results/12h_batch128/last.ckpt"
  "disjoint_h24|-|12h_batch128_h24|SSMCGM_HORIZON=24"
  "disjoint_nocov|-|disjoint_mode_nocov|SSMCGM_NO_COVARIATES=1"
  "proportional|ssmcgm_train_proportional|proportional_mode|-"
  "proportional_nocov|ssmcgm_train_proportional|proportional_mode_nocov|SSMCGM_NO_COVARIATES=1"
)

IFS='|' read -r ARM_NAME SPLIT_STEM OUT_SUBDIR ARM_ENV <<< "${ARMS[${SLURM_ARRAY_TASK_ID:-0}]}"

export SSMCGM_NUM_GPUS=1
if [ "$SPLIT_STEM" != "-" ]; then
  export SSMCGM_TRAIN_CSV="$REPO_ROOT/model_input/ssmcgm/${SPLIT_STEM}.csv"
  export SSMCGM_VAL_CSV="$REPO_ROOT/model_input/ssmcgm/${SPLIT_STEM/_train_/_val_}.csv"
fi
[ "$OUT_SUBDIR" != "-" ] && export SSMCGM_OUTPUT_DIR="$REPO_ROOT/ssmcgm_train_results/$OUT_SUBDIR"
if [ "$ARM_ENV" != "-" ]; then
  for kv in $ARM_ENV; do export "$kv"; done
fi

echo "arm=$ARM_NAME split=${SPLIT_STEM} out=${OUT_SUBDIR} env=${ARM_ENV}"

ml load math py-pytorch/2.4.1_py312
ml load devel py-pandas/2.2.1_py312 py-pyarrow/18.1.0_py312
source "$REPO_ROOT/venv-ssmcgm/bin/activate"
python "$REPO_ROOT/code/ssmcgm/ssmcgm_train_real.py"
