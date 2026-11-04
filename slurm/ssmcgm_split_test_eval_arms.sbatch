#!/bin/bash
# SSM-CGM test evaluation for the alternative split constructions.
#
# Both arms run code/ssmcgm/ssmcgm_val_eval_multihorizon.py (which despite
# its name is the multi-horizon scorer used for these test evaluations) with
# identical resources. Consolidated from two near-identical scripts.
#
#   paper         the per-participant temporal split matching the original
#                 paper's own within-participant holdout
#   proportional  participants may appear in more than one partition, windows
#                 allocated proportionally -- the robustness check on the
#                 covariate result
#
# The participant-disjoint split (the main comparison) is scored by
# ssmcgm_test_eval_arms.sbatch instead.
#
# Checkpoints are the lowest-validation-loss epoch from each training run.
#
#SBATCH --job-name=ssmcgm_split_test
#SBATCH --partition=normal
#SBATCH --qos=high_p
#SBATCH --time=01:00:00
#SBATCH --mem=32GB
#SBATCH --cpus-per-task=4
#SBATCH -G 1
#SBATCH --array=0-1
#SBATCH --output=logs/%x_%A_%a.out
set -euo pipefail
REPO_ROOT=${AIREADI_ROOT:?set AIREADI_ROOT to your aireadi-collab checkout}

# arm | split stem | checkpoint (under ssmcgm_train_results) | output subdir
ARMS=(
  "paper|paper|paper_mode/best-epoch=2-val_loss=16.1610.ckpt|test_eval_paper"
  "proportional|proportional|proportional_mode/best-epoch=0-val_loss=14.3673.ckpt|test_eval_proportional"
)
IFS='|' read -r ARM_NAME STEM CKPT OUT <<< "${ARMS[${SLURM_ARRAY_TASK_ID:-0}]}"

export SSMCGM_VAL_EVAL_TRAIN_CSV="$REPO_ROOT/model_input/ssmcgm/ssmcgm_train_${STEM}.csv"
export SSMCGM_VAL_EVAL_VAL_CSV="$REPO_ROOT/model_input/ssmcgm/ssmcgm_test_${STEM}.csv"
export SSMCGM_VAL_EVAL_CKPT="$REPO_ROOT/ssmcgm_train_results/$CKPT"
export SSMCGM_VAL_EVAL_OUTPUT_DIR="$REPO_ROOT/ssmcgm_train_results/$OUT"
echo "arm=$ARM_NAME split=$STEM"

ml load math py-pytorch/2.4.1_py312
ml load devel py-pandas/2.2.1_py312 py-pyarrow/18.1.0_py312
source "$REPO_ROOT/venv-ssmcgm/bin/activate"
python "$REPO_ROOT/code/ssmcgm/ssmcgm_val_eval_multihorizon.py"
