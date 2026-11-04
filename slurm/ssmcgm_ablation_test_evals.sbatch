#!/bin/bash
#SBATCH --job-name=ssmcgm_ablation_test
#SBATCH --partition=normal
#SBATCH --qos=normal
#SBATCH --time=02:00:00
#SBATCH --mem=64GB
#SBATCH --cpus-per-task=4
#SBATCH -G 1
#SBATCH --array=0-2
#SBATCH --output=logs/%x_%A_%a.out
#
# Test-set MAE for the three SSM-CGM covariate-ablation cells that have none.
# Closes AUDIT.md's "test MAE for only one of four cells" warning, so the
# ablation conclusion can rest on test data rather than validation loss.
#
# Uses ssmcgm_test_eval_v2.py, which adds SSMCGM_TRAIN_CSV and
# SSMCGM_NO_COVARIATES to the proven test_eval.py. Both are required here:
# the proportional cells are built from a different split, and the nocov
# cells were trained on a stripped feature schema that the dataset built at
# eval time must match exactly or the checkpoint will not load.
#
# The checkpoint is chosen by MINIMUM val_loss parsed from the filename, not
# by `ls | head` and not by whatever the training script printed as "best" --
# its ModelCheckpoint top-k ranking resets on every Slurm requeue.
set -euo pipefail
REPO_ROOT=${AIREADI_ROOT:?set AIREADI_ROOT to your aireadi-collab checkout}
CELLS=(disjoint_mode_nocov proportional_mode proportional_mode_nocov)
NOCOV=(1 0 1)
SPLIT=(ssmcgm ssmcgm_proportional ssmcgm_proportional)

CELL=${CELLS[$SLURM_ARRAY_TASK_ID]}
export SSMCGM_NO_COVARIATES=${NOCOV[$SLURM_ARRAY_TASK_ID]}
TAG=${SPLIT[$SLURM_ARRAY_TASK_ID]}

if [ "$TAG" = "ssmcgm" ]; then
  export SSMCGM_TRAIN_CSV="$REPO_ROOT/model_input/ssmcgm/ssmcgm_train.csv"
  export SSMCGM_TEST_CSV="$REPO_ROOT/model_input/ssmcgm/ssmcgm_test.csv"
else
  export SSMCGM_TRAIN_CSV="$REPO_ROOT/model_input/ssmcgm/ssmcgm_train_proportional.csv"
  export SSMCGM_TEST_CSV="$REPO_ROOT/model_input/ssmcgm/ssmcgm_test_proportional.csv"
fi

CKPT=$(ls "$REPO_ROOT/ssmcgm_train_results/$CELL"/best-epoch=*.ckpt \
       | sed -E 's/.*val_loss=([0-9.]+)\.ckpt/\1 &/' | sort -g | head -1 | cut -d' ' -f2-)
export SSMCGM_TEST_CKPT="$CKPT"
export SSMCGM_TEST_OUTPUT_DIR="$REPO_ROOT/ssmcgm_train_results/test_eval_${CELL}"

echo "cell=$CELL nocov=$SSMCGM_NO_COVARIATES"
echo "ckpt=$CKPT"
echo "train=$SSMCGM_TRAIN_CSV"
echo "test=$SSMCGM_TEST_CSV"

ml load math py-pytorch/2.4.1_py312
ml load devel py-pandas/2.2.1_py312 py-pyarrow/18.1.0_py312
source "$REPO_ROOT/venv-ssmcgm/bin/activate"
python "$REPO_ROOT/code/ssmcgm/ssmcgm_test_eval_v2.py"
