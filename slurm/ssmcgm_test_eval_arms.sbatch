#!/bin/bash
# SSM-CGM test evaluation on the participant-disjoint split, by decoder length.
#
# Both arms run code/ssmcgm/ssmcgm_test_eval.py with identical resources.
# Consolidated from two near-identical scripts.
#
# Decoder length is fixed at training time and one checkpoint scores every
# horizon up to that length in a single pass, so each decoder needs its own
# checkpoint and its own eval:
#   h12  the published 1-hour configuration (script defaults)
#   h24  the 2-hour decoder used for the 2-hour cross-model comparison
#
#SBATCH --job-name=ssmcgm_test_eval
#SBATCH --partition=normal
#SBATCH --qos=high_p
#SBATCH --time=01:00:00
#SBATCH --mem=32GB
#SBATCH --cpus-per-task=4
#SBATCH -G 1
#SBATCH --array=0-1
#SBATCH --output=/scratch/users/%u/logs/%x_%A_%a.out
set -euo pipefail
REPO_ROOT=${AIREADI_ROOT:?set AIREADI_ROOT to your aireadi-collab checkout}

# arm | horizon | checkpoint (under ssmcgm_train_results) | output subdir
ARMS=(
  "h12|-|-|-"
  "h24|24|12h_batch128_h24/best-epoch=0-val_loss=12.2779.ckpt|test_eval_h24"
)
IFS='|' read -r ARM_NAME HORIZON CKPT OUT <<< "${ARMS[${SLURM_ARRAY_TASK_ID:-0}]}"
[ "$HORIZON" != "-" ] && export SSMCGM_TEST_HORIZON="$HORIZON"
[ "$CKPT" != "-" ] && export SSMCGM_TEST_CKPT="$REPO_ROOT/ssmcgm_train_results/$CKPT"
[ "$OUT" != "-" ] && export SSMCGM_TEST_OUTPUT_DIR="$REPO_ROOT/ssmcgm_train_results/$OUT"
echo "arm=$ARM_NAME horizon=$HORIZON"

ml load math py-pytorch/2.4.1_py312
ml load devel py-pandas/2.2.1_py312 py-pyarrow/18.1.0_py312
source "$REPO_ROOT/venv-ssmcgm/bin/activate"
python "$REPO_ROOT/code/ssmcgm/ssmcgm_test_eval.py"
