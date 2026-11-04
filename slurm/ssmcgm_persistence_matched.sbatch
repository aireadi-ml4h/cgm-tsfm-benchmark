#!/bin/bash
#SBATCH --job-name=ssmcgm_persistence_matched
#SBATCH --partition=normal
#SBATCH --qos=normal
#SBATCH --time=02:00:00
#SBATCH --mem=64GB
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/%x_%j.out
#
# Persistence baseline on SSM-CGM's OWN windows -- the matched-window check
# T9 flags as the most important open item.
#
# No GPU and no checkpoint: this only iterates the dataloader and reads
# x["encoder_target"] / x["decoder_target"]. qos=normal so it does not take a
# high_p slot from the GPU training jobs.
#
# 64GB (not 32GB): the script reads back a large streamed CSV at the end, the
# same step that made ssmcgm_test_eval.py's memory profile non-trivial.
set -euo pipefail
REPO_ROOT=${AIREADI_ROOT:?set AIREADI_ROOT to your aireadi-collab checkout}
ml load math py-pytorch/2.4.1_py312
ml load devel py-pandas/2.2.1_py312 py-pyarrow/18.1.0_py312
source "$REPO_ROOT/venv-ssmcgm/bin/activate"

export SSMCGM_HORIZON="${SSMCGM_HORIZON:-24}"
python "$REPO_ROOT/code/ssmcgm/ssmcgm_persistence_matched.py"
