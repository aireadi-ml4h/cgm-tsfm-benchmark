#!/bin/bash
#SBATCH --job-name=ssmcgm_pers
#SBATCH --partition=normal
#SBATCH --qos=high_p
#SBATCH --time=1-00:00:00
#SBATCH --mem=96GB
#SBATCH --cpus-per-task=8
#SBATCH -G 1
#SBATCH --output=/scratch/users/%u/logs/%x_%j.out

set -euo pipefail
REPO_ROOT=${AIREADI_ROOT:?set AIREADI_ROOT to your aireadi-collab checkout}
ml load math py-pytorch/2.4.1_py312
ml load devel py-pandas/2.2.1_py312 py-pyarrow/18.1.0_py312
source "$REPO_ROOT/venv-ssmcgm/bin/activate"

export SSMCGM_PERS_SPLIT="${SSMCGM_PERS_SPLIT:-test}"
export SSMCGM_PERS_HISTORY_DAYS="${SSMCGM_PERS_HISTORY_DAYS:-1,3,7}"
export SSMCGM_PERS_SHARD="${SSMCGM_PERS_SHARD:-0}"
export SSMCGM_PERS_NUM_SHARDS="${SSMCGM_PERS_NUM_SHARDS:-1}"
python "$REPO_ROOT/code/ssmcgm/ssmcgm_personalization_finetune.py"
