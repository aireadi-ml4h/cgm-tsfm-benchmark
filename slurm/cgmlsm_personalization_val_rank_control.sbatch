#!/bin/bash
#SBATCH --job-name=cgmlsm_pers_rank_ctrl
#SBATCH --partition=normal
#SBATCH --time=01:00:00
#SBATCH --mem=24GB
#SBATCH --cpus-per-task=2
#SBATCH --output=logs/%x_%j.out
set -euo pipefail
REPO_ROOT=${AIREADI_ROOT:?set AIREADI_ROOT to your aireadi-collab checkout}
ml load math py-pytorch/2.4.1_py312
source "$REPO_ROOT/venv-cgmlsm/bin/activate"
python "$REPO_ROOT/code/analysis/cgmlsm_personalization_val_rank_control.py"
