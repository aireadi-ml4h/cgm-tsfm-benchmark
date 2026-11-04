#!/bin/bash
#SBATCH --job-name=cov_per_part_sig
#SBATCH --partition=normal
#SBATCH --time=00:15:00
#SBATCH --mem=8GB
#SBATCH --cpus-per-task=2
#SBATCH --output=logs/%x_%j.out
set -euo pipefail
REPO_ROOT=${AIREADI_ROOT:?set AIREADI_ROOT to your aireadi-collab checkout}
ml load math py-pytorch/2.4.1_py312
ml load devel py-pandas/2.2.1_py312
source "$REPO_ROOT/venv/bin/activate"
python "$REPO_ROOT/code/analysis/covariate_per_participant_significance.py"
