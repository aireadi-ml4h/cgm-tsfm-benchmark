#!/bin/bash
#SBATCH --job-name=cgmlsm_arm_matched
#SBATCH --partition=normal
#SBATCH --time=00:20:00
#SBATCH --mem=16GB
#SBATCH --cpus-per-task=2
#SBATCH --output=/scratch/users/%u/logs/%x_%j.out

# CPU-only: reads the per-bin predictions CSVs both arms already wrote, runs no
# model, so no GPU and --partition can include `normal`.
set -euo pipefail
REPO_ROOT=${AIREADI_ROOT:?set AIREADI_ROOT to your aireadi-collab checkout}
ml load math py-pytorch/2.4.1_py312
source "$REPO_ROOT/venv-cgmlsm/bin/activate"

python "$REPO_ROOT/code/analysis/cgmlsm_arm_matched_comparison.py"
