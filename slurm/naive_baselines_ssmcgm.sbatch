#!/bin/bash
#SBATCH --job-name=naive_baselines_ssmcgm
#SBATCH --partition=normal
#SBATCH --time=00:30:00
#SBATCH --mem=32GB
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/%x_%j.out

set -euo pipefail
REPO_ROOT=${AIREADI_ROOT:?set AIREADI_ROOT to your cgm-tsfm-benchmark checkout}
ml load math py-pytorch/2.4.1_py312
ml load devel py-pandas/2.2.1_py312 py-pyarrow/18.1.0_py312
source "$REPO_ROOT/venv-ssmcgm/bin/activate"

python "$REPO_ROOT/code/baselines/naive_baselines_ssmcgm.py"
