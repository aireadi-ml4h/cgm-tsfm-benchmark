#!/bin/bash
#SBATCH --job-name=lgbm_cov_loo
#SBATCH --partition=normal
#SBATCH --qos=normal
#SBATCH --time=02:00:00
#SBATCH --mem=32GB
#SBATCH --cpus-per-task=8
#SBATCH --output=logs/%x_%j.out

set -euo pipefail
REPO_ROOT=${AIREADI_ROOT:?set AIREADI_ROOT to your aireadi-collab checkout}
ml load devel py-numpy/1.26.3_py312 py-pandas/2.2.1_py312
ml load math py-scipy/1.12.0_py312 py-scikit-learn/1.5.1_py312 py-lightgbm/4.5.0_py312

python3 "$REPO_ROOT/code/baselines/lightgbm_zeroshot_baseline_covariates.py"
