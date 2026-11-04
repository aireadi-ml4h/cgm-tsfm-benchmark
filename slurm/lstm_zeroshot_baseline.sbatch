#!/bin/bash
#SBATCH --job-name=lstm_zeroshot
#SBATCH --partition=normal
#SBATCH --qos=high_p
#SBATCH --time=01:30:00
#SBATCH --mem=32GB
#SBATCH --cpus-per-task=4
#SBATCH -G 1
#SBATCH --output=logs/%x_%j.out

set -euo pipefail
REPO_ROOT=${AIREADI_ROOT:?set AIREADI_ROOT to your cgm-tsfm-benchmark checkout}
ml load math py-pytorch/2.9.1_py314 py-pandas/2.3.3_py314 py-pyarrow/22.0.0_py314
source "$REPO_ROOT/venv-chronos/bin/activate"

python "$REPO_ROOT/code/baselines/lstm_zeroshot_baseline.py"
