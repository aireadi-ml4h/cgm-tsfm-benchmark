#!/bin/bash
#SBATCH --job-name=mamba_zeroshot
#SBATCH --partition=normal
#SBATCH --qos=high_p
#SBATCH --time=02:00:00
#SBATCH --mem=32GB
#SBATCH --cpus-per-task=4
#SBATCH -G 1
#SBATCH --output=logs/%x_%j.out

set -euo pipefail
REPO_ROOT=${AIREADI_ROOT:?set AIREADI_ROOT to your cgm-tsfm-benchmark checkout}
ml load math py-pytorch/2.4.1_py312 py-scipy/1.12.0_py312
ml load devel py-pandas/2.2.1_py312 py-pyarrow/18.1.0_py312
source "$REPO_ROOT/venv-ssmcgm/bin/activate"

python "$REPO_ROOT/code/baselines/mamba_zeroshot_baseline.py"
