#!/bin/bash
#SBATCH --job-name=build_ssmcgm_pop_split
#SBATCH --partition=normal
#SBATCH --qos=high_p
#SBATCH --time=01:00:00
#SBATCH --mem=16GB
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/%x_%j.out

set -euo pipefail
REPO_ROOT=${AIREADI_ROOT:?set AIREADI_ROOT to your cgm-tsfm-benchmark checkout}
ml load math py-pytorch/2.4.1_py312
ml load devel py-pandas/2.2.1_py312 py-pyarrow/18.1.0_py312
source "$REPO_ROOT/venv-ssmcgm/bin/activate"

: "${SSMCGM_POP_SPLIT_MODE:?set SSMCGM_POP_SPLIT_MODE to paper or proportional}"
export SSMCGM_POP_SPLIT_MODE
python "$REPO_ROOT/code/model_input/build_ssmcgm_population_split_input.py"
