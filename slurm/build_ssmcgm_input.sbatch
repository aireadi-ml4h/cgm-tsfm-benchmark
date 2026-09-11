#!/bin/bash
#SBATCH --job-name=build_ssmcgm_input
#SBATCH --partition=normal
#SBATCH --qos=high_p
#SBATCH --time=01:00:00
#SBATCH --mem=16GB
#SBATCH --cpus-per-task=4
#SBATCH --output=/scratch/users/%u/logs/%x_%j.out

set -euo pipefail
REPO_ROOT=${AIREADI_ROOT:?set AIREADI_ROOT to your aireadi-collab checkout}
ml load math py-pytorch/2.4.1_py312
ml load devel py-pandas/2.2.1_py312 py-pyarrow/18.1.0_py312
source "$REPO_ROOT/venv-ssmcgm/bin/activate"

# Regenerates model_input/ssmcgm/ssmcgm_{train,val,test}.csv now that
# predmeal_flag is real (code/meal_detection/generate_meal_probability.py's
# output, joined in via build_participant_grid.py), not the earlier
# disclosed placeholder -- see build_ssmcgm_input.py's module docstring.
python "$REPO_ROOT/code/model_input/build_ssmcgm_input.py"
