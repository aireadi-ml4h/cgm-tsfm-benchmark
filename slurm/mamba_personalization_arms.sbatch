#!/bin/bash
# Mamba per-participant personalization: real history vs. shuffled control.
#
# Both arms run code/baselines/mamba_personalization.py with identical resources.
# Consolidated from two near-identical scripts. See
# lstm_personalization_arms.sbatch for what the shuffled control establishes.
#
# Walltime is the longer of the two arms (2 h; the control ran in 1 h).
#
#SBATCH --job-name=mamba_personalization
#SBATCH --partition=normal
#SBATCH --qos=high_p
#SBATCH --time=02:00:00
#SBATCH --mem=32GB
#SBATCH --cpus-per-task=4
#SBATCH -G 1
#SBATCH --array=0-1
#SBATCH --output=/scratch/users/%u/logs/%x_%A_%a.out
set -euo pipefail
REPO_ROOT=${AIREADI_ROOT:?set AIREADI_ROOT to your aireadi-collab checkout}

ARMS=("real|-" "shuffled|MAMBA_PERS_SHUFFLE=1")
IFS='|' read -r ARM_NAME ARM_ENV <<< "${ARMS[${SLURM_ARRAY_TASK_ID:-0}]}"
[ "$ARM_ENV" != "-" ] && export "$ARM_ENV"
echo "arm=$ARM_NAME env=$ARM_ENV"

ml load math py-pytorch/2.4.1_py312 py-scipy/1.12.0_py312
ml load devel py-pandas/2.2.1_py312 py-pyarrow/18.1.0_py312
source "$REPO_ROOT/venv-ssmcgm/bin/activate"
python "$REPO_ROOT/code/baselines/mamba_personalization.py"
