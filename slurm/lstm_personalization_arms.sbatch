#!/bin/bash
# LSTM per-participant personalization: real history vs. shuffled control.
#
# Both arms run code/baselines/lstm_personalization.py with identical resources.
# Consolidated from two near-identical scripts.
#
#   real      fit each participant's adapter on that participant's own history
#   shuffled  NEGATIVE CONTROL -- fit on a different participant's history.
#             If personalization gains were an artifact of extra training
#             rather than adaptation to the individual, this arm would match
#             the real one. It does not.
#
# Walltime is the longer of the two arms (2 h; the control ran in 1 h).
#
# Submit both:     sbatch lstm_personalization_arms.sbatch
# Submit one arm:  sbatch --array=1 lstm_personalization_arms.sbatch
#
#SBATCH --job-name=lstm_personalization
#SBATCH --partition=normal
#SBATCH --qos=high_p
#SBATCH --time=02:00:00
#SBATCH --mem=32GB
#SBATCH --cpus-per-task=4
#SBATCH -G 1
#SBATCH --array=0-1
#SBATCH --output=logs/%x_%A_%a.out
set -euo pipefail
REPO_ROOT=${AIREADI_ROOT:?set AIREADI_ROOT to your aireadi-collab checkout}

ARMS=("real|-" "shuffled|LSTM_PERS_SHUFFLE=1")
IFS='|' read -r ARM_NAME ARM_ENV <<< "${ARMS[${SLURM_ARRAY_TASK_ID:-0}]}"
[ "$ARM_ENV" != "-" ] && export "$ARM_ENV"
echo "arm=$ARM_NAME env=$ARM_ENV"

ml load math py-pytorch/2.9.1_py314 py-pandas/2.3.3_py314 py-pyarrow/22.0.0_py314
source "$REPO_ROOT/venv-chronos/bin/activate"
python3 "$REPO_ROOT/code/baselines/lstm_personalization.py"
