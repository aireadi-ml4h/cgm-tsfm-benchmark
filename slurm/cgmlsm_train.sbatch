#!/bin/bash
#SBATCH --job-name=cgmlsm_train
#SBATCH --partition=normal
#SBATCH --qos=high_p
#SBATCH --time=02:00:00
#SBATCH --mem=32GB
#SBATCH --cpus-per-task=4
#SBATCH -G 1
#SBATCH --output=logs/%x_%j.out

set -euo pipefail
REPO_ROOT=${AIREADI_ROOT:?set AIREADI_ROOT to your cgm-tsfm-benchmark checkout}
ml load math py-pytorch/2.4.1_py312
source "$REPO_ROOT/venv-cgmlsm/bin/activate"

# Real run: 20 epochs over 22,783 train chunks (~356 steps/epoch, ~7,120
# steps total). cgmlsm_train_smoketest.sbatch measured ~11 it/s at this
# batch size on this GPU/model size (job 41043241), so this should finish
# in well under 2h -- walltime set generously above that estimate per
# feedback_slurm_walltime (re-running a timed-out job costs more than a
# generous --time).
python "$REPO_ROOT/code/cgmlsm/cgmlsm_train.py"
