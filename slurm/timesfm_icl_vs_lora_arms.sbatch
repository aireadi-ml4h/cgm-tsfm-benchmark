#!/bin/bash
# TimesFM in-context vs. LoRA personalization comparison, before and after
# re-tuning the LoRA hyperparameters.
#
# Both arms run code/analysis/timesfm_incontext_vs_lora_comparison.py with identical
# resources. Consolidated from two near-identical scripts.
#
#   untuned  compares against the initial LoRA configuration
#   tuned    compares against LoRA after its learning rate and update-step
#            count were re-selected on validation participants. This is the
#            comparison reported in the paper -- ICL still wins at every
#            horizon, so the ICL advantage is not an artifact of an
#            under-tuned LoRA baseline.
#
#SBATCH --job-name=timesfm_icl_vs_lora
#SBATCH --partition=normal
#SBATCH --time=00:30:00
#SBATCH --mem=16GB
#SBATCH --cpus-per-task=4
#SBATCH --array=0-1
#SBATCH --output=logs/%x_%A_%a.out
set -euo pipefail
REPO_ROOT=${AIREADI_ROOT:?set AIREADI_ROOT to your cgm-tsfm-benchmark checkout}

ARMS=("untuned|-" "tuned|TIMESFM_LORA_SUFFIX=_tuned")
IFS='|' read -r ARM_NAME ARM_ENV <<< "${ARMS[${SLURM_ARRAY_TASK_ID:-0}]}"
[ "$ARM_ENV" != "-" ] && export "$ARM_ENV"
echo "arm=$ARM_NAME env=$ARM_ENV"

ml load math py-pytorch/2.9.1_py314 py-transformers/4.57.3_py314 py-einops/0.8.1_py314 py-pandas/2.3.3_py314 py-pyarrow/22.0.0_py314
source "$REPO_ROOT/venv-chronos/bin/activate"
python "$REPO_ROOT/code/analysis/timesfm_incontext_vs_lora_comparison.py"
