#!/bin/bash
#SBATCH --job-name=timesfm_loo_ablation
#SBATCH --partition=normal
#SBATCH --qos=high_p
#SBATCH --time=12:00:00
#SBATCH --mem=32GB
#SBATCH --cpus-per-task=4
#SBATCH -G 1
#SBATCH --output=/scratch/users/%u/logs/%x_%j.out

set -euo pipefail
REPO_ROOT=${AIREADI_ROOT:?set AIREADI_ROOT to your aireadi-collab checkout}
export HF_HOME=$SCRATCH/hf_cache
mkdir -p "$HF_HOME"

ml load math py-pytorch/2.9.1_py314 py-huggingface-hub/0.36.0_py314 py-safetensors/0.7.0_py314 py-numpy/2.3.5_py314 py-pandas/2.3.3_py314 py-pyarrow/22.0.0_py314
ml load math py-jax/0.8.1_py314 py-scikit-learn/1.8.0_py314
source "$REPO_ROOT/venv-timesfm/bin/activate"

# Mirrors slurm/chronos2_loo_ablation.sbatch exactly: same window_index.csv,
# same model_input/zeroshot_eval/*.csv, one covariate channel dropped at a
# time from ACTIVE_COVARIATE_FEATURE_CHANNELS, glucose_only skipped per arm
# (identical across arms, already in the full baseline). All timesfm results
# live under one parent (timesfm_zeroshot_results/); LOO arms go under loo/,
# full runs under full/.
CHANNELS=(heart_rate stress respiratory_rate oxygen_saturation physical_activity_steps physical_activity_calorie_rate)

export TIMESFM_ONLY_WITH_COVARIATES=1
for channel in "${CHANNELS[@]}"; do
    echo "=== LOO ablation: dropping $channel ===" >&2
    export TIMESFM_ABLATE_CHANNELS="$channel"
    export TIMESFM_RESULTS_DIR_OVERRIDE="$REPO_ROOT/timesfm_zeroshot_results/loo/drop_${channel}"
    python "$REPO_ROOT/code/timesfm/timesfm_zeroshot_eval.py"
done
