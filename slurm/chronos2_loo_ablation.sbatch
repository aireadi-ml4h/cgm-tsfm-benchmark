#!/bin/bash
#SBATCH --job-name=chronos2_loo_ablation
#SBATCH --partition=normal
#SBATCH --qos=high_p
#SBATCH --time=12:00:00
#SBATCH --mem=32GB
#SBATCH --cpus-per-task=4
#SBATCH -G 1
#SBATCH --output=logs/%x_%j.out

set -euo pipefail
REPO_ROOT=${AIREADI_ROOT:?set AIREADI_ROOT to your aireadi-collab checkout}
export HF_HOME=$AIREADI_SCRATCH/hf_cache
mkdir -p "$HF_HOME"

ml load math py-pytorch/2.9.1_py314 py-transformers/4.57.3_py314 py-einops/0.8.1_py314 py-pandas/2.3.3_py314 py-pyarrow/22.0.0_py314
source "$REPO_ROOT/venv-chronos/bin/activate"

# Leave-one-out ablation on the imputed (LOCF'd) input, one covariate channel
# dropped at a time. Same model_input/zeroshot_eval/*.csv, same window_index.csv
# as the full-covariate baseline -- only which columns get fed into
# predict_df's df/future_df changes (chronos2_zeroshot_eval.py's
# ACTIVE_COVARIATE_FEATURE_CHANNELS), so every arm scores on the identical
# window set as chronos2_zeroshot_results/full/imputed/ for a clean paired
# comparison. glucose_only is skipped per arm (no covariates to ablate,
# identical across all arms and already in the baseline results). All
# chronos2 results live under one parent (chronos2_zeroshot_results/); LOO
# arms go under loo/, full runs under full/.
CHANNELS=(heart_rate stress respiratory_rate oxygen_saturation physical_activity_steps physical_activity_calorie_rate)

export CHRONOS2_ONLY_WITH_COVARIATES=1
for channel in "${CHANNELS[@]}"; do
    echo "=== LOO ablation: dropping $channel ===" >&2
    export CHRONOS2_ABLATE_CHANNELS="$channel"
    export CHRONOS2_RESULTS_DIR_OVERRIDE="$REPO_ROOT/chronos2_zeroshot_results/loo/drop_${channel}"
    python "$REPO_ROOT/code/chronos2/chronos2_zeroshot_eval.py"
done
