#!/bin/bash
# TimesFM 2.5 single-covariate ("isolated channel") ablation arms.
#
# Mirrors chronos2_isolated_arms.sbatch: every arm runs the same entry point
# (code/timesfm/timesfm_zeroshot_eval.py) on the same covariate-eligible windows with
# the same resources, differing only in which covariate channels reach the
# model. Consolidated into one array job from eleven near-identical scripts.
#
# The six base wearable channels are:
#   heart_rate, stress, respiratory_rate, oxygen_saturation,
#   physical_activity_steps, physical_activity_calorie_rate
#
# TIMESFM_ABLATE_CHANNELS lists the channels REMOVED, so a "<channel>_only"
# arm ablates the other five. Derived channels (sleep stage, sleep regularity,
# predicted meal probability, time-of-day, static attributes) are not in the
# base set, so their arms ablate all six and opt the derived channel back in
# via its own TIMESFM_INCLUDE_* flag.
#
# Two arms are not single-channel isolations:
#   drop_harmful  keeps everything except steps and stress (the two channels
#                 that hurt individually)
#
# The heart-rate isolation for TimesFM lives in
# ../timesfm_zeroshot_eval_heartrate_only.sbatch, not here.
#
# Note the 40 min walltime vs. Chronos-2's 20 min: covariate inference goes
# through forecast_with_covariates(), whose in-context regression step needs
# jax + scikit-learn and is slower.
#
# Submit all arms:        sbatch timesfm_isolated_arms.sbatch
# Submit one arm:         sbatch --array=2 timesfm_isolated_arms.sbatch
#
#SBATCH --job-name=tfm_isolated
#SBATCH --partition=gpu
#SBATCH --time=00:40:00
#SBATCH --mem=32GB
#SBATCH --cpus-per-task=4
#SBATCH -G 1
#SBATCH --array=0-10
#SBATCH --output=/scratch/users/%u/logs/%x_%A_%a.out
set -euo pipefail

# arm_name | channels to ablate | extra TIMESFM_INCLUDE_* flag (may be empty)
ARMS=(
  "stress_only|heart_rate,respiratory_rate,oxygen_saturation,physical_activity_steps,physical_activity_calorie_rate|"
  "respiratory_rate_only|heart_rate,stress,oxygen_saturation,physical_activity_steps,physical_activity_calorie_rate|"
  "oxygen_saturation_only|heart_rate,stress,respiratory_rate,physical_activity_steps,physical_activity_calorie_rate|"
  "steps_only|heart_rate,stress,respiratory_rate,oxygen_saturation,physical_activity_calorie_rate|"
  "calorie_only|heart_rate,stress,respiratory_rate,oxygen_saturation,physical_activity_steps|"
  "sleep_stage_only|heart_rate,stress,respiratory_rate,oxygen_saturation,physical_activity_steps,physical_activity_calorie_rate|TIMESFM_INCLUDE_SLEEP_STAGE"
  "sleep_regularity_only|heart_rate,stress,respiratory_rate,oxygen_saturation,physical_activity_steps,physical_activity_calorie_rate|TIMESFM_INCLUDE_SLEEP_REGULARITY"
  "predmeal_prob_only|heart_rate,stress,respiratory_rate,oxygen_saturation,physical_activity_steps,physical_activity_calorie_rate|TIMESFM_INCLUDE_PREDMEAL_PROB"
  "tod_only|heart_rate,stress,respiratory_rate,oxygen_saturation,physical_activity_steps,physical_activity_calorie_rate|TIMESFM_INCLUDE_TOD"
  "static_only|heart_rate,stress,respiratory_rate,oxygen_saturation,physical_activity_steps,physical_activity_calorie_rate|TIMESFM_INCLUDE_STATIC"
  "drop_harmful|physical_activity_steps,stress|"
)

IFS='|' read -r ARM_NAME ABLATE EXTRA_FLAG <<< "${ARMS[${SLURM_ARRAY_TASK_ID:-0}]}"

REPO_ROOT=${AIREADI_ROOT:?set AIREADI_ROOT to your aireadi-collab checkout}
export HF_HOME=$SCRATCH/hf_cache
mkdir -p "$HF_HOME"

export TIMESFM_ONLY_WITH_COVARIATES=1
export TIMESFM_ABLATE_CHANNELS="$ABLATE"
# drop_harmful kept its original (un-suffixed) results directory.
if [ "$ARM_NAME" = "drop_harmful" ]; then
  export TIMESFM_RESULTS_DIR_OVERRIDE="$REPO_ROOT/timesfm_zeroshot_results/full/drop_harmful"
else
  export TIMESFM_RESULTS_DIR_OVERRIDE="$REPO_ROOT/timesfm_zeroshot_results/full/${ARM_NAME}_v2"
fi
[ -n "$EXTRA_FLAG" ] && export "$EXTRA_FLAG=1"

echo "arm=$ARM_NAME ablate=$ABLATE extra=${EXTRA_FLAG:-none}"

ml load math py-pytorch/2.9.1_py314 py-huggingface-hub/0.36.0_py314 py-safetensors/0.7.0_py314 py-numpy/2.3.5_py314 py-pandas/2.3.3_py314 py-pyarrow/22.0.0_py314
ml load math py-jax/0.8.1_py314 py-scikit-learn/1.8.0_py314
source "$REPO_ROOT/venv-timesfm/bin/activate"
python "$REPO_ROOT/code/timesfm/timesfm_zeroshot_eval.py"
