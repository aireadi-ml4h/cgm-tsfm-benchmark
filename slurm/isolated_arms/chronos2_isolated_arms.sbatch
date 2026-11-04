#!/bin/bash
# Chronos-2 single-covariate ("isolated channel") ablation arms.
#
# Every arm runs the same entry point (code/chronos2/chronos2_zeroshot_eval.py) on the
# same covariate-eligible windows with the same resources; they differ only in
# which covariate channels reach the model. Consolidated into one array job
# from twelve near-identical scripts -- the per-arm config is the table below.
#
# The six base wearable channels are:
#   heart_rate, stress, respiratory_rate, oxygen_saturation,
#   physical_activity_steps, physical_activity_calorie_rate
#
# CHRONOS2_ABLATE_CHANNELS lists the channels REMOVED, so a "<channel>_only"
# arm ablates the other five. The derived channels (sleep stage, sleep
# regularity, predicted meal probability, time-of-day, static attributes) are
# not in the base set, so their arms ablate all six and opt the derived channel
# back in via its own CHRONOS2_INCLUDE_* flag.
#
# Two arms are not single-channel isolations:
#   combined_helpful  keeps heart_rate + stress + calorie_rate together
#                     (the channels that helped individually at 2-4 h)
#
# Submit all arms:        sbatch chronos2_isolated_arms.sbatch
# Submit one arm:         sbatch --array=2 chronos2_isolated_arms.sbatch
#
#SBATCH --job-name=c2_isolated
#SBATCH --partition=gpu
#SBATCH --time=00:20:00
#SBATCH --mem=32GB
#SBATCH --cpus-per-task=4
#SBATCH -G 1
#SBATCH --array=0-11
#SBATCH --output=logs/%x_%A_%a.out
set -euo pipefail

# arm_name | channels to ablate | extra CHRONOS2_INCLUDE_* flag (may be empty)
ARMS=(
  "heartrate_only|stress,respiratory_rate,oxygen_saturation,physical_activity_steps,physical_activity_calorie_rate|"
  "stress_only|heart_rate,respiratory_rate,oxygen_saturation,physical_activity_steps,physical_activity_calorie_rate|"
  "respiratory_rate_only|heart_rate,stress,oxygen_saturation,physical_activity_steps,physical_activity_calorie_rate|"
  "oxygen_saturation_only|heart_rate,stress,respiratory_rate,physical_activity_steps,physical_activity_calorie_rate|"
  "steps_only|heart_rate,stress,respiratory_rate,oxygen_saturation,physical_activity_calorie_rate|"
  "calorie_only|heart_rate,stress,respiratory_rate,oxygen_saturation,physical_activity_steps|"
  "sleep_stage_only|heart_rate,stress,respiratory_rate,oxygen_saturation,physical_activity_steps,physical_activity_calorie_rate|CHRONOS2_INCLUDE_SLEEP_STAGE"
  "sleep_regularity_only|heart_rate,stress,respiratory_rate,oxygen_saturation,physical_activity_steps,physical_activity_calorie_rate|CHRONOS2_INCLUDE_SLEEP_REGULARITY"
  "predmeal_prob_only|heart_rate,stress,respiratory_rate,oxygen_saturation,physical_activity_steps,physical_activity_calorie_rate|CHRONOS2_INCLUDE_PREDMEAL_PROB"
  "tod_only|heart_rate,stress,respiratory_rate,oxygen_saturation,physical_activity_steps,physical_activity_calorie_rate|CHRONOS2_INCLUDE_TOD"
  "static_only|heart_rate,stress,respiratory_rate,oxygen_saturation,physical_activity_steps,physical_activity_calorie_rate|CHRONOS2_INCLUDE_STATIC"
  "combined_helpful|respiratory_rate,oxygen_saturation,physical_activity_steps|"
)

IFS='|' read -r ARM_NAME ABLATE EXTRA_FLAG <<< "${ARMS[${SLURM_ARRAY_TASK_ID:-0}]}"

REPO_ROOT=${AIREADI_ROOT:?set AIREADI_ROOT to your cgm-tsfm-benchmark checkout}
export HF_HOME=$AIREADI_SCRATCH/hf_cache
mkdir -p "$HF_HOME"

export CHRONOS2_ONLY_WITH_COVARIATES=1
export CHRONOS2_ABLATE_CHANNELS="$ABLATE"
export CHRONOS2_RESULTS_DIR_OVERRIDE="$REPO_ROOT/chronos2_zeroshot_results/full/${ARM_NAME}_v2"
[ -n "$EXTRA_FLAG" ] && export "$EXTRA_FLAG=1"

echo "arm=$ARM_NAME ablate=$ABLATE extra=${EXTRA_FLAG:-none}"

ml load math py-pytorch/2.9.1_py314 py-transformers/4.57.3_py314 py-einops/0.8.1_py314 py-pandas/2.3.3_py314 py-pyarrow/22.0.0_py314
source "$REPO_ROOT/venv-chronos/bin/activate"
python "$REPO_ROOT/code/chronos2/chronos2_zeroshot_eval.py"
