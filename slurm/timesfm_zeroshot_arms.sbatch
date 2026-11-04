#!/bin/bash
# TimesFM 2.5 zero-shot evaluation arms.
#
# Mirrors chronos2_zeroshot_arms.sbatch. All eleven arms run the same entry
# point (code/timesfm/timesfm_zeroshot_eval.py) with the same modules, venv, and
# resources; they differ only in which covariate channels are supplied and
# where results are written. Consolidated into one array job from eleven
# near-identical scripts -- the per-arm config is the table below.
#
# Env-var semantics (read by timesfm_zeroshot_eval.py):
#   TIMESFM_ONLY_WITH_COVARIATES  skip the glucose-only condition
#   TIMESFM_INCLUDE_<X>           opt a derived channel INTO the feature set
#   TIMESFM_ABLATE_CHANNELS       comma-separated channels REMOVED from it
#   TIMESFM_RESULTS_DIR_OVERRIDE  where predictions/metrics are written
#
# Unlike Chronos-2, there is no MINUTE_OF_DAY channel in the SSM-CGM-overlap
# arm, and the heart-rate isolation lives here rather than in isolated_arms/.
#
# Walltime is set to the longest arm (12 h). The original per-arm walltimes
# were 12 h except: ssmcgm_overlap and heartrate_only 4 h, isolated_channel
# 40 min.
#
# Submit all arms:   sbatch timesfm_zeroshot_arms.sbatch
# Submit one arm:    sbatch --array=6 timesfm_zeroshot_arms.sbatch
#
#SBATCH --job-name=tfm_zeroshot
#SBATCH --partition=gpu
#SBATCH --time=12:00:00
#SBATCH --mem=32GB
#SBATCH --cpus-per-task=4
#SBATCH -G 1
#SBATCH --array=0-10
#SBATCH --output=logs/%x_%A_%a.out
set -euo pipefail

# arm_name | extra env (space-separated KEY=VAL) | results subdir
# "-" means leave at the script's own default.
ARMS=(
  # Baseline: the only arm that also scores the glucose-only condition.
  "base|-|-"

  # Single derived channel added to the six base wearable channels.
  "with_predmeal_prob|TIMESFM_ONLY_WITH_COVARIATES=1 TIMESFM_INCLUDE_PREDMEAL_PROB=1|full/with_predmeal_prob"
  "with_sleep_stage|TIMESFM_ONLY_WITH_COVARIATES=1 TIMESFM_INCLUDE_SLEEP_STAGE=1|full/with_sleep_stage"
  "with_sleep_regularity|TIMESFM_ONLY_WITH_COVARIATES=1 TIMESFM_INCLUDE_SLEEP_REGULARITY=1|full/with_sleep_regularity"
  "with_tod|TIMESFM_ONLY_WITH_COVARIATES=1 TIMESFM_INCLUDE_TOD=1|full/with_tod"
  "with_static|TIMESFM_ONLY_WITH_COVARIATES=1 TIMESFM_INCLUDE_STATIC=1|full/with_static"

  # The subset of inputs shared with SSM-CGM.
  "with_ssmcgm_overlap|TIMESFM_ONLY_WITH_COVARIATES=1 TIMESFM_INCLUDE_SLEEP_STAGE=1 TIMESFM_INCLUDE_PREDMEAL_PROB=1 TIMESFM_INCLUDE_SLEEP_REGULARITY=1 TIMESFM_INCLUDE_TOD=1|full/with_ssmcgm_overlap"

  # Heart-rate isolation (the other single-channel arms are in isolated_arms/).
  "heartrate_only|TIMESFM_ONLY_WITH_COVARIATES=1 TIMESFM_ABLATE_CHANNELS=stress,respiratory_rate,oxygen_saturation,physical_activity_steps,physical_activity_calorie_rate|full/heartrate_only"

  # The 2x2 SpO2 analysis: drop the applicability indicator, or the reading.
  "loo_drop_oxysat_applicable|TIMESFM_ONLY_WITH_COVARIATES=1 TIMESFM_ABLATE_CHANNELS=oxygen_saturation_applicable|loo/drop_oxygen_saturation_applicable"
  "loo_drop_oxysat_reading_only|TIMESFM_ONLY_WITH_COVARIATES=1 TIMESFM_ABLATE_CHANNELS=oxygen_saturation_reading_only|loo/drop_oxygen_saturation_reading_only"

  # Driver for the isolated-channel sweep; writes to the script's default dir.
  "isolated_channel|TIMESFM_ONLY_WITH_COVARIATES=1|-"
)

IFS='|' read -r ARM_NAME ARM_ENV RESULTS_SUBDIR <<< "${ARMS[${SLURM_ARRAY_TASK_ID:-0}]}"

REPO_ROOT=${AIREADI_ROOT:?set AIREADI_ROOT to your cgm-tsfm-benchmark checkout}
export HF_HOME=$AIREADI_SCRATCH/hf_cache
mkdir -p "$HF_HOME"

if [ "$ARM_ENV" != "-" ]; then
  for kv in $ARM_ENV; do export "$kv"; done
fi
[ "$RESULTS_SUBDIR" != "-" ] && export TIMESFM_RESULTS_DIR_OVERRIDE="$REPO_ROOT/timesfm_zeroshot_results/$RESULTS_SUBDIR"

echo "arm=$ARM_NAME env=${ARM_ENV} results=${RESULTS_SUBDIR}"

ml load math py-pytorch/2.9.1_py314 py-huggingface-hub/0.36.0_py314 py-safetensors/0.7.0_py314 py-numpy/2.3.5_py314 py-pandas/2.3.3_py314 py-pyarrow/22.0.0_py314
ml load math py-jax/0.8.1_py314 py-scikit-learn/1.8.0_py314
source "$REPO_ROOT/venv-timesfm/bin/activate"
python "$REPO_ROOT/code/timesfm/timesfm_zeroshot_eval.py"
