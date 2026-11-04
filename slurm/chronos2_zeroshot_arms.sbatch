#!/bin/bash
# Chronos-2 zero-shot evaluation arms.
#
# All thirteen arms run the same entry point (code/chronos2/chronos2_zeroshot_eval.py)
# with the same modules, venv, and resources; they differ only in which
# covariate channels are supplied and where results are written. Consolidated
# into one array job from thirteen near-identical scripts -- the per-arm config
# is the table below.
#
# Env-var semantics (read by chronos2_zeroshot_eval.py):
#   CHRONOS2_ONLY_WITH_COVARIATES  skip the glucose-only condition
#   CHRONOS2_INCLUDE_<X>           opt a derived channel INTO the feature set
#   CHRONOS2_ABLATE_CHANNELS       comma-separated channels REMOVED from it
#   CHRONOS2_RESULTS_DIR_OVERRIDE  where predictions/metrics are written
#   CHRONOS2_EVAL_DIR_OVERRIDE     which exported window set to read
#
# Walltime is set to the longest arm (12 h). The original per-arm walltimes
# were 12 h except: isolated_channel 40 min, and the two drop_oxysat arms 2 h.
#
# Submit all arms:   sbatch chronos2_zeroshot_arms.sbatch
# Submit one arm:    sbatch --array=6 chronos2_zeroshot_arms.sbatch
#
#SBATCH --job-name=c2_zeroshot
#SBATCH --partition=gpu
#SBATCH --time=12:00:00
#SBATCH --mem=32GB
#SBATCH --cpus-per-task=4
#SBATCH -G 1
#SBATCH --array=0-12
#SBATCH --output=logs/%x_%A_%a.out
set -euo pipefail

# arm_name | extra env (space-separated KEY=VAL) | results subdir | eval-set subdir
# "-" means leave at the script's own default.
ARMS=(
  # Baseline: the only arm that also scores the glucose-only condition.
  "base|-|-|-"

  # Single derived channel added to the six base wearable channels.
  "with_predmeal_prob|CHRONOS2_ONLY_WITH_COVARIATES=1 CHRONOS2_INCLUDE_PREDMEAL_PROB=1|full/with_predmeal_prob|-"
  "with_sleep_stage|CHRONOS2_ONLY_WITH_COVARIATES=1 CHRONOS2_INCLUDE_SLEEP_STAGE=1|full/with_sleep_stage|-"
  "with_sleep_regularity|CHRONOS2_ONLY_WITH_COVARIATES=1 CHRONOS2_INCLUDE_SLEEP_REGULARITY=1|full/with_sleep_regularity|-"
  "with_tod|CHRONOS2_ONLY_WITH_COVARIATES=1 CHRONOS2_INCLUDE_TOD=1|full/with_tod|-"
  "with_static|CHRONOS2_ONLY_WITH_COVARIATES=1 CHRONOS2_INCLUDE_STATIC=1|full/with_static|-"

  # The subset of inputs shared with SSM-CGM.
  "with_ssmcgm_overlap|CHRONOS2_ONLY_WITH_COVARIATES=1 CHRONOS2_INCLUDE_SLEEP_STAGE=1 CHRONOS2_INCLUDE_PREDMEAL_PROB=1 CHRONOS2_INCLUDE_SLEEP_REGULARITY=1 CHRONOS2_INCLUDE_TOD=1 CHRONOS2_INCLUDE_MINUTE_OF_DAY=1|full/with_ssmcgm_overlap|-"

  # Exploratory combination arms: derived channels paired with SpO2 removal.
  "with_predmeal_prob_drop_oxysat|CHRONOS2_ONLY_WITH_COVARIATES=1 CHRONOS2_INCLUDE_PREDMEAL_PROB=1 CHRONOS2_ABLATE_CHANNELS=oxygen_saturation|full/with_predmeal_prob_drop_oxysat|-"
  "with_sleep_meal_drop_oxysat|CHRONOS2_ONLY_WITH_COVARIATES=1 CHRONOS2_INCLUDE_SLEEP_STAGE=1 CHRONOS2_INCLUDE_SLEEP_REGULARITY=1 CHRONOS2_INCLUDE_PREDMEAL_PROB=1 CHRONOS2_ABLATE_CHANNELS=oxygen_saturation|full/with_sleep_meal_drop_oxysat|-"

  # The 2x2 SpO2 analysis: drop the applicability indicator, or the reading.
  "loo_drop_oxysat_applicable|CHRONOS2_ONLY_WITH_COVARIATES=1 CHRONOS2_ABLATE_CHANNELS=oxygen_saturation_applicable|loo/drop_oxygen_saturation_applicable|-"
  "loo_drop_oxysat_reading_only|CHRONOS2_ONLY_WITH_COVARIATES=1 CHRONOS2_ABLATE_CHANNELS=oxygen_saturation_reading_only|loo/drop_oxygen_saturation_reading_only|-"

  # Driver for the isolated-channel sweep; writes to the script's default dir.
  "isolated_channel|CHRONOS2_ONLY_WITH_COVARIATES=1|-|-"

  # Robustness: wearable forward/backward imputation removed upstream.
  "noimpute|-|full/noimpute|zeroshot_eval_noimpute"
)

IFS='|' read -r ARM_NAME ARM_ENV RESULTS_SUBDIR EVAL_SUBDIR <<< "${ARMS[${SLURM_ARRAY_TASK_ID:-0}]}"

REPO_ROOT=${AIREADI_ROOT:?set AIREADI_ROOT to your cgm-tsfm-benchmark checkout}
export HF_HOME=$AIREADI_SCRATCH/hf_cache
mkdir -p "$HF_HOME"

if [ "$ARM_ENV" != "-" ]; then
  for kv in $ARM_ENV; do export "$kv"; done
fi
[ "$RESULTS_SUBDIR" != "-" ] && export CHRONOS2_RESULTS_DIR_OVERRIDE="$REPO_ROOT/chronos2_zeroshot_results/$RESULTS_SUBDIR"
[ "$EVAL_SUBDIR" != "-" ] && export CHRONOS2_EVAL_DIR_OVERRIDE="$REPO_ROOT/model_input/$EVAL_SUBDIR"

echo "arm=$ARM_NAME env=${ARM_ENV} results=${RESULTS_SUBDIR} eval=${EVAL_SUBDIR}"

ml load math py-pytorch/2.9.1_py314 py-transformers/4.57.3_py314 py-einops/0.8.1_py314 py-pandas/2.3.3_py314 py-pyarrow/22.0.0_py314
source "$REPO_ROOT/venv-chronos/bin/activate"
python "$REPO_ROOT/code/chronos2/chronos2_zeroshot_eval.py"
