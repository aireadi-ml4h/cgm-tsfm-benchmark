#!/bin/bash
#SBATCH --job-name=chronos2_loo_ablation_ssmcgm_overlap
#SBATCH --partition=normal
#SBATCH --qos=high_p
#SBATCH --time=04:00:00
#SBATCH --mem=32GB
#SBATCH --cpus-per-task=4
#SBATCH -G 1
#SBATCH --output=/scratch/users/%u/logs/%x_%j.out

set -euo pipefail
REPO_ROOT=${AIREADI_ROOT:?set AIREADI_ROOT to your aireadi-collab checkout}
export HF_HOME=$SCRATCH/hf_cache
mkdir -p "$HF_HOME"

ml load math py-pytorch/2.9.1_py314 py-transformers/4.57.3_py314 py-einops/0.8.1_py314 py-pandas/2.3.3_py314 py-pyarrow/22.0.0_py314
source "$REPO_ROOT/venv-chronos/bin/activate"

# Leave-one-out sweep *within* the matched-feature arm
# (slurm/chronos2_zeroshot_eval_ssmcgm_overlap.sbatch), NOT within the
# full/imputed baseline. Distinct from the existing loo/ tree, which drops one
# of the 6 physiological channels relative to full/imputed and is already
# complete: that answers "what does each wearable channel contribute to the
# baseline", whereas this answers "inside the SSM-CGM-matched feature set,
# which of the 5 ADDED channels causes the degradation" -- a question only
# raised by the overlap arm's result (worse than baseline at all five
# horizons, +0.094 to +0.704 mg/dL, p=1.2e-05 at 4h).
#
# Every arm keeps window selection identical (same window_index.csv, same
# model_input/zeroshot_eval/*.csv); CHRONOS2_ABLATE_CHANNELS only filters
# which columns reach predict_df's df/future_df. So all arms score the exact
# same window set as the overlap arm and each other.
#
# Ordered calendar-block-first, deliberately: the singleton arms already point
# hard at tod (alone: +0.786 at 4h, p=1.5e-08, versus the combined arm's
# +0.704 -- i.e. the effects saturate rather than add, so tod is likely the
# whole story). drop_calendar is the decisive test; if it lands back at the
# full/imputed baseline, the remaining three arms are confirmatory rather
# than exploratory. Running them anyway for completeness, but the ordering
# means the answer is available from the first arm's output.
#
# Note on names: drop_tod removes BOTH tod_sin and tod_cos -- the pair has no
# independent switch, it is keyed on tod_sin's name (chronos2_zeroshot_eval.py
# line 190), same convention as oxygen_saturation dropping its own
# _applicable flag. So drop_calendar and drop_tod differ only by
# minute_of_day, which is what makes the two a clean split of "the calendar
# block" from "the redundant third encoding of it".
ARMS=(
  "drop_calendar:tod_sin,minute_of_day"
  "drop_tod:tod_sin"
  "drop_minute_of_day:minute_of_day"
  "drop_sleep_stage:sleep_stage"
  "drop_predmeal_prob:predmeal_prob"
  "drop_sleep_regularity_index:sleep_regularity_index"
)

for arm in "${ARMS[@]}"; do
  name="${arm%%:*}"
  channels="${arm##*:}"
  echo "=== ${name} (ablating: ${channels}) ==="

  # Identical to the overlap arm's env, plus CHRONOS2_ABLATE_CHANNELS. All
  # five INCLUDE_* flags stay on in every arm -- the ablation is expressed
  # only through ABLATE_CHANNELS, so an arm never differs from the overlap
  # reference by more than the one thing being tested.
  CHRONOS2_ONLY_WITH_COVARIATES=1 \
  CHRONOS2_INCLUDE_SLEEP_STAGE=1 \
  CHRONOS2_INCLUDE_PREDMEAL_PROB=1 \
  CHRONOS2_INCLUDE_SLEEP_REGULARITY=1 \
  CHRONOS2_INCLUDE_TOD=1 \
  CHRONOS2_INCLUDE_MINUTE_OF_DAY=1 \
  CHRONOS2_ABLATE_CHANNELS="${channels}" \
  CHRONOS2_RESULTS_DIR_OVERRIDE="$REPO_ROOT/chronos2_zeroshot_results/loo_ssmcgm_overlap/${name}" \
    python "$REPO_ROOT/code/chronos2/chronos2_zeroshot_eval.py"
done

echo "=== all ${#ARMS[@]} arms done ==="
