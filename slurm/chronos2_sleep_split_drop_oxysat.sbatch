#!/bin/bash
#SBATCH --job-name=chronos2_sleep_split_drop_oxysat
#SBATCH --partition=normal
#SBATCH --qos=high_p
#SBATCH --time=02:00:00
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

# Splits the two sleep channels, which full/with_sleep_meal_drop_oxysat added
# together. That arm established the PAIR is harmful on the best channel set
# (+0.085/+0.129/+0.163/+0.234/+0.214 mg/dL vs
# full/with_predmeal_prob_drop_oxysat, significant on both the naive and the
# cluster-by-participant test at all five horizons) but cannot attribute it.
#
# Each arm below adds exactly ONE sleep channel to
# full/with_predmeal_prob_drop_oxysat (the best config found), so each is one
# controlled step from that reference and one controlled step from the joint
# arm.
#
# Prediction, stated in advance so the result can falsify it rather than be
# rationalized after: sleep_regularity_index should be largely INERT and
# sleep_stage should carry essentially all the harm. Reason -- SRI is one
# value per calendar day broadcast to every bin of that day
# (build_participant_grid.compute_sleep_regularity_index), so for any 12h
# context window lying entirely within one UTC calendar day (start hour in
# [0,12], roughly half of all windows) it is EXACTLY CONSTANT, and a constant
# covariate is zeroed exactly by instance normalization -- the same mechanism
# that erases the statics (chronos2/model.py:408 + chronos_bolt.py:113,
# scale==0 -> eps=1e-5 -> (c-c)/1e-5 == 0). For the remaining windows it is a
# two-level step, non-degenerate but very low variance. sleep_stage, by
# contrast, is a real per-bin categorical that varies throughout the window.
#
# If SRI instead shows real harm, that is the more interesting outcome: it
# would mean a near-constant channel damages the forecast through occupying
# an extra attention slot rather than through its values, which would also
# weaken the current reading of full/with_static (whose ~0.01-0.09 mg/dL
# wobble is presently attributed to that same route being negligible).
ARMS=(
  "with_sleep_stage_predmeal_drop_oxysat:CHRONOS2_INCLUDE_SLEEP_STAGE"
  "with_sleep_reg_predmeal_drop_oxysat:CHRONOS2_INCLUDE_SLEEP_REGULARITY"
)

for arm in "${ARMS[@]}"; do
  name="${arm%%:*}"
  flag="${arm##*:}"
  echo "=== ${name} (extra flag: ${flag}) ==="

  # `env` rather than a bare NAME=VALUE prefix: the flag name comes from a
  # variable, and bash only treats a prefix word as an assignment when it
  # looks like one at parse time -- "${flag}"=1 would be passed as an
  # ordinary argument instead, silently leaving the flag unset and making
  # both arms identical.
  env CHRONOS2_ONLY_WITH_COVARIATES=1 \
      CHRONOS2_INCLUDE_PREDMEAL_PROB=1 \
      "${flag}=1" \
      CHRONOS2_ABLATE_CHANNELS=oxygen_saturation \
      CHRONOS2_RESULTS_DIR_OVERRIDE="$REPO_ROOT/chronos2_zeroshot_results/full/${name}" \
      python "$REPO_ROOT/code/chronos2/chronos2_zeroshot_eval.py"
done

echo "=== both arms done ==="
