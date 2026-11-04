#!/bin/bash
#SBATCH --job-name=chronos2_pers_cov
#SBATCH --partition=normal
# QOS: normal, NOT high_p. high_p is a priority boost (priority 10000 vs 1) but
# carries MaxJobsPU=4, so it caps this array at 4 concurrent tasks; normal has no
# per-user job cap (512 CPU/user, ~128 tasks at 4 cores each). the owner partition is the
# group's own partition with a handful of active users, so the boost buys little
# and the cap costs a lot. Every other sbatch in this project uses high_p because
# they are single jobs where the cap never binds -- do not copy that here.
#SBATCH --qos=normal
#SBATCH --time=08:00:00
#SBATCH --mem=64GB
#SBATCH --cpus-per-task=4
#SBATCH -G 1
#SBATCH --array=0-7
#SBATCH --output=logs/%x_%A_%a.out

# PHASE 2, TEST SPLIT: the four nested covariate arms.
# NOT SUBMITTED WITHOUT SIGN-OFF. Two things to check first:
#   1. slurm/chronos2_personalization_config.sh no longer carries its PENDING
#      banner (i.e. sweep 5 has landed and the lr is bracketed).
#   2. You have read the power section below. This experiment is expected to
#      return a null that means nothing.
#
# READ THIS BEFORE SPENDING THE GPU TIME
# ======================================
# Measured MDE on E_cov (code/analysis/chronos2_personalization_power_calibration.py,
# chronos2_personalization_results/power_calibration_cov.csv): the smallest
# covariate increment the primary test can detect 80% of the time is
# 0.13-0.35 mg/dL on 30min/1h/2h. Every covariate effect this project has ever
# resolved on those horizons is 0.002-0.26 mg/dL, mostly under 0.1. And those
# MDEs are LOWER bounds -- they come from zero-shot pairs, and fine-tuning adds
# per-participant variance a zero-shot pair does not have.
#
# So this is powered only for the case where personalization multiplies a
# covariate's effect by roughly 3-10x. That IS the pre-registered hypothesis
# (a zero-shot model may not know how to use THIS person's heart rate while a
# personalized one could learn to), and it is a legitimate reason to run. But a
# null must be reported as "not detectable at n=194, MDE 0.13-0.35 mg/dL",
# never as "covariates do not help once you personalize". The report script
# labels such rows "null, UNDERPOWERED" for exactly this reason.
#
# EIGHT ARRAY TASKS = 4 arms x 2 fit lengths
# ==========================================
#   arm1  5phys                                heart_rate, stress,
#                                              respiratory_rate, steps,
#                                              calorie_rate
#   arm2  + predmeal_prob                      isolates the meal signal
#   arm3  + oxygen_saturation (+ its flag)     isolates SpO2
#   arm4  + sleep_stage                        isolates the sleep-stage channel
#
# Nested so each increment is isolated against its immediate parent:
# arm2-arm1, arm3-arm2, arm4-arm2. Deliberately NOT a sweep -- four
# pre-specified arms with three stated attributions, because there is no
# multiple-comparison correction anywhere in this project and 409 windows will
# not support an exploratory search.
#
# Channels come from chronos2_zeroshot_eval's own CHRONOS2_INCLUDE_* /
# CHRONOS2_ABLATE_CHANNELS vars, not from a second list in this file, so each
# arm is provably the same channel set as the zero-shot arm it is compared
# against. That is also what made the mode=none plumbing check meaningful
# (job 41931340: all four arms reproduce their zero-shot twin to 5e-5 mg/dL).
#
# TIMING: 18.3 s/fit measured at fit@24/s200/batch 224 -- 1.9x the glucose-only
# 9.6 s/fit, not 7x, because the batch is variate-scaled rather than the model
# being bigger. ~232 participants -> ~1.2 h per fit@24 task, more for fit@6
# (higher lr, same steps, but fit@6 measured slower in the val sweeps).
# 8h walltime against that is deliberately generous.
#
# ARM 3 IS EXPECTED TO DO NOTHING, for two independent reasons worth stating
# up front so the result is not over-read: oxygen_saturation was measured
# HARMFUL zero-shot (the most robust covariate effect in the whole sweep,
# cluster p=0.0008 at 30min), AND it is only ~14% present in the 7-day history
# it would have to be learned from, with 35 of 232 participants having none at
# all (model_input/personalization/history_coverage_cov_test.csv).

# PERS_BAR / PERS_K: default 7 / 7d is the PRIMARY family (job 42049992).
# Override to run the PARALLEL family on a lower eligibility bar, e.g.
#   sbatch --export=ALL,PERS_BAR=3,PERS_K=3d slurm/chronos2_personalization_cov_arms.sbatch
# One script rather than a near-copy, so the arm definitions, the channel
# wiring and the batch-size scaling cannot drift between the two families --
# the whole point is that only the window set differs.
#
# THE PARALLEL FAMILY, AND WHAT IT DOES AND DOES NOT BUY
# =====================================================
# The bar-7 set is 409 windows / 194 participants at 1h, where the MDE is
# 0.13-0.35 mg/dL and every covariate effect this project has resolved is
# 0.002-0.26 -- so the primary family is expected to return an uninformative
# null. Lowering the bar admits every window >= B days into a record:
#
#   bar   1h windows  participants   MDE 30min/1h/2h    MDE ratio vs FULL test set
#   7d           409           194   0.12-0.17 / 0.14-0.33 / 0.32-0.36   1.2-2.9
#   3d         1,192           257   0.09-0.10 / 0.10-0.25 / 0.15-0.23   ~1.18
#   1d         1,588           268   0.07-0.09 / 0.08-0.23 / 0.11-0.20   0.89-1.03
#
# At bar=3 the design's power cost is nearly gone: a drop-SpO2-sized effect
# (-0.085 at 30min, cluster p=0.0008 on the full set) sits above the 0.09-0.10
# threshold instead of below it.
#
# WHAT IS GIVEN UP: a bar=B set has different windows AND different
# participants from a bar=B' set, so arms across bars are not comparable to
# each other. Each bar is a self-contained 2-arm experiment against zero-shot.
# The 1/3/7-day dose-response comparison lives ONLY in the bar-7 primary
# family, which is why this is an addition and not a replacement.
#
# WHY bar=3/k=3d RATHER THAN bar=1/k=1d, which has marginally more power: the
# bar must be >= k, so a lower bar forces a shorter history, and at k=1d
# personalization is barely working (sweep 6: ~neutral at 30min and 1h, only 2h
# helps) -- a covariate increment would have almost nothing to sit on top of.
# At k=3d personalization is real (-0.44 at 1h, -0.96 at 2h) and 3 days of
# wearable coverage is near-complete.
#
# The per-k configs from sweep 6 apply unchanged: k is a real hyperparameter
# input here, and pers_set_fit "$PERS_K" picks the config tuned at that k.

set -euo pipefail
REPO_ROOT=${AIREADI_ROOT:?set AIREADI_ROOT to your aireadi-collab checkout}
PERS_BAR="${PERS_BAR:-7}"
PERS_K="${PERS_K:-7d}"
export HF_HOME=$AIREADI_SCRATCH/hf_cache
mkdir -p "$HF_HOME"

ml load math py-pytorch/2.9.1_py314 py-transformers/4.57.3_py314 py-einops/0.8.1_py314 py-pandas/2.3.3_py314 py-pyarrow/22.0.0_py314
source "$REPO_ROOT/venv-chronos/bin/activate"
source "$REPO_ROOT/slurm/chronos2_personalization_config.sh"

# arm dir : ablate : include_predmeal : include_sleep_stage : fit
TASKS=(
  "cov_arm1_5phys:oxygen_saturation:0:0:fit6"
  "cov_arm1_5phys:oxygen_saturation:0:0:fit24"
  "cov_arm2_5phys_meal:oxygen_saturation:1:0:fit6"
  "cov_arm2_5phys_meal:oxygen_saturation:1:0:fit24"
  "cov_arm4_5phys_meal_sleep:oxygen_saturation:1:1:fit6"
  "cov_arm4_5phys_meal_sleep:oxygen_saturation:1:1:fit24"
  "cov_arm3_5phys_meal_oxysat::1:0:fit6"
  "cov_arm3_5phys_meal_oxysat::1:0:fit24"
)
# Ordered arm1, arm2, arm4, arm3. arm1 and arm2 first because they are the
# all-numeric pair -- no categorical channel, hence no target-encoding caveat,
# hence the arm2-arm1 predmeal_prob attribution is the one clean result this
# phase can produce. arm3 last because it is the one expected to do nothing
# (see the banner above), so it is the cheapest thing to lose if the array is
# cancelled.

# `stage`, not `sleep`. the cluster's submit filter scans the script for sleeper
# jobs and rejects any bare `sleep` token ("ERROR: sleeper job detected",
# submission refused outright). A loop variable named `sleep` is enough to trip
# it. The longer tokens (sleep_stage, CHRONOS2_INCLUDE_SLEEP_STAGE) are fine --
# it is the standalone word that matches.
IFS=':' read -r arm ablate meal stage fit <<< "${TASKS[$SLURM_ARRAY_TASK_ID]}"
# The config tuned at THIS k (sweep 6). In the primary family that is the k=7d
# config -- the same one phase 1's p7d arm uses, which is what makes
# "arm1 vs p7d" (does adding wearables help a personalized model?) a
# channels-only comparison. In the parallel family it is the k=3d config, and
# the corresponding glucose-only reference is phase 1's p3d arm.
pers_set_fit "$PERS_K" "$fit"

# Results root carries the bar, so the two families can never append into one
# another's prediction files -- predictions are written with mode="a", so a
# collision would silently interleave two window sets in one file.
BAR_SUFFIX=""
[ "$PERS_BAR" != "7" ] && BAR_SUFFIX="_bar${PERS_BAR}d"

export CHRONOS2_PERS_CONDITION=with_covariates
export CHRONOS2_PERS_HISTORY_BAR_DAYS="$PERS_BAR"
export CHRONOS2_PERS_HISTORY_DAYS="${PERS_K%d}"
export CHRONOS2_ABLATE_CHANNELS="$ablate"
export CHRONOS2_INCLUDE_PREDMEAL_PROB="$meal"
export CHRONOS2_INCLUDE_SLEEP_STAGE="$stage"
export CHRONOS2_PERS_RESULTS_DIR="$REPO_ROOT/chronos2_personalization_results/test${BAR_SUFFIX}/$arm/$fit"

echo "=== arm=$arm $fit  bar=${PERS_BAR}d k=$PERS_K"
echo "    ablate='$ablate' predmeal=$meal sleep_stage=$stage"
echo "    lr=$CHRONOS2_PERS_LEARNING_RATE steps=$CHRONOS2_PERS_NUM_STEPS"
echo "    fit_prediction_length=$CHRONOS2_PERS_FIT_PREDICTION_LENGTH"
echo "    -> $CHRONOS2_PERS_RESULTS_DIR"

python "$REPO_ROOT/code/chronos2/chronos2_personalization.py"
