#!/bin/bash
#SBATCH --job-name=chronos2_pers_test
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

# PHASE 1, TEST SPLIT: the proposal's second research question.
# NOT SUBMITTED WITHOUT SIGN-OFF -- this is the actual experiment, and the
# config it uses comes from slurm/chronos2_personalization_config.sh, which
# still carries a PENDING banner until sweep 5 lands. Check that first.
#
# EIGHT ARRAY TASKS = 4 arms x 2 fit lengths
# ==========================================
#   p1d, p3d, p7d        the participant's own 1/3/7 days of history
#   p7d_shuffled         NEGATIVE CONTROL: a different, deranged-assignment
#                        participant's 7 days. NOT OPTIONAL. Any gain over
#                        zero-shot is otherwise equally well explained by the
#                        model adapting to AI-READI / Dexcom-G6 / 5-minute-grid
#                        data in general as by adapting to the PERSON. Read
#                        this arm's row in the report BEFORE the others.
#
# One task per (arm, fit length) rather than sharding participants. At the
# measured 33.8 s/fit (fit@6) and ~15 s/fit (fit@24) over ~263 participants
# that is ~2.5 h and ~1.1 h per task, so no task needs splitting and each one
# owns its whole output directory -- which keeps the participant-granular
# resume logic simple (one participants_done_shard0.csv per directory, no
# cross-shard merge).
#
# Walltime is 8h against a ~2.5h worst case, deliberately generous: re-running
# a timed-out fine-tuning job costs more than the queue time saved by a tight
# estimate, and a stochastic fit cannot be topped up mid-participant.
#
# WHY EVERY ARM IS SCORED ON ONE SHARED WINDOW SET
# ================================================
# AI-READI is a ~10-day wear, so 7 days of prior history only exists for
# windows in the last ~3 days of a record. Per-arm eligibility would give the
# three history lengths three different, progressively later, progressively
# smaller window sets, and any 1d-vs-7d difference would then mix "amount of
# history" with "which windows and which people got scored". So the bar is
# fixed at 7 days for EVERY arm (E: 1,024 windows / 260 participants at 1h).
# The user reaffirmed 2026-09-03 that keeping 1/3/7-day comparable is the
# priority, over the larger-sample alternative.
#
# RESUME: participant-granular, and re-running this job is safe. Unlike the
# zero-shot script, predictions cannot be topped up per window -- fine-tuning
# is stochastic, so a participant interrupted mid-horizon is discarded and
# redone from scratch. Handled inside chronos2_personalization.py.

# PERS_BAR: default 7 is the PRIMARY family, where all arms share one window set
# so 1d/3d/7d stay mutually comparable. Override to produce a glucose-only
# REFERENCE arm inside the parallel (lower-bar) family, e.g.
#   sbatch --array=4,5 --export=ALL,PERS_BAR=3 slurm/chronos2_personalization_test_arms.sbatch
# which runs p3d at bar=3.
#
# WHY THAT REFERENCE ARM IS NEEDED and is not just symmetry: the parallel
# covariate family's headline question is "do wearables add anything ON TOP OF
# a personalized model" -- covariate arm1 vs the glucose-only arm at the same k.
# Its comparator has to be a PERSONALIZED glucose-only arm; a zero-shot one
# answers a different question. And it has to live on the same window set: the
# bar=3 covariate arms are scored on E_cov_bar3d, which contains windows absent
# from the bar=7 E, so comparing against the bar=7 p3d arm would collapse onto
# the intersection and throw away exactly the power the lower bar was for.
# Without this arm the parallel family can only answer "personalization helps,
# given these channels", not the channels question itself.

set -euo pipefail
REPO_ROOT=${AIREADI_ROOT:?set AIREADI_ROOT to your cgm-tsfm-benchmark checkout}
PERS_BAR="${PERS_BAR:-7}"
export HF_HOME=$AIREADI_SCRATCH/hf_cache
mkdir -p "$HF_HOME"

ml load math py-pytorch/2.9.1_py314 py-transformers/4.57.3_py314 py-einops/0.8.1_py314 py-pandas/2.3.3_py314 py-pyarrow/22.0.0_py314
source "$REPO_ROOT/venv-chronos/bin/activate"
source "$REPO_ROOT/slurm/chronos2_personalization_config.sh"

# arm : history_days : shuffle_donor : fit
TASKS=(
  "p7d:7d:0:fit6"
  "p7d:7d:0:fit24"
  "p7d_shuffled:7d:1:fit6"
  "p7d_shuffled:7d:1:fit24"
  "p3d:3d:0:fit6"
  "p3d:3d:0:fit24"
  "p1d:1d:0:fit6"
  "p1d:1d:0:fit24"
)
# Ordered so that p7d and its negative control run FIRST. If the array is
# cancelled partway, what exists is the arm the effect is expected in plus the
# control that makes it interpretable -- rather than three history lengths and
# no control, which would be uninterpretable on its own.

IFS=':' read -r arm k shuffle fit <<< "${TASKS[$SLURM_ARRAY_TASK_ID]}"
# lr and num_steps are PER-k (sweep 6 -- the k=7d config does not transfer to
# shorter histories), so they come from the one key, never set by hand here.
pers_set_fit "$k" "$fit"

if [ "$PERS_BAR" != "7" ] && [ "${k%d}" -gt "$PERS_BAR" ]; then
  echo "task $SLURM_ARRAY_TASK_ID: k=$k exceeds bar=${PERS_BAR}d, nothing to run" >&2
  exit 0
fi

# Results root carries the bar, so the two families cannot append into one
# another's prediction files (predictions are written with mode="a", so a
# collision would silently interleave two window sets in one file).
BAR_SUFFIX=""
[ "$PERS_BAR" != "7" ] && BAR_SUFFIX="_bar${PERS_BAR}d"

export CHRONOS2_PERS_CONDITION=glucose_only
export CHRONOS2_PERS_HISTORY_BAR_DAYS="$PERS_BAR"
export CHRONOS2_PERS_HISTORY_DAYS="${k%d}"
export CHRONOS2_PERS_SHUFFLE_DONOR="$shuffle"
export CHRONOS2_PERS_RESULTS_DIR="$REPO_ROOT/chronos2_personalization_results/test${BAR_SUFFIX}/$arm/$fit"

echo "=== arm=$arm k=$k bar=${PERS_BAR}d shuffled=$shuffle $fit"
echo "    lr=$CHRONOS2_PERS_LEARNING_RATE steps=$CHRONOS2_PERS_NUM_STEPS"
echo "    fit_prediction_length=$CHRONOS2_PERS_FIT_PREDICTION_LENGTH"
echo "    -> $CHRONOS2_PERS_RESULTS_DIR"

python "$REPO_ROOT/code/chronos2/chronos2_personalization.py"
