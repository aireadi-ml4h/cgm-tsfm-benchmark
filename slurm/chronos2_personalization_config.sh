#!/bin/bash
# THE FROZEN FINE-TUNING CONFIG. Sourced by every test-split arm, phase 1 and
# phase 2, so that all arms provably share one config and the only thing
# differing between them is what each arm is meant to vary (history length in
# phase 1, channels in phase 2).
#
# ONE FILE, NOT COPIES IN EACH SBATCH. If these numbers were pasted into each
# arm's script, an edit to one would silently break the "identical config"
# property the whole comparison rests on -- and nothing would error.
#
# PROVENANCE: val split only, never test. Sweeps 1-5, all on 80 val
# participants at k=7d. See code/CHRONOS2_PERSONALIZATION.md for the full
# trail; the short version of what each axis is bracketed by:
#
#   finetune_mode  LoRA. Beats full fine-tuning at every fit length and every
#                  primary horizon (sweep 2). Settled.
#   fit length     horizon-matched. fit_prediction_length is the LOSS horizon,
#                  not the context, and it is the dominant hyperparameter
#                  (sweep 2): at fit@48 the 6 bins of a 30-minute forecast are
#                  1/8 of the gradient, which was the entire original 30min
#                  regression.
#   num_steps      PER-k (see the table below). 200 at k=7d, bracketed on both
#                  sides at both fit lengths -- s400 and s800 both degrade, the
#                  s800 runs badly (sweep 4). 76 at k=3d and 25/50 at k=1d,
#                  because the k=7d value does not transfer (sweep 6).
#   learning_rate  BRACKETED ON BOTH SIDES at both fit lengths by sweep 5
#                  (job 41984052, 2026-09-03). Both ladders turn over, so
#                  neither value below is on a grid edge:
#                    fit@6  s200, 30min dMAE: -0.064 / -0.225 / -0.389 /
#                           +0.503 / +1.261  at lr 5e-5/1e-4/2e-4/4e-4/8e-4
#                           -> optimum 2e-4, and it is a real peak
#                    fit@24 s200:  2h dMAE -1.183 / -1.450 / -1.085 / +0.225
#                           and 1h -0.372 / -0.441 / +0.108 / +1.287
#                           at lr 5e-5/1e-4/2e-4/4e-4  -> optimum 1e-4 on both
#                  Sweep 5 CONFIRMED sweep 4's winners rather than moving them,
#                  which is the outcome that makes them safe to freeze.
#   batch_size     32 WINDOWS per step. chronos2_personalization.py multiplies
#                  by the variate count before passing it to fit(), because
#                  fit()'s own batch_size counts variates -- so phase 1 and
#                  phase 2 see the same number of training windows per step
#                  rather than phase 2 seeing ~1/7 as many.
#
# FROZEN 2026-09-03 after sweep 5. Every axis is now bracketed on both sides:
# mode (sweep 2), fit length (sweep 2), steps (sweep 4), learning rate
# (sweep 5). Nothing here sits on a grid edge, which is the precondition
# code/CHRONOS2_PERSONALIZATION.md sets for touching the test split.
#
# STILL OUTSTANDING BEFORE THE 1d AND 3d ARMS SPECIFICALLY: this config was
# tuned at k=7d, where a participant has ~1,825 distinct training windows. At
# k=1d they have 97. The k=1d/3d val transfer check
# (slurm/chronos2_personalization_val_ktransfer.sbatch) measures whether the
# config carries over. The k=7d arm and the negative control do not depend on
# it.

set -euo pipefail

# --- shared by every arm -------------------------------------------------
export CHRONOS2_PERS_SPLIT=test
export CHRONOS2_PERS_FINETUNE_MODE=lora
export CHRONOS2_PERS_BATCH_SIZE=32   # WINDOWS per step; see note above
# NOTE: num_steps and learning_rate are NOT set here -- they are per
# (history length, fit length) and are set together by pers_set_fit below.
# Setting a default here would let an arm that forgets to call pers_set_fit
# run silently on the wrong config.
export CHRONOS2_PERS_LORA_R=8
export CHRONOS2_PERS_SEED=42

# --- per (history length, fit length) ------------------------------------
# fit@6 scores 30min; fit@24 scores 1h and 2h (and 3h/4h, which are therefore
# NOT horizon-matched for their own horizon -- a disclosed limitation, printed
# in the report's own 3h/4h block header).
#
# num_steps IS PER-k, AND THAT IS A DESIGN CHANGE NEEDING SIGN-OFF.
# ================================================================
# The k=7d config does not transfer: run unchanged at k=1d it is SIGNIFICANTLY
# WORSE than zero-shot (30min +2.38 mg/dL, cluster p<0.0001; job 41989805).
# The mechanism is passes over the pool of distinct training windows, which is
# set by the history length -- 200 steps x 32 windows draws ~6,400 samples
# against a pool of ~1,825 at k=7d (~3.5 passes) but only 97 at k=1d
# (~66 passes), where a large lr stops adapting and starts memorising.
#
# Sweep 6 (job 41997554) re-tuned steps (and lr) at k=1d and k=3d on val. The
# values below are each the best of 6-12 combos at that (k, fit length),
# selected on mean dMAE over 30min/1h/2h -- the SAME rule used at k=7d, so the
# selection rule is identical across arms even though the values are not.
#
# What this changes about the science: with one shared config, dose-response
# reads "7d helps, 3d neutral, 1d actively harmful". With per-k configs it
# reads as a clean monotone effect (see CHRONOS2_PERSONALIZATION.md). Those are
# answers to two different questions -- "fixed recipe" vs "best recipe at each
# history length" -- and picking between them is the user's call, not a
# detail. Neither violates the val-only rule.
#
# Bracketed at every k: one rung up and one rung down both degrade.
declare -A PERS_LR=(
  [7d_fit6]=2e-4   [7d_fit24]=1e-4    # sweep 5
  [3d_fit6]=2e-4   [3d_fit24]=1e-4    # sweep 6
  [1d_fit6]=2e-4   [1d_fit24]=5e-5    # sweep 6
)
declare -A PERS_STEPS=(
  [7d_fit6]=200    [7d_fit24]=200     # sweep 4/5
  [3d_fit6]=76     [3d_fit24]=76      # sweep 6; s150 and s200 both worse
  [1d_fit6]=25     [1d_fit24]=50      # sweep 6; s11 and s100+ both worse
)

# Sets fit length, lr and steps together from ONE key, so a fit length can
# never be paired with another k's learning rate or step count -- the failure
# mode that would silently make two arms non-comparable.
pers_set_fit () {
  local k="$1" fit="$2" key="$1_$2"
  case "$fit" in
    fit6)  export CHRONOS2_PERS_FIT_PREDICTION_LENGTH=6 ;;
    fit24) export CHRONOS2_PERS_FIT_PREDICTION_LENGTH=24 ;;
    *) echo "pers_set_fit: expected fit6|fit24, got '$fit'" >&2; return 1 ;;
  esac
  if [ -z "${PERS_LR[$key]:-}" ]; then
    echo "pers_set_fit: no tuned config for '$key' -- k must be 1d|3d|7d" >&2
    return 1
  fi
  export CHRONOS2_PERS_LEARNING_RATE="${PERS_LR[$key]}"
  export CHRONOS2_PERS_NUM_STEPS="${PERS_STEPS[$key]}"
}
