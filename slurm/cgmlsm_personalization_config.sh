#!/bin/bash
# THE FROZEN CGM-LSM PERSONALIZATION CONFIG. Sourced by every test-split arm so
# that all arms provably share one config and the only thing differing between
# them is what each arm is meant to vary (history length, or own-vs-donor
# history). Mirrors slurm/chronos2_personalization_config.sh in role and shape.
#
# ONE FILE, NOT COPIES IN EACH SBATCH. If these numbers were pasted into each
# arm's script, an edit to one would silently break the "identical config"
# property the whole comparison rests on -- and nothing would error.
#
# PROVENANCE: val split only, never test. Three rounds on all 250 eligible val
# participants (jobs 42313956 / 42323142 / 42378599, ranked by 42386098).
# Full trail below; the short version of what each axis is bracketed by:
#
#   base ckpt      2 epochs at lr=1e-2 from the T1D-excluded GlucoFM-Bench
#                  pretrain. Bracketed: lr=2e-2 destroys the model outright
#                  (100% PAD, zero scoreable predictions), lr=5e-3 is worse,
#                  1 epoch worse, 3-4 epochs worse. This SUPERSEDES the old
#                  settled ep1/lr=5e-6 checkpoint -- see the seed caveat below.
#   learning_rate  5e-3 at EVERY k. Bracketed on both sides: 2e-3 is worse and
#                  1e-2 is worse (round 3 exists purely to establish that
#                  upper side, after rounds 1-2 both ended on the grid edge).
#   num_steps      PER-k: 5 / 10 / 15 at k=1d / 3d / 7d. Each is interior to
#                  its own per-k grid. The step budget scales with history
#                  length because the pool of distinct fit chunks does -- at
#                  stride 12, k=7d yields ~157 chunks, k=3d ~67, k=1d only ~13,
#                  so a fixed budget would mean ~5 passes at k=7d but ~60 at
#                  k=1d, where a model stops adapting and starts memorising.
#                  That is the mechanism behind Chronos-2's k=1d failure, and
#                  the per-k grids were designed around it rather than
#                  rediscovering it.
#   batch_size     8 chunks per step.
#   fit_stride     12 bins (1h) between overlapping fit chunks.
#   seed           42, deliberately -- see below.
#
# SELECTION CRITERION: own_vs_zs, i.e. how much better the personalized model
# is than the same base with no fine-tuning. NOT the own-minus-donor gap. An
# earlier version of this work ranked on that gap to strip out the
# non-person-specific part, and it was gameable: the gap widens either by
# helping own-history or by HARMING donor-history, and the aggressive recipes
# do the latter. At lr=1e-2/k=7d a stranger's history lands 5.6-7.5 mg/dL WORSE
# than zero-shot, inflating the gap to +11.3 while the actual deliverable is
# only the 7th best. The donor arm is a NEGATIVE CONTROL, not a target.
#
# ================== TWO CAVEATS THAT MUST TRAVEL WITH ANY REPORT ==============
#
# 1. THE PERSON-SPECIFIC DECOMPOSITION IS NOISY, THE TOTAL IS NOT. At the
#    frozen k=7d config a donor's history scores -0.07 (i.e. nothing), which
#    would read as "the control passes cleanly, the whole gain is
#    person-specific". DO NOT REPORT IT THAT WAY. Across the three top k=7d
#    configs -- statistically indistinguishable on the total gain, spread 0.32
#    mg/dL -- the donor effect swings +1.69 / -0.07 / -1.15. The near-zero at
#    the frozen config is within that swing, so treating it as a clean control
#    is reading noise. Report the total gain, and report the donor comparison
#    with its instability stated.
#
#    This differs from the Chronos-2 arm, where a donor's history was worse
#    than zero-shot at all 5 horizons and the control did pass cleanly. Do not
#    carry that interpretation over to CGM-LSM.
#
# 2. SEED 42 EVERYWHERE, BY EXPLICIT USER DECISION (2026-09-06), AND THE BASE
#    IS SEED-SENSITIVE. The base recipe's test-window 2h MAE is
#    24.5/26.3/30.1/30.6 across seeds 42/2024/7/123 -- seed 42 is the LUCKIEST,
#    ~3.4 mg/dL better than the 4-seed mean. It is used anyway because every
#    other model in this comparison (Chronos-2, SSM-CGM, TimesFM) also runs a
#    single seed 42 with no stability check, so switching CGM-LSM alone would
#    make it LESS comparable, not more. The comparison is fair in that every
#    model gets the same unmeasured lottery; it is NOT unbiased. A project-wide
#    multi-seed check is an open TO-DO (model_input/HANDOFF.md). Do not "fix"
#    this for CGM-LSM in isolation.
# ==============================================================================
#
# VAL EFFECT SIZES AT THE FROZEN CONFIGS (mean over 30min/1h/2h, 250 val
# participants). These are SELECTION statistics -- do not quote them as effect
# sizes; the test-split numbers are the result:
#   k=1d  own_vs_zs +4.04   (donor +1.55)
#   k=3d  own_vs_zs +4.68   (donor +1.47)
#   k=7d  own_vs_zs +4.98   (donor -0.07, see caveat 1)
# Monotone in history length, which is the expected dose-response.

set -euo pipefail

# --- shared by every arm -------------------------------------------------
export CGMLSM_PERS_BASE_CKPT="$AIREADI_SCRATCH/aireadi_cgmlsm/finetune_sweep4_ep2_lr1e-2/final"
export CGMLSM_PERS_BATCH_SIZE=8
export CGMLSM_PERS_FIT_STRIDE=12
export CGMLSM_PERS_SEED=42
# NOTE: num_steps and learning_rate are NOT set here -- num_steps is per-k and
# is set together with lr by pers_set_k below. A default here would let an arm
# that forgets to call pers_set_k run silently on the wrong config.

# --- per history length --------------------------------------------------
declare -A PERS_LR=(
  [1d]=5e-3   [3d]=5e-3   [7d]=5e-3     # bracketed both sides at every k
)
declare -A PERS_STEPS=(
  [1d]=5      [3d]=10     [7d]=15       # each interior to its own per-k grid
)

# Sets lr and steps together from ONE key, so a history length can never be
# paired with another k's step count -- the failure mode that would silently
# make two arms non-comparable.
pers_set_k () {
  local k="$1"
  if [ -z "${PERS_LR[$k]:-}" ]; then
    echo "pers_set_k: no tuned config for '$k' -- must be 1d|3d|7d" >&2
    return 1
  fi
  export CGMLSM_PERS_LR="${PERS_LR[$k]}"
  export CGMLSM_PERS_NUM_STEPS="${PERS_STEPS[$k]}"
}
