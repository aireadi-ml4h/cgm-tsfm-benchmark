"""Chronos-2 personalization fine-tuning: 1, 3, or 7 days of a held-out
participant's OWN past history, per the proposal's second research question.

Phase 1 is GLUCOSE-ONLY. Companion docs: code/CHRONOS2_PERSONALIZATION.md
(design, arms, decisions), code/model_input/build_personalization_eval_set.py
(the causal boundary and the eval set E), code/CHRONOS2_EVAL.md (the zero-shot
arms this is compared against).

This is the first use of BaseChronosPipeline.fit() anywhere in this project;
every previous Chronos-2 arm is predict_df only.

THE CAUSAL BOUNDARY, AND HOW IT IS ENFORCED
===========================================
build_personalization_eval_set.py computes, per participant p, an anchor T(p)
(p's earliest eval window, all of which sit >= 7 days into p's own record) and
history_start_kd = T(p) - k days. This script:

  1. slices p's own glucose to [history_start_kd, T(p)) -- STRICTLY before
     the anchor, exclusive upper bound;
  2. calls .fit() on that slice alone;
  3. predicts p's eval windows, all of which start at or after T(p).

The boundary is enforced by the slice, not by a mask or a flag: post-anchor
data is never loaded into the process, so nothing inside Chronos2Dataset's
random window sampling can reach it. Verified there rather than assumed --
Chronos2Dataset._construct_slice draws slice_idx from
randint(min_past, full_length - prediction_length + 1) over whatever series
it is handed, so the series bound IS the leak bound.

Cross-participant leakage is structurally impossible too: one .fit() per
participant, on one participant's array. There is no pooled fine-tune here.
(The one exception is the deliberate negative-control arm below, which fits
on a DIFFERENT participant's history on purpose.)

WHY min_past = context_length = 144
===================================
Chronos2Dataset slices context as past_tensor[:, slice_idx - context_length :
slice_idx] whenever slice_idx >= context_length. Leaving both at their
defaults would fine-tune on contexts of arbitrary length while every
evaluation window uses exactly 144 bins (12h) -- a train/eval mismatch with
no upside. Pinning both to CONTEXT_BINS makes every fine-tuning example
structurally identical to an eval window.

It also makes the 1-day arm's real information content visible rather than
hidden: a 288-bin history with min_past=144 and prediction_length=48 admits
288 - 48 + 1 - 144 = 97 distinct training windows. That is what "1 day of
history" actually amounts to, and it should be read that way, not disguised
by allowing short contexts to inflate the window count.

WHY prediction_length = 48 (4h) FOR THE FIT, BUT ALL 5 HORIZONS EVALUATED
=========================================================================
One fine-tuned model per (participant, k), fitted at the longest horizon and
evaluated at all five. The alternative -- one fit per (participant, k,
horizon) -- is 5x the fits (3,900 instead of 780) for a question nobody
asked; one adaptation per person is also what a deployment would actually do.
DISCLOSED LIMITATION: horizon-matched fine-tuning is therefore untested. The
val-split sweep includes a fit-at-6 vs fit-at-48 check at one k so the size
of that limitation is measured rather than asserted.

The anchor is computed once per participant across all five horizons
(build_personalization_eval_set.py), which is what makes a single fit
leak-free for every horizon at once.

NaN IN THE HISTORY IS PASSED THROUGH, NOT FILLED
================================================
cgm_glucose is never imputed anywhere in this project (report.md Sec 13,
Sec 15.7). The history slice keeps its residual NaN, and Chronos-2 handles
both ends of that correctly -- checked against the vendored source, not
assumed:
  - context NaN: model.py:398 builds context_mask = ~isnan(context), and
    model.py:416-419 marks a patch attended only if at least one bin in it is
    observed.
  - TRAINING-TARGET NaN: model.py:538 builds future_target_mask =
    ~isnan(future_target) when none is supplied, model.py:540 replaces those
    entries with 0.0, and model.py:562 folds the mask into loss_mask. So a
    missing bin is excluded from the loss, not learned as a real value.
This is the same treatment the eval side already gives horizon NaN (dropna
before scoring), arrived at independently by the vendor.

RESUME IS AT PARTICIPANT GRANULARITY, NOT WINDOW GRANULARITY
============================================================
chronos2_zeroshot_eval.py resumes by skipping window_ids already present in
the output file, and says why that is safe: predict_df is deterministic given
a fixed context and model. That argument does NOT hold here -- fine-tuning is
stochastic, so two invocations produce different models and their predictions
are not interchangeable. Resume therefore tracks COMPLETED PARTICIPANTS in
participants_done.csv, and on startup discards any prediction rows belonging
to a participant not listed there (a participant interrupted mid-horizon is
redone from scratch, not topped up). Seeds are recorded in summary.csv.

ARMS (CHRONOS2_PERS_HISTORY_DAYS / CHRONOS2_PERS_SHUFFLE_DONOR)
===============================================================
  ft/p1d, ft/p3d, ft/p7d   k = 1, 3, 7 days of the participant's own history
  ft/p7d_shuffled          NEGATIVE CONTROL: 7 days of a DIFFERENT, randomly
                           assigned participant's history. Not optional. A
                           gain over zero-shot is otherwise equally well
                           explained by generic adaptation to AI-READI /
                           Dexcom-G6 / 5-minute-grid data as by adaptation to
                           the person. This arm separates those two.
  mode=none                plumbing check: no .fit() call, so it must
                           reproduce full/imputed glucose_only exactly on E.
                           Run this before trusting any arm above.

Zero-shot reference arms need no run at all: compare_arms(restrict_keys=E)
re-scores the existing chronos2_zeroshot_results/ prediction files on E.

PHASE 2: COVARIATES (CHRONOS2_PERS_CONDITION=with_covariates)
=============================================================
Four nested arms, channels per CHRONOS2_PERSONALIZATION.md. The channel set is
NOT specified here -- it is read off chronos2_zeroshot_eval's own ACTIVE_*
lists, which are driven by the same CHRONOS2_INCLUDE_* / CHRONOS2_ABLATE_CHANNELS
env vars every zero-shot arm already uses. That indirection is the point: the
four phase-2 arms have EXACT existing zero-shot counterparts, so importing the
zero-shot module's channel selection AND its build_context_future_truth() makes
the mode=none reproduction a real test rather than a comparison of two
independently written column builders.

  arm 1  5phys                                 == loo/drop_oxygen_saturation
  arm 2  5phys + predmeal_prob                 == full/with_predmeal_prob_drop_oxysat
  arm 3  5phys + predmeal_prob + oxysat(+flag)  == full/with_predmeal_prob
  arm 4  5phys + predmeal_prob + sleep_stage   == full/with_sleep_stage_predmeal_drop_oxysat

(arms 2/3/4 carry predmeal_prob, so their zero-shot counterpart is the one in
chronos2_zeroshot_results_ffillfix/, not the pre-fix tree -- CHRONOS2_EVAL.md.)

THE FIT-SIDE PATH, AND THE FOUR THINGS THAT WOULD SILENTLY WRECK IT
-------------------------------------------------------------------
1. PAST-ONLY vs KNOWN-FUTURE. from_data_frame's known_covariates_names is the
   whole declaration; everything not listed is past-only. Verified in
   preprocess.py::_build_prepared_inputs (":420  past-only first, known-future
   last (Chronos2Dataset relies on this row order)") and
   dataset.py::_construct_slice (":218 future_covariates = past_tensor[
   -n_future_covariates:, slice_idx : slice_idx + prediction_length]"). So a
   known-future covariate DOES get its real future values during a training
   slice -- which is correct for oxygen_saturation_applicable, a calendar fact
   (report.md Sec 13.1), and would be the SSM-CGM oracle report.md Sec 15.5
   removed for anything physiological. The physiological channels are
   past-only, and _construct_slice ":216 future_target[n_targets:] = nan"
   plus model.py:538-562 keeps them out of the loss as well as out of the
   input.

2. batch_size COUNTS VARIATES, NOT SERIES. dataset.py:296
   (`current_batch_size += self.inputs[input_idx]["context"].shape[0]`, where
   shape[0] is n_targets + n_covariates) and fit()'s own docstring
   ("the batch size here means the number of time series, including target(s)
   and covariates"). With 1 target + 6 covariates, batch_size=32 admits FIVE
   training windows per step, not 32 -- the covariate arms would train on ~6x
   less data per step than the glucose-only arm at the same nominal setting,
   and phase 2 would be measuring that instead of covariates. Nothing errors.
   So CHRONOS2_PERS_BATCH_SIZE is interpreted here as WINDOWS per step and
   multiplied by the variate count before it reaches fit(). The multiplier is
   printed and recorded in summary.csv.

3. THE TARGET IS TAKEN FROM THE GLUCOSE RECORD, NOT FROM build_grid'S JOIN.
   build_grid() left-joins glucose onto the analysis_dataframe WEARABLE grid,
   so a naive covariate history slice is bounded by wearable coverage while
   phase 1's is bounded by the glucose record -- the two do not coincide.
   Taking the target that way would make the phase-1-vs-phase-2 difference
   partly a difference in how much target data the fit saw, and would also
   change the number of distinct training windows per participant, so
   num_steps would not mean the same thing across arms.
   Instead the history is reindexed onto the COMPLETE 5-minute range
   [history_start, anchor) and the target is joined from
   load_cleaned_glucose_binned() -- byte-identical to phase 1's history_array.
   Only the COVARIATE columns carry the wearable bound, as explicit NaN, which
   Chronos-2 masks (module docstring above). Measured before building this:
   225 of 232 covariate-eligible test participants have 100% wearable grid
   coverage over their own 7-day slice, median grid coverage 1.0000
   (code/model_input/measure_covariate_history_coverage.py ->
   model_input/personalization/history_coverage_cov_test.csv), so the NaN
   region is empty for almost everyone; the per-participant real fraction of
   every channel is recorded in participants_done_shard<N>.csv either way.

4. CATEGORICAL ENCODING IS NOT IDENTICAL TO THE ZERO-SHOT ARM'S, and cannot
   be. use_target_encoding=True is the default on both paths, and it is a
   PER-(ITEM, CATEGORY) mean of the target (preprocess.py::_encode_categorical
   -> _target_encode). On the predict side an item is one 144-bin window; on
   the fit side an item is the participant's whole history. Two consequences,
   the second sharper than the first:

   a. The encoded value of a category is that participant's mean glucose while
      in that category over the whole week, not each window's own mean.
      Arguably more meaningful; definitely different.
   b. It is computed ONCE over the full history and then Chronos2Dataset
      samples random (context, future) slices out of the already-encoded
      array. So the encoded value at bin t reflects glucose from bins after t
      -- a within-history, train-time-only leak across the sampled training
      window's own boundary. It CANNOT reach the eval horizon (the whole array
      ends strictly before the anchor, and predict_df re-encodes per window
      from context glucose only), so the causal boundary is intact. But it can
      make the fit over-trust a categorical channel that is less informative
      at eval time.

   This is confined to CATEGORICAL channels, which puts the four arms in two
   classes worth keeping straight when reading the results:
     arms 1 and 2   all-numeric, no target encoding anywhere -- so the
                    arm2-arm1 predmeal_prob attribution is clean.
     arms 3 and 4   carry a categorical (oxygen_saturation_applicable is bool,
                    hence category; sleep_stage is a string), so the arm3-arm2
                    and arm4-arm2 attributions carry (b).
   Disclosed, not fixed: use_target_encoding=False would make the fit and the
   predict path encode the same channel two different ways, which is worse,
   and there is no setting that makes a whole-history fit encode per-window.
   The mode=none verification is unaffected either way -- with fitting disabled
   the fit-side encoding is never reached.

The PREDICT side calls chronos2_zeroshot_eval.build_context_future_truth()
directly, so the ffill-per-window fix, the oxygen_saturation_applicable-in-df
requirement, and the sleep_stage "unknown" fill are all inherited rather than
reimplemented. The fit side applies the same three fills over the participant's
own history slice (within-participant by construction -- the cross-participant
contamination that fix was for cannot arise from a single-participant array).
"""
import os
import sys
import time

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
# Sibling modules live in adjacent topic directories under code/.
_CODE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _d in ("analysis", "baselines", "cgmlsm", "chronos2", "meal_detection", "model_input", "ssmcgm", "timesfm"):
    sys.path.insert(0, os.path.join(_CODE, _d))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "model_input"))

REPO_ROOT = "/path/to/aireadi-collab"
sys.path.insert(0, f"{REPO_ROOT}/code/external/chronos-forecasting/src")
from chronos import BaseChronosPipeline  # noqa: E402

from build_participant_grid import (  # noqa: E402
    build_grid, load_cleaned_glucose_binned, load_participants_covariates,
)
# Phase 2 reuses the zero-shot script's channel selection AND its predict-side
# column builder rather than reimplementing either. Imported unconditionally
# (it has no import-time side effects -- module level is env-var reads and
# constants only), so the glucose-only path is unaffected.
from chronos.chronos2.preprocess import from_data_frame  # noqa: E402
import chronos2_zeroshot_eval as zs  # noqa: E402

CONTEXT_BINS = 144  # 12h, the one context length every model in this project uses
BIN_MINUTES = 5
HORIZON_BINS = {"30min": 6, "1h": 12, "2h": 24, "3h": 36, "4h": 48}
FIT_PREDICTION_LENGTH = int(os.environ.get("CHRONOS2_PERS_FIT_PREDICTION_LENGTH", "48"))

SPLIT = os.environ.get("CHRONOS2_PERS_SPLIT", "test")
# phase 1 is glucose_only; phase 2 sets CHRONOS2_PERS_CONDITION=with_covariates
# and lets the CHRONOS2_INCLUDE_*/CHRONOS2_ABLATE_CHANNELS vars pick the arm.
CONDITION = os.environ.get("CHRONOS2_PERS_CONDITION", "glucose_only")
if CONDITION not in ("glucose_only", "with_covariates"):
    raise SystemExit(f"CHRONOS2_PERS_CONDITION must be glucose_only or "
                      f"with_covariates, got {CONDITION!r}")
WITH_COVARIATES = CONDITION == "with_covariates"
# separate eval set: E_cov is a strict subset of E (with_covariates additionally
# requires heart_rate + respiratory_rate coverage), and it reuses phase 1's
# anchors so a covariate-vs-glucose-only difference is not an anchor-position
# difference. See build_personalization_eval_set.py's "PHASE 2" docstring.
# CHRONOS2_PERS_HISTORY_BAR_DAYS: selects the eval set built at a lower
# eligibility bar -- the SECOND, PARALLEL family of results
# (build_personalization_eval_set.py's docstring). Lowering the bar to B admits
# every window >= B days into a participant's record, which is ~3-4x more
# windows and removes almost all of the power cost: measured MDE ratio against
# the full test set is 1.18 at bar=3d and 0.89-1.03 at bar=1d, versus 1.2-2.9
# at bar=7d. The cost is that a bar=B set has different windows AND different
# participants from a bar=B' set, so arms across bars are NOT comparable --
# which is why the primary analysis keeps bar=7 for every arm.
# Constraint: k <= bar, since a k-day history must fit inside the record the
# bar guarantees.
HISTORY_BAR_DAYS = int(os.environ.get("CHRONOS2_PERS_HISTORY_BAR_DAYS", "7"))
BAR_SUFFIX = "" if HISTORY_BAR_DAYS == 7 else f"_bar{HISTORY_BAR_DAYS}d"
EVAL_SET_PATH = (f"{REPO_ROOT}/model_input/personalization/eval_set_cov_{SPLIT}{BAR_SUFFIX}.csv"
                 if WITH_COVARIATES else
                 f"{REPO_ROOT}/model_input/personalization/eval_set_{SPLIT}{BAR_SUFFIX}.csv")
EVAL_DIR = (f"{REPO_ROOT}/model_input/zeroshot_eval" if SPLIT == "test"
            else f"{REPO_ROOT}/model_input/zeroshot_eval_{SPLIT}")

# The arm's channels, read off the zero-shot module so fit and predict cannot
# disagree about which channels exist. Order matters only in that
# from_data_frame puts known-future variates last regardless
# (preprocess.py:420), so it is not relied on here.
PAST_ONLY_CHANNELS = (zs.ACTIVE_COVARIATE_FEATURE_CHANNELS + zs.ACTIVE_SLEEP_STAGE_CHANNELS
                      + zs.ACTIVE_PREDMEAL_PROB_CHANNELS
                      + zs.ACTIVE_SLEEP_REGULARITY_CHANNELS) if WITH_COVARIATES else []
KNOWN_FUTURE_CHANNELS = (zs.ACTIVE_COVARIATE_APPLICABILITY_CHANNELS
                         + zs.ACTIVE_TOD_CHANNELS
                         + zs.ACTIVE_MINUTE_OF_DAY_CHANNELS) if WITH_COVARIATES else []
if WITH_COVARIATES and zs.ACTIVE_STATIC_CHANNELS:
    # Not an oversight and not a TODO: CHRONOS2_EVAL.md's "Statics" section is a
    # closed negative result (instance norm zeroes a constant channel exactly,
    # per-item target encoding collapses string statics onto the window mean),
    # and personalization replaces them outright -- the whole weight update
    # encodes person identity. Refused rather than silently passed through,
    # because a static column over a 2,016-bin history would go through a
    # DIFFERENT and untested code path from the 144-bin window it was measured
    # broken on.
    raise SystemExit("CHRONOS2_INCLUDE_STATIC is not supported in phase 2; see "
                      "CHRONOS2_PERSONALIZATION.md's statics section")
ALL_CHANNELS = PAST_ONLY_CHANNELS + KNOWN_FUTURE_CHANNELS
N_VARIATES = 1 + len(ALL_CHANNELS)  # 1 target + covariates; see gotcha 2 above

HISTORY_DAYS = int(os.environ.get("CHRONOS2_PERS_HISTORY_DAYS", "7"))
# LoRA is the default, opposite of the CGM-LSM fine-tuning decision in
# HANDOFF.md and for the opposite reason: there the model was ~4.9M params
# and LoRA had no cost to save, here a large pretrained model is being adapted
# on 288-2016 points from one person and the low-rank constraint is the point
# rather than an overhead. peft 0.20.0 is already in venv-chronos.
FINETUNE_MODE = os.environ.get("CHRONOS2_PERS_FINETUNE_MODE", "lora")  # lora | full | none
LEARNING_RATE = float(os.environ.get("CHRONOS2_PERS_LEARNING_RATE", "1e-5"))
NUM_STEPS = int(os.environ.get("CHRONOS2_PERS_NUM_STEPS", "100"))
# WINDOWS per step, not variates. fit()'s own batch_size counts variates, so
# this is multiplied by N_VARIATES below -- see gotcha 2 in the module
# docstring. Leaving it unscaled would make every covariate arm train on ~1/7
# the data per step and phase 2 would measure that instead of covariates.
BATCH_SIZE = int(os.environ.get("CHRONOS2_PERS_BATCH_SIZE", "32"))
FIT_BATCH_SIZE = BATCH_SIZE * N_VARIATES
LORA_R = int(os.environ.get("CHRONOS2_PERS_LORA_R", "8"))
SEED = int(os.environ.get("CHRONOS2_PERS_SEED", "42"))
SHUFFLE_DONOR = os.environ.get("CHRONOS2_PERS_SHUFFLE_DONOR", "0") == "1"

# Shard by participant so several jobs can share the work. Each shard loads
# the base pipeline once and fits per participant; .fit() deep-copies the
# model and returns a NEW pipeline, leaving the base untouched (verified in
# chronos2/pipeline.py), so participants inside one shard do not contaminate
# each other's adaptation.
SHARD = int(os.environ.get("CHRONOS2_PERS_SHARD", "0"))
NUM_SHARDS = int(os.environ.get("CHRONOS2_PERS_NUM_SHARDS", "1"))
MAX_PARTICIPANTS = os.environ.get("CHRONOS2_PERS_MAX_PARTICIPANTS")
MAX_PARTICIPANTS = int(MAX_PARTICIPANTS) if MAX_PARTICIPANTS else None

# ROLLING-BOUNDARY ROBUSTNESS CHECK (CHRONOS2_PERS_ROLLING=1).
#
# Every production arm fits ONCE per participant, on [T(p)-k, T(p)) where T(p)
# is that participant's fixed anchor, and then scores ALL of their windows with
# that one adapted model. Windows late in a record are therefore scored by a
# model whose history is days stale. That is deliberately conservative -- the
# history always precedes every scored window, so there is no leak -- but it
# means the production arms measure a LOWER BOUND on what personalization can
# do, and the size of the gap is unknown.
#
# Rolling mode re-fits per WINDOW on that window's own preceding k days,
# [context_start - k, context_start). Still strictly causal (the fit ends where
# the window's context begins, so it sees neither the context nor the future),
# but never stale. Comparing the two on the same windows answers "how much did
# the fixed anchor cost?" -- and if the answer is "a lot", recency of the
# adaptation window is itself a finding.
#
# Cost is one fit per window instead of per participant, ~4x more at k=7d
# (~3.9 windows/participant at 1h), which is why this runs on a SUBSAMPLE via
# CHRONOS2_PERS_MAX_PARTICIPANTS and on selected horizons only.
ROLLING = os.environ.get("CHRONOS2_PERS_ROLLING", "0") == "1"
ROLLING_HORIZONS = [h for h in os.environ.get(
    "CHRONOS2_PERS_ROLLING_HORIZONS", "1h,2h").split(",") if h]

# the default carries the condition: without it a phase-2 arm would land in
# the phase-1 directory, and since predictions are APPENDED (mode="a") that
# would silently interleave two channel sets in one file
OUTPUT_DIR = os.environ.get(
    "CHRONOS2_PERS_RESULTS_DIR",
    f"{REPO_ROOT}/chronos2_personalization_results/{SPLIT}{BAR_SUFFIX}/ft_p{HISTORY_DAYS}d"
    + ("_cov" if WITH_COVARIATES else ""),
)
if HISTORY_DAYS > HISTORY_BAR_DAYS:
    # A k-day history cannot fit inside a record only guaranteed to extend
    # `bar` days before the earliest scored window. The eval set would not even
    # have a history_start_{k}d column, so this would fail later and less
    # clearly; refused here instead.
    raise SystemExit(f"CHRONOS2_PERS_HISTORY_DAYS={HISTORY_DAYS} exceeds "
                      f"CHRONOS2_PERS_HISTORY_BAR_DAYS={HISTORY_BAR_DAYS}")
# Fine-tuned checkpoints are written to node-local SSD and die with the job.
# 780 copies of a Chronos-2-sized model is not something to keep, and nothing
# downstream reads them: each adapted model is used once, in-process, for its
# own participant's windows.
FIT_SCRATCH_DIR = os.environ.get("TMPDIR", "/tmp") + "/chronos2_pers_fit"


def load_eval_set():
    e = pd.read_csv(EVAL_SET_PATH, dtype={"person_id": str, "window_id": str})
    # Only the history columns this eval set actually has. A bar=B set carries
    # history_start_{k}d for k <= B only (build_personalization_eval_set.py
    # filters HISTORY_DAYS the same way), because a k-day history cannot fit
    # inside a record guaranteed only B days deep. Hardcoding (1, 3, 7) here
    # made every bar<7 run die on KeyError: 'history_start_7d'.
    hist_cols = [f"history_start_{k}d" for k in (1, 3, 7)
                 if f"history_start_{k}d" in e.columns]
    if f"history_start_{HISTORY_DAYS}d" not in hist_cols:
        raise SystemExit(
            f"{EVAL_SET_PATH} has no history_start_{HISTORY_DAYS}d column "
            f"(present: {hist_cols}) -- CHRONOS2_PERS_HISTORY_DAYS={HISTORY_DAYS} "
            f"is not available at bar={HISTORY_BAR_DAYS}d")
    for col in ["context_start", "t0", "anchor"] + hist_cols:
        e[col] = pd.to_datetime(e[col], utc=True)
    return e


def assign_donors(person_ids, seed):
    """Negative control: map each participant to a DIFFERENT participant whose
    history will be used instead of their own.

    A plain shuffle can leave fixed points (someone mapped to themselves),
    which would silently turn those participants back into a real
    personalization arm and dilute the control. Rotating the shuffled order by
    one guarantees a derangement -- no fixed points -- for any n >= 2."""
    rng = np.random.default_rng(seed)
    shuffled = list(person_ids)
    rng.shuffle(shuffled)
    rotated = shuffled[1:] + shuffled[:1]
    donors = dict(zip(shuffled, rotated))
    assert all(k != v for k, v in donors.items()), "donor assignment has a fixed point"
    return donors


def history_array(person_id, history_start, anchor):
    """The participant's own glucose over [history_start, anchor), on the same
    5-minute grid as the eval windows.

    Upper bound is EXCLUSIVE -- the anchor bin itself belongs to the first
    eval window's context, and while a context bin is not a leak (zero-shot
    sees it too), keeping the boundary exclusive means history and eval never
    share a single row and the invariant is trivial to state.

    Reindexed onto a complete 5-minute range so that missing bins are explicit
    NaN at their true positions rather than silently closing the gap and
    shifting everything after them in time. Chronos-2 masks those out of both
    the context and the loss (module docstring)."""
    s = load_cleaned_glucose_binned(person_id)
    if s is None:
        return None
    s = s[(s.index >= history_start) & (s.index < anchor)]
    if s.empty:
        return None
    full = pd.date_range(history_start, anchor - pd.Timedelta(minutes=BIN_MINUTES),
                          freq=f"{BIN_MINUTES}min", tz="UTC")
    s = s.reindex(full)
    return s.to_numpy(dtype=np.float32)


def history_frame(person_id, history_start, anchor, covariates_by_pid):
    """Phase 2's history: the same slice as history_array() plus this arm's
    covariate columns, as a one-item long-format frame ready for
    from_data_frame().

    THE TARGET IS DELIBERATELY NOT build_grid()'s cgm_glucose. build_grid()
    left-joins glucose onto the analysis_dataframe WEARABLE grid, so its
    glucose column is bounded by wearable coverage while phase 1's history is
    bounded by the glucose record. Reindexing onto the complete 5-minute range
    and re-joining glucose from load_cleaned_glucose_binned() makes the target
    byte-identical to history_array()'s, so the only thing phase 2 changes
    relative to phase 1 is the presence of covariate columns -- not how much
    target the fit saw, and not how many distinct training windows exist (so
    num_steps means the same thing in both phases). The covariate columns keep
    the wearable bound, as explicit NaN, which Chronos-2 masks out of both the
    context and the loss. See gotcha 3 in the module docstring.

    Returns (frame, real_fractions) where real_fractions is {channel: fraction
    of the slice that is non-null}, recorded per participant so a thin-coverage
    fit is identifiable after the fact rather than invisible."""
    values = history_array(person_id, history_start, anchor)
    if values is None:
        return None, {}
    full = pd.date_range(history_start, anchor - pd.Timedelta(minutes=BIN_MINUTES),
                          freq=f"{BIN_MINUTES}min", tz="UTC")
    frame = pd.DataFrame({"item_id": str(person_id), "bin_start": full,
                           "cgm_glucose": values})

    grid = build_grid(person_id, covariates_by_pid)
    if grid is None:
        return None, {}
    grid = grid[(grid["bin_start"] >= history_start) & (grid["bin_start"] < anchor)]
    missing = [c for c in ALL_CHANNELS if c not in grid.columns]
    if missing:
        raise SystemExit(f"build_grid has no column(s) {missing} for {person_id} -- "
                          "the arm's channel set does not match the grid's")
    frame = frame.merge(grid[["bin_start"] + ALL_CHANNELS], on="bin_start", how="left")

    real_fractions = {c: round(float(frame[c].notna().mean()), 4) for c in ALL_CHANNELS}

    # The same three fills the predict side applies, over the participant's own
    # history. Cross-participant contamination -- the thing the 2026-09-02
    # ffill fix was for -- cannot arise here: this is one participant's array,
    # so an ffill has nowhere else to reach. Values and sentinels are copied
    # from zs.build_context_future_truth rather than restated, so the two
    # cannot drift apart.
    if zs.SLEEP_STAGE_CHANNEL in ALL_CHANNELS:
        frame[zs.SLEEP_STAGE_CHANNEL] = frame[zs.SLEEP_STAGE_CHANNEL].fillna("unknown")
    if zs.PREDMEAL_PROB_CHANNEL in ALL_CHANNELS:
        frame[zs.PREDMEAL_PROB_CHANNEL] = (
            frame[zs.PREDMEAL_PROB_CHANNEL].ffill().bfill().fillna(0.0))
    if zs.SLEEP_REGULARITY_CHANNEL in ALL_CHANNELS:
        frame[zs.SLEEP_REGULARITY_CHANNEL] = (
            frame[zs.SLEEP_REGULARITY_CHANNEL].ffill().bfill().fillna(-200.0))
    # tz-naive, same reason as load_export(): from_data_frame -> normalize_df
    # calls .to_numpy().view("int64"), which a tz-aware column's object array
    # cannot satisfy. Everything here is UTC.
    frame["bin_start"] = frame["bin_start"].dt.tz_localize(None)
    return frame, real_fractions


def fit_inputs(history):
    """fit()'s `inputs` for one participant.

    glucose_only: a plain 1-D tensor, unchanged from phase 1.
    with_covariates: from_data_frame's PreparedInput, which is the vendor's own
    training path for covariates (fit()'s docstring recommends exactly this).
    known_covariates_names is the entire past-only/known-future declaration --
    everything omitted is past-only. For arms 1/2/4 the list is empty; for arm
    3 it is ["oxygen_saturation_applicable"], a calendar fact. See gotcha 1 in
    the module docstring for the source-level verification that this is
    sufficient and that no physiological future value reaches the model."""
    if not WITH_COVARIATES:
        return [torch.tensor(history)]
    return from_data_frame(
        df=history,
        target_columns=["cgm_glucose"],
        prediction_length=FIT_PREDICTION_LENGTH,
        known_covariates_names=KNOWN_FUTURE_CHANNELS or None,
        id_column="item_id",
        timestamp_column="bin_start",
    )


def finetune_for_participant(base_pipeline, history, tag):
    """One .fit() call on one participant's history slice. Returns a new
    pipeline; the base is untouched and reused for the next participant."""
    if FINETUNE_MODE == "none":
        return base_pipeline, 0.0
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    fit_start = time.time()
    kwargs = dict(
        inputs=fit_inputs(history),
        prediction_length=FIT_PREDICTION_LENGTH,
        context_length=CONTEXT_BINS,
        min_past=CONTEXT_BINS,
        learning_rate=LEARNING_RATE,
        num_steps=NUM_STEPS,
        batch_size=FIT_BATCH_SIZE,  # WINDOWS * N_VARIATES; gotcha 2
        output_dir=f"{FIT_SCRATCH_DIR}/{tag}",
        finetune_mode=FINETUNE_MODE,
        remove_printer_callback=True,
        disable_data_parallel=True,
        # forwarded to TrainingArguments via **extra_trainer_kwargs. fit()'s
        # own defaults already set save_strategy="no"/eval_strategy="no"/
        # report_to="none"; these three override the noisy ones, since this
        # loop runs .fit() hundreds of times in one job and a per-participant
        # progress bar plus a logging line every 100 steps would bury the
        # per-participant output that actually matters.
        logging_strategy="no",
        disable_tqdm=True,
    )
    if FINETUNE_MODE == "lora":
        kwargs["lora_config"] = dict(
            r=LORA_R,
            lora_alpha=2 * LORA_R,
            target_modules=["self_attention.q", "self_attention.v", "self_attention.k",
                             "self_attention.o", "output_patch_embedding.output_layer"],
        )
    tuned = base_pipeline.fit(**kwargs)
    return tuned, time.time() - fit_start


def predict_participant(pipeline, export_by_horizon, window_ids_by_horizon, person_id):
    """Forecast every eval window this participant has, at every horizon,
    using the model just adapted for them. Returns {horizon: merged df}."""
    out = {}
    for horizon, horizon_bins in HORIZON_BINS.items():
        wanted = window_ids_by_horizon.get((person_id, horizon))
        if not wanted:
            continue
        df = export_by_horizon[horizon]
        df = df[df["window_id"].isin(wanted)]
        if df.empty:
            continue
        # The zero-shot script's own column builder, called directly rather
        # than reimplemented: it carries the per-window ffill fix, the
        # oxygen_saturation_applicable-must-also-be-in-df constraint, and the
        # sleep_stage "unknown" fill. That is what makes the mode=none
        # reproduction a real test of this harness instead of a comparison of
        # two independently written builders. In the glucose_only condition it
        # returns exactly the 3-column context and future_df=None that phase 1
        # built inline.
        context, future_df, truth = zs.build_context_future_truth(df, WITH_COVARIATES)
        forecast = pipeline.predict_df(
            df=context,
            future_df=future_df,
            id_column="window_id",
            timestamp_column="bin_start",
            target="cgm_glucose",
            prediction_length=horizon_bins,
            freq=f"{BIN_MINUTES}min",
        )
        merged = forecast.merge(truth, on=["window_id", "bin_start"], how="left")
        merged["abs_error"] = (merged["predictions"] - merged["actual"]).abs()
        out[horizon] = merged
    return out


def load_export(horizon, keep_window_ids):
    """One horizon's eval export, restricted to E's windows only. Filtering on
    read keeps five horizons of export in memory at once without carrying the
    ~75% of windows that do not clear the 7-day bar."""
    path = f"{EVAL_DIR}/zeroshot_{horizon}_{CONDITION}.csv"
    df = pd.read_csv(path, dtype={"window_id": str, "person_id": str})
    df = df[df["window_id"].isin(keep_window_ids)].copy()
    # tz-naive for predict_df: normalize_df calls .to_numpy().view("int64"),
    # which fails on a tz-aware column's object array (the TypeError
    # chronos2_zeroshot_eval.py's smoke test hit). Everything here is UTC.
    df["bin_start"] = pd.to_datetime(df["bin_start"], utc=True).dt.tz_localize(None)
    df["is_context"] = df["is_context"].astype(bool)
    return df


def done_participants(path):
    if not os.path.exists(path):
        return set()
    return set(pd.read_csv(path, dtype={"person_id": str})["person_id"])


def drop_partial_participants(output_dir, done, window_to_person):
    """Remove prediction rows from participants that were interrupted
    mid-participant. See the module docstring: unlike the zero-shot script,
    partial output cannot simply be topped up here, because the model that
    produced it no longer exists and a re-fit produces a different one."""
    for horizon in HORIZON_BINS:
        path = f"{output_dir}/predictions_{horizon}_{CONDITION}.csv"
        if not os.path.exists(path):
            continue
        df = pd.read_csv(path, dtype={"window_id": str})
        person = df["window_id"].map(window_to_person)
        keep = person.isin(done)
        if not keep.all():
            print(f"  resume: dropping {int((~keep).sum())} rows from "
                  f"{person[~keep].nunique()} interrupted participant(s) in "
                  f"predictions_{horizon}_{CONDITION}.csv", file=sys.stderr)
            df[keep].to_csv(path, index=False)


def done_windows(path):
    if not os.path.exists(path):
        return set()
    return set(pd.read_csv(path, dtype={"window_id": str})["window_id"])


def drop_partial_windows(output_dir, done):
    """Rolling-mode analogue of drop_partial_participants. The unit of work is
    one window, so anything on disk whose window_id is not marked done was
    written by a process that died before recording it, and cannot be topped
    up (the model that produced it is gone)."""
    for horizon in ROLLING_HORIZONS:
        path = f"{output_dir}/predictions_{horizon}_{CONDITION}.csv"
        if not os.path.exists(path):
            continue
        df = pd.read_csv(path, dtype={"window_id": str})
        keep = df["window_id"].isin(done)
        if not keep.all():
            print(f"  resume: dropping {int((~keep).sum())} rows from "
                  f"{df.loc[~keep, 'window_id'].nunique()} interrupted window(s) "
                  f"in predictions_{horizon}_{CONDITION}.csv", file=sys.stderr)
            df[keep].to_csv(path, index=False)


def main_rolling():
    """One fit per WINDOW, on that window's own preceding k days. See the
    ROLLING block near the env vars for why this exists.

    Deliberately reuses history_frame / history_array / finetune_for_participant
    / predict_participant unchanged -- the only thing that differs from a
    production arm is which (history_start, history_end) pair is passed in. If
    this diverged into its own fit path it would stop being a check on the
    production arms."""
    job_start = time.time()
    e = load_eval_set()
    e = e[e["horizon"].isin(ROLLING_HORIZONS)].copy()
    if e.empty:
        raise SystemExit(f"eval set has no windows at horizons {ROLLING_HORIZONS}")

    all_participants = sorted(e["person_id"].unique())
    # SUBSAMPLE FIRST, THEN SHARD -- the opposite order from main(), and the
    # order matters. MAX_PARTICIPANTS here means "the size of the subsample
    # this check runs on", a property of the experiment; applying it per shard
    # instead would make the subsample NUM_SHARDS times larger and its size
    # dependent on how many jobs happened to be used. Deterministic (first N of
    # the sorted list) so a rerun scores identical windows and the comparison
    # against the fixed-anchor arm stays matched.
    if MAX_PARTICIPANTS is not None:
        all_participants = all_participants[:MAX_PARTICIPANTS]
    participants = [p for i, p in enumerate(all_participants) if i % NUM_SHARDS == SHARD]
    work = (e[e["person_id"].isin(participants)]
            .sort_values(["person_id", "horizon", "context_start"])
            .reset_index(drop=True))

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(FIT_SCRATCH_DIR, exist_ok=True)
    done_path = f"{OUTPUT_DIR}/windows_done_shard{SHARD}.csv"
    done = done_windows(done_path)
    if done:
        print(f"resuming: {len(done)} windows already complete in this shard",
              file=sys.stderr)
        drop_partial_windows(OUTPUT_DIR, done)
    todo = work[~work["window_id"].isin(done)]

    export_by_horizon = {h: load_export(h, set(work["window_id"]))
                         for h in ROLLING_HORIZONS}
    print(f"ROLLING BOUNDARY: shard {SHARD}/{NUM_SHARDS}, "
          f"{len(participants)} participants, {len(work)} windows "
          f"({len(todo)} to do), horizons={ROLLING_HORIZONS}, k={HISTORY_DAYS}d, "
          f"fit_prediction_length={FIT_PREDICTION_LENGTH}, lr={LEARNING_RATE}, "
          f"steps={NUM_STEPS}", file=sys.stderr)

    covariates_by_pid = load_participants_covariates() if WITH_COVARIATES else None
    base_pipeline = BaseChronosPipeline.from_pretrained(
        "amazon/chronos-2", device_map="cuda" if torch.cuda.is_available() else "cpu")

    k_delta = pd.Timedelta(days=HISTORY_DAYS)
    fit_seconds_total, n_fitted, skipped = 0.0, 0, []
    for i, row in enumerate(todo.itertuples(index=False)):
        pid, wid, horizon = row.person_id, row.window_id, row.horizon
        # THE ONLY DIFFERENCE FROM A PRODUCTION ARM. History ends at this
        # window's own context_start rather than at the participant's fixed
        # anchor, so it is never stale -- and still strictly causal, because
        # the window's context begins exactly where the history stops.
        hist_end = row.context_start
        hist_start = hist_end - k_delta

        if WITH_COVARIATES:
            history, _ = history_frame(pid, hist_start, hist_end, covariates_by_pid)
            values = None if history is None else history["cgm_glucose"].to_numpy()
        else:
            history = values = history_array(pid, hist_start, hist_end)
        if values is None or len(values) < CONTEXT_BINS + FIT_PREDICTION_LENGTH:
            skipped.append((pid, wid, 0 if values is None else len(values)))
            continue

        tuned, fit_seconds = finetune_for_participant(
            base_pipeline, history, f"{pid}_{wid}_roll")
        fit_seconds_total += fit_seconds
        n_fitted += 1

        # one window, one horizon: predict_participant keys on
        # (person_id, horizon), so a single-entry dict scores exactly this window
        preds = predict_participant(tuned, export_by_horizon, {(pid, horizon): {wid}}, pid)
        for h, merged in preds.items():
            path = f"{OUTPUT_DIR}/predictions_{h}_{CONDITION}.csv"
            merged.to_csv(path, mode="a", header=not os.path.exists(path), index=False)

        pd.DataFrame([{"window_id": wid, "person_id": pid, "horizon": horizon,
                        "history_start": hist_start, "history_end": hist_end,
                        "history_bins": len(values),
                        "real_fraction": round(float(np.isfinite(values).mean()), 4),
                        "fit_seconds": round(fit_seconds, 1)}]).to_csv(
            done_path, mode="a", header=not os.path.exists(done_path), index=False)

        if tuned is not base_pipeline:
            del tuned
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        if (i + 1) % 20 == 0 or i + 1 == len(todo):
            print(f"  {i + 1}/{len(todo)} windows, "
                  f"{fit_seconds_total / max(n_fitted, 1):.1f}s/fit avg, "
                  f"{time.time() - job_start:.0f}s elapsed", file=sys.stderr)

    if skipped:
        print(f"WARNING: skipped {len(skipped)} windows with too-short history: "
              f"{skipped[:5]}", file=sys.stderr)
        pd.DataFrame(skipped, columns=["person_id", "window_id", "history_bins"]).to_csv(
            f"{OUTPUT_DIR}/skipped_windows_shard{SHARD}.csv", index=False)
    print(f"done: {n_fitted} fits, {fit_seconds_total / max(n_fitted, 1):.1f}s/fit avg, "
          f"{time.time() - job_start:.0f}s total", file=sys.stderr)


def main():
    if ROLLING:
        return main_rolling()
    job_start = time.time()
    e = load_eval_set()

    window_to_person = dict(zip(e["window_id"], e["person_id"]))
    window_ids_by_horizon = (e.groupby(["person_id", "horizon"])["window_id"]
                              .apply(set).to_dict())
    history_col = f"history_start_{HISTORY_DAYS}d"
    if history_col not in e.columns:
        raise SystemExit(f"eval set has no {history_col} column; "
                          f"CHRONOS2_PERS_HISTORY_DAYS={HISTORY_DAYS} is not one of 1/3/7")
    anchors = e.groupby("person_id").agg(anchor=("anchor", "first"),
                                          history_start=(history_col, "first"))

    all_participants = sorted(anchors.index)
    donors = assign_donors(all_participants, SEED) if SHUFFLE_DONOR else None

    participants = [p for i, p in enumerate(all_participants) if i % NUM_SHARDS == SHARD]
    if MAX_PARTICIPANTS is not None:
        participants = participants[:MAX_PARTICIPANTS]

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(FIT_SCRATCH_DIR, exist_ok=True)
    done_path = f"{OUTPUT_DIR}/participants_done_shard{SHARD}.csv"
    done = done_participants(done_path)
    if done:
        print(f"resuming: {len(done)} participants already complete in this shard",
              file=sys.stderr)
        drop_partial_participants(OUTPUT_DIR, done, window_to_person)
    todo = [p for p in participants if p not in done]

    keep_ids = set(e.loc[e["person_id"].isin(participants), "window_id"])
    export_by_horizon = {h: load_export(h, keep_ids) for h in HORIZON_BINS}
    print(f"shard {SHARD}/{NUM_SHARDS}: {len(participants)} participants "
          f"({len(todo)} to do), split={SPLIT}, k={HISTORY_DAYS}d, mode={FINETUNE_MODE}, "
          f"shuffled_donor={SHUFFLE_DONOR}", file=sys.stderr)
    print(f"condition={CONDITION}, n_variates={N_VARIATES} "
          f"(past-only {PAST_ONLY_CHANNELS}, known-future {KNOWN_FUTURE_CHANNELS}), "
          f"fit batch_size={FIT_BATCH_SIZE} = {BATCH_SIZE} windows x {N_VARIATES} variates",
          file=sys.stderr)
    covariates_by_pid = load_participants_covariates() if WITH_COVARIATES else None

    load_start = time.time()
    base_pipeline = BaseChronosPipeline.from_pretrained(
        "amazon/chronos-2", device_map="cuda" if torch.cuda.is_available() else "cpu"
    )
    model_load_seconds = round(time.time() - load_start, 1)
    print(f"base model loaded in {model_load_seconds}s", file=sys.stderr)

    fit_seconds_total = 0.0
    n_fitted = 0
    skipped = []
    for i, pid in enumerate(todo):
        anchor = anchors.loc[pid, "anchor"]
        history_start = anchors.loc[pid, "history_start"]
        source_pid = donors[pid] if donors else pid
        if donors:
            # the donor's OWN history window, not the recipient's -- the
            # control is "someone else's week," not "someone else's data
            # sampled over my calendar dates," which for a different
            # participant might not even be inside their record
            history_start = anchors.loc[source_pid, "history_start"]
            anchor = anchors.loc[source_pid, "anchor"]

        expected_bins = HISTORY_DAYS * 24 * 60 // BIN_MINUTES
        cov_fractions = {}
        if WITH_COVARIATES:
            history, cov_fractions = history_frame(source_pid, history_start, anchor,
                                                    covariates_by_pid)
            values = None if history is None else history["cgm_glucose"].to_numpy()
        else:
            history = values = history_array(source_pid, history_start, anchor)
        if values is None or len(values) < CONTEXT_BINS + FIT_PREDICTION_LENGTH:
            # too short for Chronos2Dataset to sample a single window from
            # (it raises after MAX_REJECTED_SAMPLES otherwise). Recorded and
            # skipped rather than silently fitted on a padded array.
            skipped.append((pid, source_pid, 0 if values is None else len(values)))
            continue
        real_frac = float(np.isfinite(values).mean())

        tuned, fit_seconds = finetune_for_participant(base_pipeline, history, f"{pid}_k{HISTORY_DAYS}")
        fit_seconds_total += fit_seconds
        n_fitted += 1

        preds = predict_participant(tuned, export_by_horizon, window_ids_by_horizon, pid)
        for horizon, merged in preds.items():
            path = f"{OUTPUT_DIR}/predictions_{horizon}_{CONDITION}.csv"
            merged.to_csv(path, mode="a", header=not os.path.exists(path), index=False)

        # participant marked done only AFTER every one of their horizons is on
        # disk, so an interruption can never leave a participant listed as
        # complete with only some horizons written
        # cov_fractions is per-channel non-null fraction of the history slice
        # -- the audit trail for gotcha 3: a participant fitted on a history
        # whose covariates are mostly NaN (thin wearable coverage) is
        # identifiable here rather than indistinguishable from a well-covered
        # one. Empty dict in the glucose_only condition, so the column set is
        # unchanged for phase 1.
        pd.DataFrame([{"person_id": pid, "donor_person_id": source_pid,
                        "history_bins": len(values), "expected_bins": expected_bins,
                        "real_fraction": round(real_frac, 4),
                        **{f"cov_frac_{c}": v for c, v in cov_fractions.items()},
                        "fit_seconds": round(fit_seconds, 1)}]).to_csv(
            done_path, mode="a", header=not os.path.exists(done_path), index=False)

        if tuned is not base_pipeline:
            del tuned
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        if (i + 1) % 10 == 0 or i + 1 == len(todo):
            print(f"  {i + 1}/{len(todo)} participants, "
                  f"{fit_seconds_total / max(n_fitted, 1):.1f}s/fit avg, "
                  f"{time.time() - job_start:.0f}s elapsed", file=sys.stderr)

    if skipped:
        print(f"WARNING: skipped {len(skipped)} participants with too-short history: "
              f"{skipped[:5]}", file=sys.stderr)
        pd.DataFrame(skipped, columns=["person_id", "source_person_id", "history_bins"]).to_csv(
            f"{OUTPUT_DIR}/skipped_shard{SHARD}.csv", index=False)

    rows = []
    for horizon in HORIZON_BINS:
        path = f"{OUTPUT_DIR}/predictions_{horizon}_{CONDITION}.csv"
        if not os.path.exists(path):
            continue
        full = pd.read_csv(path, dtype={"window_id": str})
        scored = full.dropna(subset=["actual"])
        rows.append({
            "horizon": horizon, "split": SPLIT, "condition": CONDITION,
            "history_days": HISTORY_DAYS, "shuffled_donor": SHUFFLE_DONOR,
            "finetune_mode": FINETUNE_MODE, "learning_rate": LEARNING_RATE,
            "num_steps": NUM_STEPS, "batch_size": BATCH_SIZE,
            # both recorded: batch_size is windows/step (comparable across
            # arms), fit_batch_size is what fit() actually received
            "fit_batch_size": FIT_BATCH_SIZE, "n_variates": N_VARIATES,
            "past_only_channels": "|".join(PAST_ONLY_CHANNELS),
            "known_future_channels": "|".join(KNOWN_FUTURE_CHANNELS),
            "lora_r": LORA_R if FINETUNE_MODE == "lora" else None,
            "fit_prediction_length": FIT_PREDICTION_LENGTH, "seed": SEED,
            "mae_mgdl": scored["abs_error"].mean(),
            "rmse_mgdl": float(np.sqrt((scored["abs_error"] ** 2).mean())),
            "n_scored": len(scored), "n_total": len(full),
            "n_windows": full["window_id"].nunique(),
            "n_participants_this_shard": n_fitted,
            "mean_fit_seconds": round(fit_seconds_total / max(n_fitted, 1), 1),
            "model_load_seconds": model_load_seconds,
            "cumulative_job_seconds": round(time.time() - job_start, 1),
        })
    if rows:
        summary = pd.DataFrame(rows)
        summary.to_csv(f"{OUTPUT_DIR}/summary_shard{SHARD}.csv", index=False)
        print(summary.to_string(index=False), file=sys.stderr)


if __name__ == "__main__":
    main()
