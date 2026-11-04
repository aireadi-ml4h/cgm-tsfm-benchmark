"""Chronos-2 zero-shot evaluation over model_input/zeroshot_eval/*.csv.

For each (horizon, covariate condition) file: splits every window into a
context dataframe (is_context=True rows -> Chronos-2's `df`) and, for the
with_covariates condition, a future dataframe (is_context=False rows,
oxygen_saturation_applicable only -> Chronos-2's `future_df`). Calls
BaseChronosPipeline.predict_df in chunks of CHUNK_SIZE windows at a time
(not one call per combo) and appends each chunk's scored predictions to
predictions_<horizon>_<condition>.csv immediately, so a timeout mid-combo
loses at most one chunk, not the whole file. Rerunning the same job resumes:
window_ids already present in that combo's output file are skipped, not
recomputed, since predict_df is deterministic given a fixed context/model
(no sampling here -- point forecast only).

Two NaN-handling decisions, deliberately different, not an oversight:
- Context cgm_glucose (and covariates) is passed through UNMODIFIED,
  including its rare residual NaN (~0.1% of rows) -- Chronos-2's own
  preprocessing keeps NaN in the target array (na_value=np.nan in
  chronos2/preprocess.py) rather than rejecting it, and never filling this
  ourselves avoids the context/horizon leakage risk discussed for this
  project's earlier (reverted) interpolation attempt.
- Horizon cgm_glucose NaN (~0.01%) is never seen by the model at all --
  predict_df has no input path for it -- so it is handled only at scoring
  time, via dropna() before computing MAE/RMSE, not anywhere upstream.

oxygen_saturation_applicable goes in future_df, not df: it is a calendar
fact (is this bin inside the participant's timezone-specific overnight
window) knowable in advance with certainty, unlike the physiological
covariates, which stay past-only in df and are never given to future_df.

sleep_stage (added 2026-08-28, one of SSM-CGM's own time_varying_unknown_
categoricals -- see build_ssmcgm_input.py) is opt-in via
CHRONOS2_INCLUDE_SLEEP_STAGE=1, OFF by default: the existing with_covariates
baseline (chronos2_zeroshot_results/full/imputed) is already reported/
compared against SSM-CGM in report.md, so adding a new channel to its
default feature set would silently redefine an already-published number
instead of producing a new, separately comparable arm. When on, it is
past-only in df (like the physiological covariates, not
oxygen_saturation_applicable): a real deployment does not know a person's
future sleep stage, and export_zeroshot_windows.py already blanks it during
the horizon for this reason. Missing values (not currently in a recorded
sleep session -- the common case outside of sleep, see windows.py's
sleep_asleep note) are filled "unknown", matching build_ssmcgm_input.py's
own treatment, rather than left NaN: confirmed directly that Chronos-2's
predict_df accepts a string/categorical column in df without error, but an
explicit category is clearer than leaving it NaN and rediscovering later
whether NaN-as-categorical is silently doing something Chronos-2 does not
intend.

oxygen_saturation_applicable is independently ablatable (CHRONOS2_ABLATE_
CHANNELS=oxygen_saturation_applicable, on top of the existing per-channel
LOO mechanism), separate from ablating oxygen_saturation itself: SSM-CGM
lists it as its own time_varying_known_real, and the two questions --
"does the raw oxygen_saturation reading help" vs. "does knowing whether the
sensor is even in its collection window help" -- are distinct. Ablating
oxygen_saturation itself still drops both, unchanged from before (a model
that never sees the reading has no use for a flag about it).

predmeal_flag / predmeal_prob (added 2026-08-28): same opt-in pattern as
sleep_stage (CHRONOS2_INCLUDE_PREDMEAL_PROB=1, OFF by default). This is the
real, causally-generated meal probability (code/meal_detection/
generate_meal_probability.py), not the earlier disclosed placeholder --
past-only in df like the physiological covariates, never in future_df: its
horizon-period values would require glucose a real forecast doesn't have
yet, exactly the same reasoning as build_ssmcgm_input.py's decision to move
it to time_varying_unknown_real there. Missing values (near a real coverage
gap, or the first ~72 bins of any segment -- see generate_meal_probability.py)
are ffilled/bfilled within the window's own context, falling back to 0.0
only if the entire context is unresolvable, mirroring oxygen_saturation's
own fallback policy.

minute_of_day (added 2026-09-01, CHRONOS2_INCLUDE_MINUTE_OF_DAY=1, OFF by
default): completes the calendar block SSM-CGM actually uses -- its
TIME_VARYING_KNOWN_REALS carries minute_of_day AND tod_sin/tod_cos together,
so an arm claiming matched features needs all three. Its own flag rather
than part of CHRONOS2_INCLUDE_TOD; see the constant's comment for why.

CHRONOS2_INCLUDE_STATIC (age/glucose_center/glucose_scale/clinical_site/
study_group) is retained but should be considered a closed negative result,
NOT a route to static-feature parity with SSM-CGM, and is deliberately left
OUT of the ssmcgm_overlap arm (slurm/chronos2_zeroshot_eval_ssmcgm_overlap.
sbatch). The claim in ACTIVE_STATIC_CHANNELS' own comment below -- that a
constant-valued dynamic column is "indistinguishable in effect from a real
per-window static-covariate API" -- is wrong, verified against the vendored
pipeline source on 2026-09-01:
  - chronos2/dataset.py:184 stacks covariates into the BATCH dim
    ((n_targets + n_covariates, history_length)), and chronos2/model.py:408
    then instance-norms that tensor over dim=-1, i.e. per variate over time.
    For a constant column scale == 0, replaced by eps=1e-5
    (chronos_bolt.py:113), so scaled_x == (c - c)/1e-5 == 0.0 exactly. Every
    constant channel arrives at the model as all zeros; age=45 and age=80
    are bit-identical. The known-future half is normalized with the same
    variate's past loc_scale (model.py:454), so it zeroes too.
  - String statics are additionally destroyed before that: cast to
    "category", then _target_encode (chronos2/preprocess.py:612) is
    PER-ITEM, so a within-window-constant category encodes to
    (1*item_mean + 144*item_mean)/145 == the window's own mean context
    glucose. clinical_site/study_group therefore collapse onto
    glucose_center's value and carry no site/group identity at all.
Corroborated empirically: full/with_static's MAE differs from the
with_covariates baseline by -0.005 to -0.088 mg/dL across the five horizons
-- non-zero only because five all-zero variates still occupy attention
slots, not because any information transferred. age/clinical_site/
study_group/participant_id cannot reach Chronos-2 zero-shot by any path in
this API (predict_df/from_data_frame have no static parameter); that gap is
structural and is reported as such rather than papered over.
"""
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
from chronos import BaseChronosPipeline

REPO_ROOT = os.environ["AIREADI_ROOT"]
# CHRONOS2_EVAL_DIR_OVERRIDE / CHRONOS2_RESULTS_DIR_OVERRIDE: used for the
# non-LOCF wearables variant so its predictions land in their own directory,
# never overwriting the existing LOCF'd baseline results.
EVAL_DIR = os.environ.get("CHRONOS2_EVAL_DIR_OVERRIDE", f"{REPO_ROOT}/model_input/zeroshot_eval")
OUTPUT_DIR = os.environ.get("CHRONOS2_RESULTS_DIR_OVERRIDE", f"{REPO_ROOT}/chronos2_zeroshot_results/full/imputed")
CHUNK_SIZE = 500  # windows per predict_df call / per checkpoint

# 3h/4h added 2026-08-27, Chronos-2 only for now -- exploratory extension
# past report.md Sec 15.2's cited-literature horizons (30min/1h/2h), see
# code/CHRONOS2_EVAL.md.
HORIZON_BINS = {"30min": 6, "1h": 12, "2h": 24, "3h": 36, "4h": 48}
COVARIATE_FEATURE_CHANNELS = [
    "heart_rate", "stress", "respiratory_rate", "oxygen_saturation",
    "physical_activity_steps", "physical_activity_calorie_rate",
]
COVARIATE_APPLICABILITY_CHANNELS = ["oxygen_saturation_applicable"]

# CHRONOS2_ABLATE_CHANNELS: comma-separated leave-one-out channel(s) to drop
# from the with_covariates condition, for the per-model LOO ablation (which
# channels a stream-level correlation/importance check can only ever
# suggest, not settle -- the ablation is the actual test of what Chronos-2
# does with each signal). Window selection is untouched (same
# model_input/zeroshot_eval/*.csv, same window_index.csv) -- this only
# controls which columns get fed into predict_df's df/future_df, so every
# ablation arm scores on the identical window set as the full-covariate
# baseline. Dropping oxygen_saturation also drops oxygen_saturation_applicable:
# that flag disambiguates "not applicable" from "missing" for a value that,
# in this arm, isn't being given to the model at all, so keeping it in would
# feed a dangling calendar fact about a channel the model never sees.
ABLATE_CHANNELS = [c.strip() for c in os.environ.get("CHRONOS2_ABLATE_CHANNELS", "").split(",") if c.strip()]
# "oxygen_saturation_reading_only": completes the 2x2 (reading x flag) design
# alongside the existing "oxygen_saturation" (drops both) and
# "oxygen_saturation_applicable" (drops flag, keeps reading) arms -- drops
# ONLY the raw reading, keeping the flag active. Without this cell, "drop
# flag alone looks the same as drop both" is ambiguous between "the flag is
# the real problem" and "the two are just collinear, so removing either one
# alone barely moves a linear covariate fit" -- can't tell which from the
# other two arms alone.
ACTIVE_COVARIATE_FEATURE_CHANNELS = [
    c for c in COVARIATE_FEATURE_CHANNELS
    if c not in ABLATE_CHANNELS and not (c == "oxygen_saturation" and "oxygen_saturation_reading_only" in ABLATE_CHANNELS)
]
ACTIVE_COVARIATE_APPLICABILITY_CHANNELS = (
    [] if ("oxygen_saturation" in ABLATE_CHANNELS or "oxygen_saturation_applicable" in ABLATE_CHANNELS)
    else COVARIATE_APPLICABILITY_CHANNELS
)

# CHRONOS2_INCLUDE_SLEEP_STAGE: opt-in extra channel, see module docstring.
SLEEP_STAGE_CHANNEL = "sleep_stage"
INCLUDE_SLEEP_STAGE = os.environ.get("CHRONOS2_INCLUDE_SLEEP_STAGE", "0") == "1"
ACTIVE_SLEEP_STAGE_CHANNELS = (
    [] if (not INCLUDE_SLEEP_STAGE or SLEEP_STAGE_CHANNEL in ABLATE_CHANNELS) else [SLEEP_STAGE_CHANNEL]
)

# CHRONOS2_INCLUDE_PREDMEAL_PROB: opt-in extra channel, see module docstring.
PREDMEAL_PROB_CHANNEL = "predmeal_prob"
INCLUDE_PREDMEAL_PROB = os.environ.get("CHRONOS2_INCLUDE_PREDMEAL_PROB", "0") == "1"
ACTIVE_PREDMEAL_PROB_CHANNELS = (
    [] if (not INCLUDE_PREDMEAL_PROB or PREDMEAL_PROB_CHANNEL in ABLATE_CHANNELS) else [PREDMEAL_PROB_CHANNEL]
)

# CHRONOS2_INCLUDE_TOD: opt-in time-of-day covariates, mirrors
# timesfm_zeroshot_eval.py's addition (see its module docstring for the
# SSM-CGM citation). Computed directly from bin_start, already present in
# every zeroshot_eval CSV. Treated like oxygen_saturation_applicable (real
# future value in future_df, not carried forward) since it's a calendar
# fact known with certainty in advance -- unlike TimesFM, Chronos-2 already
# receives real timestamps and may derive some periodicity internally, but
# this makes that signal explicit and ablatable rather than implicit.
TOD_SIN_CHANNEL, TOD_COS_CHANNEL = "tod_sin", "tod_cos"
INCLUDE_TOD = os.environ.get("CHRONOS2_INCLUDE_TOD", "0") == "1"
ACTIVE_TOD_CHANNELS = (
    [] if (not INCLUDE_TOD or TOD_SIN_CHANNEL in ABLATE_CHANNELS) else [TOD_SIN_CHANNEL, TOD_COS_CHANNEL]
)

# CHRONOS2_INCLUDE_MINUTE_OF_DAY: opt-in, and deliberately a SEPARATE flag
# from CHRONOS2_INCLUDE_TOD rather than folded into it, even though
# build_ssmcgm_input.py carries minute_of_day and tod_sin/tod_cos together in
# one TIME_VARYING_KNOWN_REALS list. The three are one scalar in three
# encodings -- (tod_sin, tod_cos) recovers minute_of_day exactly via atan2 --
# so bundling them would make "does the calendar block help" and "does the
# redundant third encoding of it help" the same untestable question. Kept
# separable so the ssmcgm_overlap arm can turn both on for feature parity
# while the LOO sweep can still drop this one alone.
#
# Known-future (real values in future_df), same as tod_sin/tod_cos: clock
# arithmetic, knowable with certainty in advance. Note this channel is
# instance-normalized per window like every other covariate, so a window
# straddling midnight sees the raw 1439 -> 0 discontinuity that the sin/cos
# pair exists to smooth over; that asymmetry is inherent to the encoding,
# not something to correct for here, since the point is to reproduce
# SSM-CGM's own feature set rather than to improve on it.
MINUTE_OF_DAY_CHANNEL = "minute_of_day"
INCLUDE_MINUTE_OF_DAY = os.environ.get("CHRONOS2_INCLUDE_MINUTE_OF_DAY", "0") == "1"
ACTIVE_MINUTE_OF_DAY_CHANNELS = (
    [] if (not INCLUDE_MINUTE_OF_DAY or MINUTE_OF_DAY_CHANNEL in ABLATE_CHANNELS) else [MINUTE_OF_DAY_CHANNEL]
)

# CHRONOS2_INCLUDE_SLEEP_REGULARITY: opt-in extra channel (added 2026-08-28,
# not an SSM-CGM paper variable -- a project addition). See
# build_participant_grid.compute_sleep_regularity_index for the real
# Phillips et al. 2017 definition and export_zeroshot_windows.py for why it
# is blanked in the horizon despite being technically knowable in advance.
# -200 fallback (fillna): the real Phillips scale is -100..+100, so -200 is
# unambiguous, matching this project's own sentinel-outside-real-range
# convention (oxygen_saturation's -1.0) rather than 0.0, which is itself a
# real, meaningful value on this particular scale ("as regular as chance").
SLEEP_REGULARITY_CHANNEL = "sleep_regularity_index"
INCLUDE_SLEEP_REGULARITY = os.environ.get("CHRONOS2_INCLUDE_SLEEP_REGULARITY", "0") == "1"
ACTIVE_SLEEP_REGULARITY_CHANNELS = (
    [] if (not INCLUDE_SLEEP_REGULARITY or SLEEP_REGULARITY_CHANNEL in ABLATE_CHANNELS) else [SLEEP_REGULARITY_CHANNEL]
)

# CHRONOS2_INCLUDE_STATIC: opt-in static covariates, mirrors
# timesfm_zeroshot_eval.py's addition (see its module docstring for the
# SSM-CGM citation and export_zeroshot_windows.py for how these are
# computed). Chronos-2's predict_df has no dedicated static-covariate
# parameter (confirmed directly in chronos2/pipeline.py's signature) --
# unlike TimesFM, so these are just included as ordinary df/future_df
# columns.
#
# CORRECTION (2026-09-01): this comment previously claimed the
# constant-valued-column approach was "indistinguishable in effect from a
# real per-window static-covariate API." That is false -- instance
# normalization zeroes every constant channel, and per-item target encoding
# separately collapses the string statics onto the window mean glucose. See
# this module's docstring for the line-by-line trace and the empirical
# corroboration. This flag is kept only so the negative result stays
# reproducible; do not use it to claim static-feature parity with SSM-CGM.
STATIC_NUMERICAL_CHANNELS = ["age", "glucose_center", "glucose_scale"]
STATIC_CATEGORICAL_CHANNELS = ["clinical_site", "study_group"]
INCLUDE_STATIC = os.environ.get("CHRONOS2_INCLUDE_STATIC", "0") == "1"
ACTIVE_STATIC_CHANNELS = (
    [] if not INCLUDE_STATIC
    else [c for c in STATIC_NUMERICAL_CHANNELS + STATIC_CATEGORICAL_CHANNELS if c not in ABLATE_CHANNELS]
)


def build_context_future_truth(df, with_covariates):
    if ACTIVE_TOD_CHANNELS or ACTIVE_MINUTE_OF_DAY_CHANNELS:
        minute_of_day = df["bin_start"].dt.hour * 60 + df["bin_start"].dt.minute
        tod = minute_of_day / 1440.0
        # All three derived unconditionally once either flag is on -- which of
        # them actually reaches the model is decided by the ACTIVE_* lists
        # below, not here. float, matching build_ssmcgm_input.py's own
        # .astype(float) rather than leaving it an int dtype that
        # from_data_frame would cast anyway.
        df = df.assign(minute_of_day=minute_of_day.astype(float),
                       tod_sin=np.sin(2 * np.pi * tod), tod_cos=np.cos(2 * np.pi * tod))

    context_cols = ["window_id", "bin_start", "cgm_glucose"]
    if with_covariates:
        # oxygen_saturation_applicable must appear in df too, not just
        # future_df -- predict_df's validate_df rejects any future_df column
        # absent from df ("future_df cannot contain columns not present in
        # df"), caught by the smoke test. Its context-period values are the
        # same kind of calendar fact as its horizon-period ones, so this is
        # a correct inclusion, not a workaround.
        context_cols += (ACTIVE_COVARIATE_FEATURE_CHANNELS + ACTIVE_COVARIATE_APPLICABILITY_CHANNELS
                          + ACTIVE_SLEEP_STAGE_CHANNELS + ACTIVE_PREDMEAL_PROB_CHANNELS + ACTIVE_TOD_CHANNELS
                          + ACTIVE_MINUTE_OF_DAY_CHANNELS
                          + ACTIVE_SLEEP_REGULARITY_CHANNELS + ACTIVE_STATIC_CHANNELS)
    context = df.loc[df["is_context"], context_cols].reset_index(drop=True)
    if with_covariates and ACTIVE_SLEEP_STAGE_CHANNELS:
        context[SLEEP_STAGE_CHANNEL] = context[SLEEP_STAGE_CHANNEL].fillna("unknown")
    # FIXED 2026-09-02: both of these previously called
    # .ffill().bfill().fillna(...) on the whole CHUNK_SIZE=500-window
    # `context` frame at once, not per window -- so a window whose context
    # opened with NaN inherited the PRECEDING window's trailing value, which
    # is frequently a different participant. Measured on
    # zeroshot_1h_with_covariates.csv before the fix: predmeal_prob had 103
    # of 1,792 windows with a leading-NaN context, 49 of them (2.7%) preceded
    # by a different participant; sleep_regularity_index had 238, 163 of them
    # (9.1%) cross-participant. This also contradicted this module's own
    # docstring, which already claimed the fill happened "within the window's
    # own context" -- the docstring described the intent, the code did
    # something else.
    #
    # groupby(window_id).transform keeps the fill strictly inside each
    # window's own context rows. `context` is already filtered to
    # is_context=True, so bfill can only pull from a LATER CONTEXT bin, never
    # across the context/horizon boundary -- the rule report.md Sec 15.7's
    # interpolation revert established. fillna(...) still applies per window
    # when a window's entire context is NaN.
    #
    # Every arm run before this date used the chunk-level fill; those results
    # are not comparable to post-fix results for any arm carrying either
    # channel (see CHRONOS2_EVAL.md's arm inventory for which).
    if with_covariates and ACTIVE_PREDMEAL_PROB_CHANNELS:
        context[PREDMEAL_PROB_CHANNEL] = (
            context.groupby("window_id")[PREDMEAL_PROB_CHANNEL]
            .transform(lambda s: s.ffill().bfill())
            .fillna(0.0)
        )
    if with_covariates and ACTIVE_SLEEP_REGULARITY_CHANNELS:
        context[SLEEP_REGULARITY_CHANNEL] = (
            context.groupby("window_id")[SLEEP_REGULARITY_CHANNEL]
            .transform(lambda s: s.ffill().bfill())
            .fillna(-200.0)
        )

    horizon = df.loc[~df["is_context"]]
    future_df = None
    if with_covariates:
        future_df = horizon[["window_id", "bin_start"] + ACTIVE_COVARIATE_APPLICABILITY_CHANNELS
                             + ACTIVE_TOD_CHANNELS + ACTIVE_MINUTE_OF_DAY_CHANNELS
                             + ACTIVE_STATIC_CHANNELS].reset_index(drop=True)

    truth = horizon[["window_id", "bin_start", "cgm_glucose"]].rename(
        columns={"cgm_glucose": "actual"}
    ).reset_index(drop=True)
    return context, future_df, truth


def already_done_window_ids(output_path):
    if not os.path.exists(output_path):
        return set()
    return set(pd.read_csv(output_path, usecols=["window_id"], dtype={"window_id": str})["window_id"])


def run_chunk(pipeline, df_chunk, horizon_bins, with_covariates):
    context, future_df, truth = build_context_future_truth(df_chunk, with_covariates)
    forecast = pipeline.predict_df(
        df=context,
        future_df=future_df,
        id_column="window_id",
        timestamp_column="bin_start",
        target="cgm_glucose",
        prediction_length=horizon_bins,
        freq="5min",
    )
    merged = forecast.merge(truth, on=["window_id", "bin_start"], how="left")
    merged["abs_error"] = (merged["predictions"] - merged["actual"]).abs()
    return merged


def run_one(pipeline, horizon_name, horizon_bins, with_covariates, max_windows=None):
    combo_start = time.time()
    condition = "with_covariates" if with_covariates else "glucose_only"
    path = f"{EVAL_DIR}/zeroshot_{horizon_name}_{condition}.csv"
    output_path = f"{OUTPUT_DIR}/predictions_{horizon_name}_{condition}.csv"
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    df = pd.read_csv(path, dtype={"window_id": str, "person_id": str})
    # parsed as UTC then made tz-naive: predict_df's normalize_df calls
    # .to_numpy().view("int64") on this column, which only works on a plain
    # datetime64 array. A tz-aware column's .to_numpy() returns an object
    # array of per-element Timestamps instead, which .view() can't
    # reinterpret -- caught by the smoke test (TypeError: Cannot change
    # data-type for array of references). Everything here is already UTC,
    # so dropping the tz loses no information.
    df["bin_start"] = pd.to_datetime(df["bin_start"], utc=True).dt.tz_localize(None)
    df["is_context"] = df["is_context"].astype(bool)

    all_window_ids = df["window_id"].unique()
    if max_windows is not None:
        all_window_ids = all_window_ids[:max_windows]
        df = df[df["window_id"].isin(all_window_ids)]
    done = already_done_window_ids(output_path)
    remaining = [w for w in all_window_ids if w not in done]

    if done:
        print(f"{horizon_name} {condition}: resuming, {len(done)}/{len(all_window_ids)} windows already done",
              file=sys.stderr)

    for start in range(0, len(remaining), CHUNK_SIZE):
        chunk_ids = set(remaining[start:start + CHUNK_SIZE])
        df_chunk = df[df["window_id"].isin(chunk_ids)]
        merged = run_chunk(pipeline, df_chunk, horizon_bins, with_covariates)
        header = not os.path.exists(output_path)
        merged.to_csv(output_path, mode="a", header=header, index=False)
        print(f"{horizon_name} {condition}: checkpointed {start + len(chunk_ids)}/{len(remaining)} "
              f"remaining windows ({len(done) + start + len(chunk_ids)}/{len(all_window_ids)} total)",
              file=sys.stderr)

    # Score from the full, now-complete output file, not just this run's chunks,
    # so a resumed run's summary row reflects every window, not only the ones
    # processed in this particular process invocation.
    full = pd.read_csv(output_path, dtype={"window_id": str})
    scored = full.dropna(subset=["actual"])
    n_total, n_scored = len(full), len(scored)
    mae = scored["abs_error"].mean()
    rmse = float(np.sqrt((scored["abs_error"] ** 2).mean()))
    elapsed = time.time() - combo_start
    print(
        f"{horizon_name} {condition}: DONE MAE={mae:.3f} mg/dL RMSE={rmse:.3f} mg/dL "
        f"(scored {n_scored}/{n_total} horizon bins, {n_total - n_scored} skipped for missing ground truth) "
        f"in {elapsed:.1f}s this run ({len(remaining)} windows newly processed, {len(done)} already done)",
        file=sys.stderr,
    )
    return {
        "horizon": horizon_name, "with_covariates": with_covariates,
        "mae_mgdl": mae, "rmse_mgdl": rmse,
        "n_scored": n_scored, "n_total": n_total,
        "elapsed_seconds_this_run": round(elapsed, 1),
        "n_windows_processed_this_run": len(remaining),
    }


def main():
    # CHRONOS2_MAX_WINDOWS / CHRONOS2_SMOKE_COMBOS: set for a cheap sanity
    # run before committing to the full population (see slurm/*_smoke.sbatch).
    max_windows = os.environ.get("CHRONOS2_MAX_WINDOWS")
    max_windows = int(max_windows) if max_windows else None
    smoke_combos = os.environ.get("CHRONOS2_SMOKE_COMBOS")  # e.g. "1" -> only the first combo

    job_start = time.time()
    pipeline = BaseChronosPipeline.from_pretrained(
        "amazon/chronos-2", device_map="cuda" if torch.cuda.is_available() else "cpu"
    )
    model_load_seconds = round(time.time() - job_start, 1)
    print(f"model loaded in {model_load_seconds}s", file=sys.stderr)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    summary_path = f"{OUTPUT_DIR}/summary.csv"

    # CHRONOS2_ONLY_WITH_COVARIATES: for LOO ablation runs -- glucose_only has
    # no covariates to ablate, so it's identical across every arm and not
    # worth recomputing.
    only_with_covariates = os.environ.get("CHRONOS2_ONLY_WITH_COVARIATES", "0") == "1"
    covariate_conditions = (True,) if only_with_covariates else (False, True)
    combos = [(h, b, c) for h, b in HORIZON_BINS.items() for c in covariate_conditions]
    if smoke_combos:
        combos = combos[:int(smoke_combos)]

    rows = []
    for horizon_name, horizon_bins, with_covariates in combos:
        row = run_one(pipeline, horizon_name, horizon_bins, with_covariates, max_windows)
        row["model_load_seconds"] = model_load_seconds
        row["cumulative_job_seconds"] = round(time.time() - job_start, 1)
        rows.append(row)
        # rewritten after every combo finishes, not just at the end, so a
        # summary of whatever's done so far always exists on disk, including
        # timing, if the job stops partway through.
        pd.DataFrame(rows).to_csv(summary_path, index=False)

    print(pd.DataFrame(rows).to_string(index=False), file=sys.stderr)


if __name__ == "__main__":
    main()
