"""TimesFM 2.5 zero-shot evaluation over model_input/zeroshot_eval/*.csv.

Parallels code/chronos2/chronos2_zeroshot_eval.py exactly where the two models' APIs
allow it (same window files, same horizons, same scoring/chunking/resume
pattern, same ablation mechanism) and departs from it only where TimesFM's
actual API forces a different mechanism -- each departure is a real API
constraint, confirmed empirically before writing this, not a convenience.

TWO REAL API DIFFERENCES FROM CHRONOS-2, both confirmed by hand before this
script was written, not assumed from documentation:

1. TimesFM has no single call that takes a target dataframe plus an optional
   future-covariates dataframe. Glucose-only uses `model.forecast(horizon,
   inputs=[...])`, a plain list of context arrays. With-covariates uses the
   separate `model.forecast_with_covariates(inputs=[...],
   dynamic_numerical_covariates={...})` method, which additionally requires
   `timesfm[xreg]` (`jax`, `scikit-learn`) -- installed here via the cluster's
   own `py-jax/0.8.1_py314`/`py-scikit-learn/1.8.0_py314` modules, not pip:
   a `pip install jax[cuda]` attempt in this Python 3.14 environment
   backtracked all the way to jax 0.2.22 (2021, predates the `ml_dtypes`
   dependency entirely, silently dropped the `cuda` extra) because
   `ml_dtypes` has no prebuilt wheel for cp314 and failed to build from
   source; the module-provided jax is a real, already-built cp314 package
   and avoids this entirely. jax's own CUDA plugin still fails to
   initialize under the cluster's shared-node CUDA context (a jax-specific,
   not project-specific, limitation) and falls back to CPU for the xreg
   linear-regression step only -- the main forecast still runs on GPU via
   PyTorch, and xreg's ridge regression is cheap enough on CPU that this
   does not meaningfully affect runtime.

2. `dynamic_numerical_covariates` must be supplied at context+horizon
   length, confirmed directly (a context-only-length array raises
   `ValueError: expected a positive input`) -- unlike Chronos-2's df/
   future_df split, there is no way to tell TimesFM "this covariate's
   future is unknown." The horizon portion is filled with the last real
   (LOCF-filled) context value carried forward, exactly the "shifted/
   repeated if past-only" approach documented for this case in `Model
   Context and Horizon Lengths.xlsx` before this project started using
   TimesFM -- a flat placeholder, not real future data, so this does not
   reintroduce the future-covariate leakage this project's whole design
   avoids elsewhere (chronos2_zeroshot_eval.py's future_df, SSM-CGM's
   oracle fix). `oxygen_saturation_applicable` is the one exception, same
   as Chronos-2's treatment of it: its real future value is used instead of
   a carried-forward placeholder, since it is a calendar fact knowable in
   advance, not a physiological reading.

   `dynamic_numerical_covariates` additionally rejects NaN outright
   (confirmed directly: raises `ValueError: The truth value of an empty
   array is ambiguous`), unlike TimesFM's own `forecast()` context input
   (confirmed tolerant of NaN, matching Chronos-2's target handling) and
   unlike Chronos-2's covariates (also NaN-tolerant). Covariate columns are
   therefore forward- then backward-filled within each window's own 144-bin
   context before use, and any column still entirely NaN after that (the
   same structural gap already found and fixed in SSM-CGM's
   `build_ssmcgm_input.py`: a window can be fully outside oxygen_saturation's
   overnight-only applicable period) falls back to 0.0, outside the real
   [50,100] range. `cgm_glucose` itself is never filled, in either
   condition, for either model: fabricating the forecasting target would
   misrepresent what is actually being forecast, and TimesFM's `forecast()`/
   `forecast_with_covariates()` target/context path both tolerate its rare
   residual NaN natively (confirmed directly), so there is no API reason to
   fill it the way there is for the covariates.

sleep_stage (added 2026-08-28, mirrors chronos2_zeroshot_eval.py's addition):
opt-in via TIMESFM_INCLUDE_SLEEP_STAGE=1, OFF by default, for the same
reason as Chronos-2 -- the existing with_covariates baseline
(timesfm_zeroshot_results/full/imputed) is already reported and should not
be silently redefined. Passed via `dynamic_categorical_covariates` (a
separate parameter from `dynamic_numerical_covariates`, confirmed directly
to accept a string-valued context+horizon list per window, same shape
convention as the numerical covariates). Missing values filled "unknown"
(build_ssmcgm_input.py's own convention); horizon filled with the last
context value carried forward, same flat-placeholder treatment as the
other physiological (non-applicability) covariates -- a real deployment
does not know a person's future sleep stage either.

oxygen_saturation_applicable is independently ablatable
(TIMESFM_ABLATE_CHANNELS=oxygen_saturation_applicable), separate from
ablating oxygen_saturation itself -- same reasoning as Chronos-2's version.

predmeal_flag / predmeal_prob (added 2026-08-28): same opt-in pattern
(TIMESFM_INCLUDE_PREDMEAL_PROB=1, OFF by default), same as sleep_stage --
past-only (fed via dynamic_numerical_covariates, horizon filled with the
last context value carried forward, same flat-placeholder treatment as the
other physiological covariates, NOT the real future value the way
oxygen_saturation_applicable gets). See chronos2_zeroshot_eval.py's/
build_ssmcgm_input.py's docstrings for why this is a real signal now, not
the earlier placeholder, and why it is past-only rather than known-in-advance
despite its own generating function being causal relative to its own
timestamp.
"""
import os
import sys
import time

import numpy as np
import pandas as pd
import timesfm

REPO_ROOT = os.environ["AIREADI_ROOT"]
EVAL_DIR = os.environ.get("TIMESFM_EVAL_DIR_OVERRIDE", f"{REPO_ROOT}/model_input/zeroshot_eval")
OUTPUT_DIR = os.environ.get("TIMESFM_RESULTS_DIR_OVERRIDE", f"{REPO_ROOT}/timesfm_zeroshot_results/full/imputed")
CHUNK_SIZE = 500  # windows per forecast()/forecast_with_covariates() call / per checkpoint
CONTEXT_LEN = 144

# Same horizons as code/chronos2/chronos2_zeroshot_eval.py, all 5 -- report.md Sec
# 15.2's 3 cited-literature horizons (30min/1h/2h) plus the 3h/4h
# exploratory extension, matched here per instruction to cover every
# horizon Chronos-2 was run at, not just the original 3.
HORIZON_BINS = {"30min": 6, "1h": 12, "2h": 24, "3h": 36, "4h": 48}
COVARIATE_FEATURE_CHANNELS = [
    "heart_rate", "stress", "respiratory_rate", "oxygen_saturation",
    "physical_activity_steps", "physical_activity_calorie_rate",
]
COVARIATE_APPLICABILITY_CHANNELS = ["oxygen_saturation_applicable"]

# TIMESFM_ABLATE_CHANNELS: same leave-one-out mechanism as
# CHRONOS2_ABLATE_CHANNELS, for parity with Chronos-2's ablation runs
# (e.g. dropping oxygen_saturation, per code/CHRONOS2_EVAL.md's finding).
ABLATE_CHANNELS = [c.strip() for c in os.environ.get("TIMESFM_ABLATE_CHANNELS", "").split(",") if c.strip()]
# "oxygen_saturation_reading_only": completes the 2x2 (reading x flag)
# design, same reasoning as chronos2_zeroshot_eval.py's version -- drops
# ONLY the raw reading, keeping oxygen_saturation_applicable active.
ACTIVE_COVARIATE_FEATURE_CHANNELS = [
    c for c in COVARIATE_FEATURE_CHANNELS
    if c not in ABLATE_CHANNELS and not (c == "oxygen_saturation" and "oxygen_saturation_reading_only" in ABLATE_CHANNELS)
]
ACTIVE_COVARIATE_APPLICABILITY_CHANNELS = (
    [] if ("oxygen_saturation" in ABLATE_CHANNELS or "oxygen_saturation_applicable" in ABLATE_CHANNELS)
    else COVARIATE_APPLICABILITY_CHANNELS
)

# TIMESFM_INCLUDE_SLEEP_STAGE: opt-in extra channel, see module docstring.
SLEEP_STAGE_CHANNEL = "sleep_stage"
INCLUDE_SLEEP_STAGE = os.environ.get("TIMESFM_INCLUDE_SLEEP_STAGE", "0") == "1"
ACTIVE_SLEEP_STAGE_CHANNELS = (
    [] if (not INCLUDE_SLEEP_STAGE or SLEEP_STAGE_CHANNEL in ABLATE_CHANNELS) else [SLEEP_STAGE_CHANNEL]
)

# TIMESFM_INCLUDE_PREDMEAL_PROB: opt-in extra channel, see module docstring.
PREDMEAL_PROB_CHANNEL = "predmeal_prob"
INCLUDE_PREDMEAL_PROB = os.environ.get("TIMESFM_INCLUDE_PREDMEAL_PROB", "0") == "1"
ACTIVE_PREDMEAL_PROB_CHANNELS = (
    [] if (not INCLUDE_PREDMEAL_PROB or PREDMEAL_PROB_CHANNEL in ABLATE_CHANNELS) else [PREDMEAL_PROB_CHANNEL]
)

# TIMESFM_INCLUDE_TOD: opt-in time-of-day covariates (added 2026-08-28,
# matching SSM-CGM's own "temporal encodings: minute-of-day and 24h
# sine/cosine" feature group, arXiv:2510.04386 Sec 2). Computed directly
# from each window's own bin_start -- already present in every zeroshot_eval
# CSV -- no new precomputation pipeline needed, unlike sleep_stage/
# predmeal_prob. Unlike the physiological covariates, this is a genuine
# calendar fact known with certainty at every future timestep (same
# treatment as oxygen_saturation_applicable): real future values are used,
# not a carried-forward placeholder. This is the single biggest gap TimesFM
# had of any SSM-CGM covariate -- unlike Chronos-2 (which receives real
# `bin_start` timestamps and can derive periodicity from them internally),
# TimesFM's forecast()/forecast_with_covariates() calls here pass bare
# arrays with no timestamp at all, so before this addition TimesFM had zero
# time-of-day signal in any condition.
TOD_SIN_CHANNEL, TOD_COS_CHANNEL = "tod_sin", "tod_cos"
INCLUDE_TOD = os.environ.get("TIMESFM_INCLUDE_TOD", "0") == "1"
ACTIVE_TOD_CHANNELS = (
    [] if (not INCLUDE_TOD or TOD_SIN_CHANNEL in ABLATE_CHANNELS) else [TOD_SIN_CHANNEL, TOD_COS_CHANNEL]
)

# TIMESFM_INCLUDE_SLEEP_REGULARITY: opt-in extra channel, mirrors
# chronos2_zeroshot_eval.py's addition -- see that module's docstring for
# the -200.0 fallback rationale and build_participant_grid.
# compute_sleep_regularity_index for the real Phillips et al. 2017 SRI
# definition.
SLEEP_REGULARITY_CHANNEL = "sleep_regularity_index"
INCLUDE_SLEEP_REGULARITY = os.environ.get("TIMESFM_INCLUDE_SLEEP_REGULARITY", "0") == "1"
ACTIVE_SLEEP_REGULARITY_CHANNELS = (
    [] if (not INCLUDE_SLEEP_REGULARITY or SLEEP_REGULARITY_CHANNEL in ABLATE_CHANNELS) else [SLEEP_REGULARITY_CHANNEL]
)

# TIMESFM_INCLUDE_STATIC: opt-in static covariates (added 2026-08-28),
# completing SSM-CGM's own static covariate list (arXiv:2510.04386 Sec 2:
# "age, diabetes status, clinical site, and participant ID") plus glucose_
# scale/glucose_center (their 2nd-most-important static feature, Table 5) --
# see export_zeroshot_windows.py for how these are computed (per-window,
# causal). Unlike every other covariate here, TimesFM's forecast_with_
# covariates has a REAL, dedicated static-covariate API
# (static_numerical_covariates/static_categorical_covariates, confirmed
# directly in timesfm_2p5_base.py's signature) -- one value per window, not
# per-timestep, so no context/horizon or carried-forward logic applies.
STATIC_NUMERICAL_CHANNELS = ["age", "glucose_center", "glucose_scale"]
STATIC_CATEGORICAL_CHANNELS = ["clinical_site", "study_group"]
INCLUDE_STATIC = os.environ.get("TIMESFM_INCLUDE_STATIC", "0") == "1"
ACTIVE_STATIC_NUMERICAL_CHANNELS = (
    [] if not INCLUDE_STATIC else [c for c in STATIC_NUMERICAL_CHANNELS if c not in ABLATE_CHANNELS]
)
ACTIVE_STATIC_CATEGORICAL_CHANNELS = (
    [] if not INCLUDE_STATIC else [c for c in STATIC_CATEGORICAL_CHANNELS if c not in ABLATE_CHANNELS]
)


def build_covariate_series(window_df, horizon_bins, channel, is_applicability, fallback=0.0):
    """Returns one context+horizon-length list for `channel` in this window:
    real (LOCF-filled) context values, then either the channel's real future
    value (if is_applicability -- a calendar fact, safe to know in advance)
    or the last filled context value carried forward (a flat placeholder,
    not real future data, for every physiological covariate).

    `fallback` (used only if the whole window has zero real coverage for
    this channel) defaults to 0.0, correct for the wearable channels (a
    real, out-of-band value for e.g. a fully-missing oxygen_saturation
    window) but wrong for sleep_regularity_index, whose real scale runs
    -100..+100 and where 0.0 is itself a meaningful reading -- see
    chronos2_zeroshot_eval.py's docstring for the -200.0 override used
    there instead."""
    context_vals = window_df.loc[window_df["is_context"], channel].ffill().bfill()
    if context_vals.isna().all():
        context_vals = context_vals.fillna(fallback)
    context_vals = context_vals.tolist()
    if is_applicability:
        horizon_vals = window_df.loc[~window_df["is_context"], channel].fillna(0.0).tolist()
    else:
        horizon_vals = [context_vals[-1]] * horizon_bins
    return context_vals + horizon_vals


def build_sleep_stage_series(window_df, horizon_bins):
    """Categorical counterpart to build_covariate_series: past-only, filled
    "unknown" where not currently in a recorded sleep session (see module
    docstring), horizon carried forward from the last context value -- same
    flat-placeholder treatment as the other physiological covariates."""
    context_vals = window_df.loc[window_df["is_context"], SLEEP_STAGE_CHANNEL].fillna("unknown").tolist()
    horizon_vals = [context_vals[-1]] * horizon_bins
    return context_vals + horizon_vals


def build_tod_series(window_df):
    """tod_sin/tod_cos for every bin in the window (context AND horizon --
    both real, since time-of-day is a calendar fact known with certainty in
    advance, same as oxygen_saturation_applicable)."""
    minute_of_day = window_df["bin_start"].dt.hour * 60 + window_df["bin_start"].dt.minute
    tod = minute_of_day / 1440.0
    return (2 * np.pi * tod).apply(np.sin).tolist(), (2 * np.pi * tod).apply(np.cos).tolist()


def build_inputs_and_truth(df, horizon_bins, with_covariates):
    window_ids = df["window_id"].unique().tolist()
    inputs = []
    dyn_covs = {c: [] for c in ACTIVE_COVARIATE_FEATURE_CHANNELS + ACTIVE_COVARIATE_APPLICABILITY_CHANNELS
                + ACTIVE_PREDMEAL_PROB_CHANNELS + ACTIVE_TOD_CHANNELS + ACTIVE_SLEEP_REGULARITY_CHANNELS}
    dyn_cat_covs = {c: [] for c in ACTIVE_SLEEP_STAGE_CHANNELS}
    static_num_covs = {c: [] for c in ACTIVE_STATIC_NUMERICAL_CHANNELS}
    static_cat_covs = {c: [] for c in ACTIVE_STATIC_CATEGORICAL_CHANNELS}
    truths = []

    for wid in window_ids:
        w = df[df["window_id"] == wid].sort_values("bin_start")
        context_series = w.loc[w["is_context"], "cgm_glucose"]
        if with_covariates:
            # forecast_with_covariates's xreg fit does not tolerate NaN in the
            # target, confirmed directly: a single residual NaN in a
            # 144-length context (the rate already known from
            # chronos2_zeroshot_eval.py, ~0.1% of rows) turns the entire
            # 144-length fitted xreg_on_context into NaN, not just that one
            # position, which then poisons the internal residual forecast
            # step. Unlike glucose_only's forecast(), which internally
            # cleans NaN out of the context before ever reaching the
            # transformer (confirmed tolerant, matching Chronos-2's target
            # handling), this path has no such cleanup. Filled here (ffill/
            # bfill) ONLY for this condition and ONLY in the context, never
            # in the scored horizon truth -- glucose_only's context is left
            # raw, matching this project's default of not fabricating a
            # value the API doesn't actually require filled.
            context = context_series.ffill().bfill().to_numpy(dtype=float)
        else:
            context = context_series.to_numpy(dtype=float)
        inputs.append(context)

        horizon = w.loc[~w["is_context"]]
        for i, ts in enumerate(horizon["bin_start"]):
            truths.append({"window_id": wid, "bin_start": ts, "actual": horizon["cgm_glucose"].iloc[i]})

        if with_covariates:
            for channel in ACTIVE_COVARIATE_FEATURE_CHANNELS:
                dyn_covs[channel].append(build_covariate_series(w, horizon_bins, channel, is_applicability=False))
            for channel in ACTIVE_COVARIATE_APPLICABILITY_CHANNELS:
                dyn_covs[channel].append(build_covariate_series(w, horizon_bins, channel, is_applicability=True))
            for channel in ACTIVE_PREDMEAL_PROB_CHANNELS:
                dyn_covs[channel].append(build_covariate_series(w, horizon_bins, channel, is_applicability=False))
            for channel in ACTIVE_SLEEP_STAGE_CHANNELS:
                dyn_cat_covs[channel].append(build_sleep_stage_series(w, horizon_bins))
            if ACTIVE_TOD_CHANNELS:
                sin_vals, cos_vals = build_tod_series(w)
                dyn_covs[TOD_SIN_CHANNEL].append(sin_vals)
                dyn_covs[TOD_COS_CHANNEL].append(cos_vals)
            for channel in ACTIVE_SLEEP_REGULARITY_CHANNELS:
                dyn_covs[channel].append(build_covariate_series(w, horizon_bins, channel, is_applicability=False, fallback=-200.0))
            for channel in ACTIVE_STATIC_NUMERICAL_CHANNELS:
                static_num_covs[channel].append(float(w[channel].iloc[0]))
            for channel in ACTIVE_STATIC_CATEGORICAL_CHANNELS:
                static_cat_covs[channel].append(str(w[channel].iloc[0]))

    truth = pd.DataFrame(truths)
    return window_ids, inputs, dyn_covs, dyn_cat_covs, static_num_covs, static_cat_covs, truth


def already_done_window_ids(output_path):
    if not os.path.exists(output_path):
        return set()
    return set(pd.read_csv(output_path, usecols=["window_id"], dtype={"window_id": str})["window_id"])


def run_chunk(model, df_chunk, horizon_bins, with_covariates):
    window_ids, inputs, dyn_covs, dyn_cat_covs, static_num_covs, static_cat_covs, truth = build_inputs_and_truth(
        df_chunk, horizon_bins, with_covariates)

    if with_covariates:
        point_forecast, _xreg = model.forecast_with_covariates(
            inputs=inputs, dynamic_numerical_covariates=dyn_covs,
            dynamic_categorical_covariates=dyn_cat_covs,
            static_numerical_covariates=static_num_covs,
            static_categorical_covariates=static_cat_covs,
        )
        point_forecast = np.asarray(point_forecast)
    else:
        point_forecast, _quantiles = model.forecast(horizon=horizon_bins, inputs=inputs)

    rows = []
    for i, wid in enumerate(window_ids):
        bin_starts = truth.loc[truth["window_id"] == wid, "bin_start"].tolist()
        for j, ts in enumerate(bin_starts):
            rows.append({"window_id": wid, "bin_start": ts, "predictions": float(point_forecast[i][j])})
    predictions = pd.DataFrame(rows)

    merged = predictions.merge(truth, on=["window_id", "bin_start"], how="left")
    merged["abs_error"] = (merged["predictions"] - merged["actual"]).abs()
    return merged


def run_one(model, horizon_name, horizon_bins, with_covariates, max_windows=None):
    combo_start = time.time()
    condition = "with_covariates" if with_covariates else "glucose_only"
    path = f"{EVAL_DIR}/zeroshot_{horizon_name}_{condition}.csv"
    output_path = f"{OUTPUT_DIR}/predictions_{horizon_name}_{condition}.csv"
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    df = pd.read_csv(path, dtype={"window_id": str, "person_id": str})
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
        merged = run_chunk(model, df_chunk, horizon_bins, with_covariates)
        header = not os.path.exists(output_path)
        merged.to_csv(output_path, mode="a", header=header, index=False)
        print(f"{horizon_name} {condition}: checkpointed {start + len(chunk_ids)}/{len(remaining)} "
              f"remaining windows ({len(done) + start + len(chunk_ids)}/{len(all_window_ids)} total)",
              file=sys.stderr)

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
    max_windows = os.environ.get("TIMESFM_MAX_WINDOWS")
    max_windows = int(max_windows) if max_windows else None
    smoke_combos = os.environ.get("TIMESFM_SMOKE_COMBOS")
    only_with_covariates = os.environ.get("TIMESFM_ONLY_WITH_COVARIATES", "0") == "1"

    job_start = time.time()
    model = timesfm.TimesFM_2p5_200M_torch.from_pretrained("google/timesfm-2.5-200m-pytorch")
    model_load_seconds = round(time.time() - job_start, 1)
    print(f"model loaded in {model_load_seconds}s", file=sys.stderr)

    OUTPUT_DIR_ = OUTPUT_DIR
    os.makedirs(OUTPUT_DIR_, exist_ok=True)
    summary_path = f"{OUTPUT_DIR_}/summary.csv"

    covariate_conditions = (True,) if only_with_covariates else (False, True)
    combos = [(h, b, c) for h, b in HORIZON_BINS.items() for c in covariate_conditions]
    if smoke_combos:
        combos = combos[:int(smoke_combos)]

    # model.compile()'s output shape for the plain forecast() path only
    # matches the requested `horizon` when compiled WITHOUT return_backcast:
    # confirmed directly (compiling with return_backcast=True, then calling
    # forecast(horizon=6, ...), silently returns a degenerate (1, 998)-shape
    # constant-value array instead of (1, 6) -- a real library quirk, not a
    # usage error here, caught by the smoke test producing MAE=102.956
    # mg/dL, an impossible number for 30-minute-ahead glucose forecasting.
    # forecast_with_covariates() separately requires return_backcast=True
    # (raises ValueError otherwise). Recompiling on every condition switch
    # (cheap relative to the model's one-time from_pretrained() load) keeps
    # each call correct rather than picking one config and breaking the
    # other path.
    plain_config = timesfm.ForecastConfig(
        max_context=1024, max_horizon=256, normalize_inputs=True,
        use_continuous_quantile_head=True, fix_quantile_crossing=True,
    )
    covariate_config = timesfm.ForecastConfig(
        max_context=1024, max_horizon=256, normalize_inputs=True,
        use_continuous_quantile_head=True, fix_quantile_crossing=True,
        return_backcast=True,
    )
    compiled_for = None

    rows = []
    for horizon_name, horizon_bins, with_covariates in combos:
        if compiled_for != with_covariates:
            model.compile(covariate_config if with_covariates else plain_config)
            compiled_for = with_covariates
        row = run_one(model, horizon_name, horizon_bins, with_covariates, max_windows)
        row["model_load_seconds"] = model_load_seconds
        row["cumulative_job_seconds"] = round(time.time() - job_start, 1)
        rows.append(row)
        pd.DataFrame(rows).to_csv(summary_path, index=False)

    print(pd.DataFrame(rows).to_string(index=False), file=sys.stderr)


if __name__ == "__main__":
    main()
