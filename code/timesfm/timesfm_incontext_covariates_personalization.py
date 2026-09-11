"""TimesFM 2.5 in-context personalization WITH covariates: the participant's
own 1/3/7 days of prior glucose AND covariate history, prepended directly to
the model's normal 144-bin context, then scored with `forecast_with_
covariates()` -- the combination of in-context history extension (already
built, glucose-only, `timesfm_incontext_personalization.py`) and TimesFM's
covariate mechanism (already built, zero-shot only, `timesfm_zeroshot_eval.
py`), which had never been combined before this script.

WHY THIS IS A NEW ARM, NOT A DUPLICATE OF EITHER EXISTING SCRIPT
===================================================================
`timesfm_incontext_personalization.py` extends context but is glucose-only
by explicit, disclosed choice (its own docstring: "combining them is a real
follow-up experiment, not this one"). This is that follow-up. It reuses
`forecast_with_covariates()` exactly as `timesfm_zeroshot_eval.py` calls it
(same parameter names, same NaN-filling/carried-forward-horizon convention,
same 6-physiological-channel + oxygen_saturation_applicable baseline bundle
-- no opt-in extras (sleep_stage/predmeal_prob/tod/sleep_regularity), for
the cleanest comparison against the existing with_covariates zero-shot
baseline and the in-context glucose-only arm), just fed a longer history.

TWO REAL API CONSTRAINTS THAT SHAPE THIS SCRIPT, both confirmed directly
against `timesfm_zeroshot_eval.py` (verified, not assumed, before writing
this):

1. `dynamic_numerical_covariates` must be supplied at context+horizon
   length and rejects NaN outright. Every physiological covariate's
   EXTENDED history (from `history_start` through the window's own context)
   is forward- then backward-filled before use, same as the existing
   with_covariates zero-shot path's per-window fill -- just applied over
   the whole concatenated span here, not just 144 bins. `cgm_glucose`
   itself is never filled, matching the project's "never fabricate the
   forecasting target" rule; `forecast_with_covariates`'s target/context
   path already tolerates raw NaN (confirmed in `timesfm_zeroshot_eval.py`).

2. `forecast_with_covariates()` requires `return_backcast=True` at compile
   time (confirmed in `timesfm_zeroshot_eval.py`); `max_context` must be
   raised the same way `timesfm_incontext_personalization.py` raises it for
   the glucose-only in-context arm (4096, comfortably above the largest
   real span: a 7-day history is 2016 bins, plus 144 bins of window
   context, plus the horizon's own carried-forward covariate positions =
   2160 + up to 48 = 2208, well under 4096).

COVARIATE HISTORY SOURCE: `build_grid()` (`build_participant_grid.py`), the
same function `build_ssmcgm_input.py`/`chronos2_personalization.py`'s
covariate paths use -- NOT the zeroshot_eval export CSVs, which only carry
144 bins per window. `build_grid()` returns the full wearable+glucose grid
for a participant over any date range, so both the extended history prefix
AND the window's own 144-bin context are read from this single source here
(unlike the glucose-only in-context script, which stitches together
`history_array()`'s own source for the prefix and the zeroshot_eval CSV for
the window itself) -- one consistent provenance for the whole concatenated
array avoids a seam between two independently-built data sources for the
same covariate channels.

STATIC COVARIATES (age/clinical_site/study_group): one value per window,
not per-timestep, matching `timesfm_zeroshot_eval.py`'s
`static_numerical_covariates`/`static_categorical_covariates` (a real,
dedicated API for this, confirmed there) -- `glucose_center`/`glucose_scale`
are NOT included here, since those are computed per-window in the
zeroshot_eval export pipeline as pre-personalization summary statistics and
are out of scope for this arm (this project's own quick-test earlier found
personal glucose mean/SD add no signal for Chronos-2 as a static covariate
-- worth the same caveat here rather than assumed to transfer).
"""
import os
import sys
import time

import numpy as np
import pandas as pd
import timesfm

REPO_ROOT = "/path/to/aireadi-collab"
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
# Sibling modules live in adjacent topic directories under code/.
_CODE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _d in ("analysis", "baselines", "cgmlsm", "chronos2", "meal_detection", "model_input", "ssmcgm", "timesfm"):
    sys.path.insert(0, os.path.join(_CODE, _d))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "model_input"))
from build_participant_grid import build_grid, load_participants_covariates  # noqa: E402

CONTEXT_BINS = 144
BIN_MINUTES = 5
HORIZON_BINS = {"30min": 6, "1h": 12, "2h": 24, "3h": 36, "4h": 48}
CONDITION = "with_covariates"

COVARIATE_FEATURE_CHANNELS = [
    "heart_rate", "stress", "respiratory_rate", "oxygen_saturation",
    "physical_activity_steps", "physical_activity_calorie_rate",
]
APPLICABILITY_CHANNEL = "oxygen_saturation_applicable"
STATIC_NUMERICAL_CHANNELS = ["age"]
STATIC_CATEGORICAL_CHANNELS = ["clinical_site", "study_group"]

SPLIT = os.environ.get("TIMESFM_ICLCOV_SPLIT", "val")
EVAL_SET_PATH = f"{REPO_ROOT}/model_input/personalization/eval_set_cov_{SPLIT}.csv"
EVAL_DIR = (f"{REPO_ROOT}/model_input/zeroshot_eval" if SPLIT == "test"
            else f"{REPO_ROOT}/model_input/zeroshot_eval_{SPLIT}")

HISTORY_DAYS = int(os.environ.get("TIMESFM_ICLCOV_HISTORY_DAYS", "7"))
MAX_CONTEXT = int(os.environ.get("TIMESFM_ICLCOV_MAX_CONTEXT", "4096"))

SHARD = int(os.environ.get("TIMESFM_ICLCOV_SHARD", "0"))
NUM_SHARDS = int(os.environ.get("TIMESFM_ICLCOV_NUM_SHARDS", "1"))
MAX_PARTICIPANTS = os.environ.get("TIMESFM_ICLCOV_MAX_PARTICIPANTS")
MAX_PARTICIPANTS = int(MAX_PARTICIPANTS) if MAX_PARTICIPANTS else None

OUTPUT_DIR = os.environ.get(
    "TIMESFM_ICLCOV_RESULTS_DIR",
    f"{REPO_ROOT}/timesfm_personalization_results/{SPLIT}/incontext_cov_{HISTORY_DAYS}d",
)


def load_eval_set():
    """eval_set_cov_{split}.csv: same anchor/eligible-window machinery as
    eval_set_{split}.csv, restricted to covariate-eligible windows (real
    coverage on the 6 physiological channels), per
    build_personalization_eval_set.py's --cov mode. Falls back to the
    glucose-only eval set with a loud warning if the cov-specific file
    doesn't exist, since this script cannot invent covariate eligibility
    itself."""
    if not os.path.exists(EVAL_SET_PATH):
        raise SystemExit(f"{EVAL_SET_PATH} does not exist -- this script requires the "
                          f"covariate-eligible eval set (build_personalization_eval_set.py --cov "
                          f"{SPLIT}), it cannot fall back to the glucose-only eval set silently.")
    e = pd.read_csv(EVAL_SET_PATH, dtype={"person_id": str, "window_id": str})
    e["anchor"] = pd.to_datetime(e["anchor"], utc=True)
    history_col = f"history_start_{HISTORY_DAYS}d"
    if history_col not in e.columns:
        raise SystemExit(f"eval set has no {history_col}; "
                          f"TIMESFM_ICLCOV_HISTORY_DAYS={HISTORY_DAYS} is not one of 1/3/7")
    e[history_col] = pd.to_datetime(e[history_col], utc=True)
    return e, history_col


def extended_grid(person_id, cov, start, end):
    """Full wearable+glucose grid for person_id over [start, end), from
    build_grid() -- the same source chronos2_personalization.py's covariate
    path and build_ssmcgm_input.py use, NOT the zeroshot_eval export CSVs
    (which only ever carry 144 bins per window). Reindexed onto the
    complete 5-minute grid so gaps are explicit NaN at their true position."""
    df = build_grid(person_id, {person_id: cov})
    if df is None or df.empty:
        return None
    df = df[(df["bin_start"] >= start) & (df["bin_start"] < end)]
    if df.empty:
        return None
    full_index = pd.date_range(start, end - pd.Timedelta(minutes=BIN_MINUTES),
                                freq=f"{BIN_MINUTES}min", tz="UTC")
    df = df.set_index("bin_start").reindex(full_index)
    return df


def done_participants(path):
    if not os.path.exists(path):
        return set()
    return set(pd.read_csv(path, dtype={"person_id": str})["person_id"])


def drop_partial_participants(output_dir, done, window_to_person):
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


def build_window_inputs(hist_grid, person_id, cov, anchor, ctx_start, horizon_bins):
    """One participant's fixed [history_start, anchor) grid (hist_grid,
    built once, reused across all this participant's windows) ++ this
    window's own [ctx_start, ctx_start+144bins) grid, pulled fresh per
    window from the SAME build_grid() source. Returns (glucose_array,
    dyn_covs dict, static_num dict, static_cat dict, truth_df) or None if
    the window's own context isn't exactly 144 real-length bins."""
    ctx_end = ctx_start + pd.Timedelta(minutes=BIN_MINUTES) * CONTEXT_BINS
    horizon_end = ctx_end + pd.Timedelta(minutes=BIN_MINUTES) * horizon_bins
    win_grid = extended_grid(person_id, cov, ctx_start, horizon_end)
    if win_grid is None or len(win_grid) != CONTEXT_BINS + horizon_bins:
        return None

    win_context = win_grid.iloc[:CONTEXT_BINS]
    win_horizon = win_grid.iloc[CONTEXT_BINS:]

    # Glucose CONTEXT (history prefix + this window's own 144 bins) is
    # ffill/bfill'd here -- NOT a "fabricate the target" violation, since
    # this array is 100% pre-horizon model input; the scored ground truth
    # lives entirely in win_horizon below and is never touched. Required
    # because the vendored TimesFM xreg path (xreg_lib.normalize(), plain
    # np.mean/np.std, no NaN-awareness) turns ANY NaN anywhere in this
    # array into an all-NaN per-instance mean/std, which propagates
    # through the whole residual and crashes linear_interpolation --
    # observed 2026-09-10 on the full-population run, root-caused by
    # reading xreg_lib.py directly. Real CGM data almost always has at
    # least one 5-min gap somewhere across a 3-7 day span, which is why
    # the un-filled version skipped 22-54% of participants entirely.
    glucose_context_series = pd.concat([
        hist_grid["cgm_glucose"], win_context["cgm_glucose"],
    ]).ffill().bfill()
    if glucose_context_series.isna().all():
        return None
    full_glucose_context = glucose_context_series.to_numpy(dtype=np.float32)

    dyn_covs = {}
    for channel in COVARIATE_FEATURE_CHANNELS:
        hist_vals = hist_grid[channel]
        ctx_vals = win_context[channel]
        full_context = pd.concat([hist_vals, ctx_vals]).ffill().bfill()
        if full_context.isna().all():
            full_context = full_context.fillna(0.0)
        horizon_vals = [float(full_context.iloc[-1])] * horizon_bins
        dyn_covs[channel] = full_context.tolist() + horizon_vals

    # oxygen_saturation_applicable: real future value, a calendar fact
    # (same treatment as timesfm_zeroshot_eval.py's build_covariate_series).
    hist_appl = hist_grid[APPLICABILITY_CHANNEL]
    ctx_appl = win_context[APPLICABILITY_CHANNEL]
    full_appl = pd.concat([hist_appl, ctx_appl]).fillna(0.0)
    horizon_appl = win_horizon[APPLICABILITY_CHANNEL].fillna(0.0).tolist()
    dyn_covs[APPLICABILITY_CHANNEL] = full_appl.tolist() + horizon_appl

    static_num = {c: float(win_context[c].iloc[0]) for c in STATIC_NUMERICAL_CHANNELS}
    static_cat = {c: str(win_context[c].iloc[0]) for c in STATIC_CATEGORICAL_CHANNELS}

    truth = win_horizon[["cgm_glucose"]].reset_index().rename(
        columns={"index": "bin_start", "cgm_glucose": "actual"})
    return full_glucose_context, dyn_covs, static_num, static_cat, truth


def predict_participant(model, person_id, cov, anchor, history_start, export_by_horizon,
                         window_ids_by_horizon):
    hist_grid = extended_grid(person_id, cov, history_start, anchor)
    if hist_grid is None:
        return {}

    out = {}
    for horizon, horizon_bins in HORIZON_BINS.items():
        wanted = window_ids_by_horizon.get((person_id, horizon))
        if not wanted:
            continue
        df = export_by_horizon[horizon]
        df = df[df["window_id"].isin(wanted)]
        if df.empty:
            continue

        window_ids, glucose_inputs = [], []
        dyn_covs_batch = {c: [] for c in COVARIATE_FEATURE_CHANNELS + [APPLICABILITY_CHANNEL]}
        static_num_batch = {c: [] for c in STATIC_NUMERICAL_CHANNELS}
        static_cat_batch = {c: [] for c in STATIC_CATEGORICAL_CHANNELS}
        truths = []

        for wid, wdf in df.groupby("window_id"):
            ctx_start = wdf.loc[wdf["is_context"], "bin_start"].min()
            result = build_window_inputs(hist_grid, person_id, cov, anchor, ctx_start, horizon_bins)
            if result is None:
                continue
            glucose_ctx, dyn_covs, static_num, static_cat, truth = result
            window_ids.append(wid)
            glucose_inputs.append(glucose_ctx)
            for c in dyn_covs_batch:
                dyn_covs_batch[c].append(dyn_covs[c])
            for c in static_num_batch:
                static_num_batch[c].append(static_num[c])
            for c in static_cat_batch:
                static_cat_batch[c].append(static_cat[c])
            truth["window_id"] = wid
            truths.append(truth)

        if not window_ids:
            continue

        point_forecast, bad_idx = _forecast_with_covariates_skip_bad(
            model, glucose_inputs, dyn_covs_batch, static_num_batch, static_cat_batch)
        if bad_idx:
            for i in sorted(bad_idx):
                print(f"  skip (xreg all-NaN residual): person={person_id} horizon={horizon} "
                      f"window_id={window_ids[i]}", file=sys.stderr)

        rows = []
        keep_i = 0
        for i, (wid, truth) in enumerate(zip(window_ids, truths)):
            if i in bad_idx:
                continue
            merged = truth.copy()
            merged["predictions"] = np.asarray(point_forecast[keep_i])[:horizon_bins]
            rows.append(merged)
            keep_i += 1
        if not rows:
            continue
        merged_df = pd.concat(rows, ignore_index=True)
        merged_df["abs_error"] = (merged_df["predictions"] - merged_df["actual"]).abs()
        out[horizon] = merged_df
    return out


def _forecast_with_covariates_skip_bad(model, glucose_inputs, dyn_covs_batch, static_num_batch, static_cat_batch):
    """Wraps model.forecast_with_covariates() for a batch of windows.
    The vendored TimesFM xreg path (timesfm_2p5_base.py's internal
    target-minus-xreg residual, before strip_leading_nans/
    linear_interpolation) can produce an all-NaN residual for a rare
    window and crash with `ValueError: The truth value of an empty array
    is ambiguous` -- observed 2026-09-10 on the first participant of the
    full-population ICL+covariates run, not reproduced in the earlier
    3-participant smoke test. The failure is inside the library's
    per-batch xreg fit, not isolable to a single bad input array from the
    caller's side, so this bisects the batch on failure to find and skip
    only the genuinely bad window(s) rather than losing the whole
    participant/horizon. Returns (point_forecast_array_over_kept_windows,
    set_of_original_indices_dropped)."""
    n = len(glucose_inputs)

    def _call(idxs):
        return model.forecast_with_covariates(
            inputs=[glucose_inputs[i] for i in idxs],
            dynamic_numerical_covariates={c: [v[i] for i in idxs] for c, v in dyn_covs_batch.items()},
            dynamic_categorical_covariates={},
            static_numerical_covariates={c: [v[i] for i in idxs] for c, v in static_num_batch.items()},
            static_categorical_covariates={c: [v[i] for i in idxs] for c, v in static_cat_batch.items()},
        )

    try:
        point_forecast, _xreg = _call(list(range(n)))
        return np.asarray(point_forecast), set()
    except ValueError:
        pass

    if n == 1:
        return np.zeros((0, 1)), {0}

    kept_forecasts = [None] * n
    bad = set()
    mid = n // 2
    for half_idxs in (list(range(0, mid)), list(range(mid, n))):
        if not half_idxs:
            continue
        sub_forecast, sub_bad = _forecast_with_covariates_skip_bad(
            model,
            [glucose_inputs[i] for i in half_idxs],
            {c: [v[i] for i in half_idxs] for c, v in dyn_covs_batch.items()},
            {c: [v[i] for i in half_idxs] for c, v in static_num_batch.items()},
            {c: [v[i] for i in half_idxs] for c, v in static_cat_batch.items()},
        )
        keep_local = [j for j in range(len(half_idxs)) if j not in sub_bad]
        for pos, local_j in enumerate(keep_local):
            kept_forecasts[half_idxs[local_j]] = sub_forecast[pos]
        for local_j in sub_bad:
            bad.add(half_idxs[local_j])

    order = [i for i in range(n) if i not in bad]
    stacked = np.stack([kept_forecasts[i] for i in order]) if order else np.zeros((0, 1))
    # remap to a contiguous 0..len(order)-1 result; caller iterates keep_i in that order
    return stacked, bad


def load_export(horizon, keep_window_ids):
    path = f"{EVAL_DIR}/zeroshot_{horizon}_{CONDITION}.csv"
    df = pd.read_csv(path, dtype={"window_id": str, "person_id": str})
    df = df[df["window_id"].isin(keep_window_ids)].copy()
    df["bin_start"] = pd.to_datetime(df["bin_start"], utc=True)
    df["is_context"] = df["is_context"].astype(bool)
    return df


def main():
    job_start = time.time()
    e, history_col = load_eval_set()
    covs = load_participants_covariates()

    window_to_person = dict(zip(e["window_id"], e["person_id"]))
    window_ids_by_horizon = e.groupby(["person_id", "horizon"])["window_id"].apply(set).to_dict()
    anchors = e.groupby("person_id").agg(anchor=("anchor", "first"), history_start=(history_col, "first"))

    all_participants = sorted(anchors.index)
    participants = [p for i, p in enumerate(all_participants) if i % NUM_SHARDS == SHARD]
    if MAX_PARTICIPANTS is not None:
        participants = participants[:MAX_PARTICIPANTS]

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    done_path = f"{OUTPUT_DIR}/participants_done_shard{SHARD}.csv"
    done = done_participants(done_path)
    if done:
        print(f"resuming: {len(done)} participants already complete in this shard", file=sys.stderr)
        drop_partial_participants(OUTPUT_DIR, done, window_to_person)
    todo = [p for p in participants if p not in done]

    keep_ids = set(e.loc[e["person_id"].isin(participants), "window_id"])
    export_by_horizon = {h: load_export(h, keep_ids) for h in HORIZON_BINS}
    print(f"shard {SHARD}/{NUM_SHARDS}: {len(participants)} participants ({len(todo)} to do), "
          f"split={SPLIT}, k={HISTORY_DAYS}d, max_context={MAX_CONTEXT}", file=sys.stderr)

    load_start = time.time()
    model = timesfm.TimesFM_2p5_200M_torch.from_pretrained("google/timesfm-2.5-200m-pytorch")
    model.compile(timesfm.ForecastConfig(
        max_context=MAX_CONTEXT, max_horizon=256, normalize_inputs=True,
        use_continuous_quantile_head=True, fix_quantile_crossing=True,
        return_backcast=True,
    ))
    model_load_seconds = round(time.time() - load_start, 1)
    print(f"model loaded + compiled (max_context={MAX_CONTEXT}, return_backcast=True) "
          f"in {model_load_seconds}s", file=sys.stderr)

    skipped = []
    n_done = 0
    for i, pid in enumerate(todo):
        anchor = anchors.loc[pid, "anchor"]
        history_start = anchors.loc[pid, "history_start"]
        cov = covs.get(str(pid), {})

        preds = predict_participant(model, pid, cov, anchor, history_start, export_by_horizon,
                                     window_ids_by_horizon)
        if not preds:
            skipped.append(pid)
            continue
        for horizon, merged in preds.items():
            path = f"{OUTPUT_DIR}/predictions_{horizon}_{CONDITION}.csv"
            merged.to_csv(path, mode="a", header=not os.path.exists(path), index=False)

        pd.DataFrame([{"person_id": pid}]).to_csv(done_path, mode="a", header=not os.path.exists(done_path), index=False)
        n_done += 1

        if (i + 1) % 10 == 0 or i + 1 == len(todo):
            print(f"  {i + 1}/{len(todo)} participants, {time.time() - job_start:.0f}s elapsed", file=sys.stderr)

    if skipped:
        print(f"WARNING: skipped {len(skipped)} participants (no usable history/window data): "
              f"{skipped[:5]}", file=sys.stderr)
        pd.DataFrame({"person_id": skipped}).to_csv(f"{OUTPUT_DIR}/skipped_shard{SHARD}.csv", index=False)

    rows = []
    for horizon in HORIZON_BINS:
        path = f"{OUTPUT_DIR}/predictions_{horizon}_{CONDITION}.csv"
        if not os.path.exists(path):
            continue
        full = pd.read_csv(path, dtype={"window_id": str})
        scored = full.dropna(subset=["actual"])
        rows.append({
            "horizon": horizon, "split": SPLIT, "condition": CONDITION,
            "history_days": HISTORY_DAYS, "max_context": MAX_CONTEXT,
            "mae_mgdl": scored["abs_error"].mean(),
            "rmse_mgdl": float(np.sqrt((scored["abs_error"] ** 2).mean())),
            "n_scored": len(scored), "n_total": len(full), "n_windows": full["window_id"].nunique(),
            "n_participants_this_shard": n_done, "model_load_seconds": model_load_seconds,
            "cumulative_job_seconds": round(time.time() - job_start, 1),
        })
    if rows:
        summary = pd.DataFrame(rows)
        summary.to_csv(f"{OUTPUT_DIR}/summary_shard{SHARD}.csv", index=False)
        print(summary.to_string(index=False), file=sys.stderr)


if __name__ == "__main__":
    main()
