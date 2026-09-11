"""Chronos-2 in-context personalization WITH covariates: the combination of
code/chronos2/chronos2_incontext_personalization.py's history-extension mechanism
(participant's own 1/3/7-day history prepended to the window's own 144-bin
context, zero weight updates) and code/chronos2/chronos2_zeroshot_eval.py's
covariate mechanism (predict_df's df/future_df, standard 6-channel
physiological bundle + oxygen_saturation_applicable) -- never combined
before this script, mirroring the equivalent, already-built-and-debugged
code/timesfm/timesfm_incontext_covariates_personalization.py exactly where the two
models' APIs allow.

WHY THIS IS A REAL GAP, NOT AN OVERSIGHT LEFT AS-IS
======================================================
chronos2_incontext_personalization.py's own docstring says: "no covariates
in this arm; combining in-context history injection with covariates is a
real follow-up experiment, not this one." Chronos-2 got a LoRA+covariates
sweep (cov_arm1-4) but never this ICL+covariates equivalent -- this closes
that specific hole, giving Chronos-2 the same three-mechanism x covariates
matrix TimesFM now has (zero-shot / ICL / LoRA, each with and without
covariates, LoRA's covariate arm aside since TimesFM's LoRA architecturally
cannot take covariates at all -- Chronos-2 has no such block for either
mechanism).

COVARIATE HISTORY SOURCE: build_grid() (build_participant_grid.py), the
SAME function chronos2_personalization.py's covariate path, build_ssmcgm_
input.py, and timesfm_incontext_covariates_personalization.py all use --
NOT chronos2_incontext_personalization.py's own history_array() (glucose-
only, sourced from load_cleaned_glucose_binned, no covariate columns) and
NOT the zeroshot_eval export CSVs (only 144 bins per window). Both the
extended history prefix AND the window's own 144-bin context are read from
this single source, exactly like the TimesFM version, for the same reason:
one consistent provenance avoids a seam between two independently-built
data sources for the same channels.

REAL BUG ALREADY FOUND AND FIXED ONCE IN THE TIMESFM VERSION, APPLIED HERE
PREEMPTIVELY: TimesFM's vendored xreg_lib.normalize() computes plain
np.mean/np.std on the raw glucose context with no NaN-awareness, so any
real CGM gap anywhere in a multi-day span turned the whole per-window
normalization into NaN and crashed the model (fixed by ffill/bfill on the
glucose CONTEXT array, never the scored truth). Chronos-2's own model has
built-in context-NaN masking for the glucose target (context_mask =
~isnan(context), confirmed in this project's own chronos2_incontext_
personalization.py docstring), so that specific crash mode is not expected
to reproduce here -- but predict_df's covariate columns are NOT confirmed
NaN-safe over a multi-day span (chronos2_zeroshot_eval.py's own covariate
construction only explicitly ffill/bfill's predmeal_prob and sleep_
regularity_index within a 144-bin window; the 6 physiological channels are
passed to predict_df with whatever raw NaN they have). Applying the same
ffill/bfill-then-zero-fallback treatment to ALL active covariate channels
here, over the full extended span, defensively -- cheap insurance against
an analogous failure mode, not a confirmed bug in Chronos-2's own code.

GLUCOSE TARGET NaN: left exactly as chronos2_incontext_personalization.py
already does (NOT filled) -- Chronos-2's own context_mask handles it, and
filling would violate this project's "cgm_glucose is never fabricated"
rule for no benefit here.

SCOPE: standard 6-channel physiological bundle + oxygen_saturation_
applicable only (CHRONOS2_INCLUDE_* env vars all default off, matching
chronos2_zeroshot_eval.py's own defaults) -- the same scope TimesFM's
ICL+covariates arm was built and run at. Addon channels (sleep_stage,
predmeal_prob, tod, sleep_regularity, static) are out of scope for this
script; it reuses chronos2_zeroshot_eval's ACTIVE_* lists so those flags
WOULD work if set, but this is not smoke-tested for that configuration.
"""
import os
import sys
import time

import numpy as np
import pandas as pd
import torch

REPO_ROOT = "/path/to/aireadi-collab"
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
# Sibling modules live in adjacent topic directories under code/.
_CODE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _d in ("analysis", "baselines", "cgmlsm", "chronos2", "meal_detection", "model_input", "ssmcgm", "timesfm"):
    sys.path.insert(0, os.path.join(_CODE, _d))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "model_input"))
sys.path.insert(0, f"{REPO_ROOT}/code/external/chronos-forecasting/src")
from chronos import BaseChronosPipeline  # noqa: E402
from build_participant_grid import build_grid, load_participants_covariates  # noqa: E402
import chronos2_zeroshot_eval as zs  # noqa: E402

CONTEXT_BINS = 144
BIN_MINUTES = 5
MODEL_CONTEXT_LENGTH = 2048  # Chronos-2's real, verified hard cap
HORIZON_BINS = {"30min": 6, "1h": 12, "2h": 24, "3h": 36, "4h": 48}
CONDITION = "with_covariates"

COVARIATE_FEATURE_CHANNELS = zs.ACTIVE_COVARIATE_FEATURE_CHANNELS
APPLICABILITY_CHANNELS = zs.ACTIVE_COVARIATE_APPLICABILITY_CHANNELS

SPLIT = os.environ.get("CHRONOS2_ICLCOV_SPLIT", "val")
EVAL_SET_PATH = f"{REPO_ROOT}/model_input/personalization/eval_set_cov_{SPLIT}.csv"

HISTORY_DAYS = int(os.environ.get("CHRONOS2_ICLCOV_HISTORY_DAYS", "7"))

SHARD = int(os.environ.get("CHRONOS2_ICLCOV_SHARD", "0"))
NUM_SHARDS = int(os.environ.get("CHRONOS2_ICLCOV_NUM_SHARDS", "1"))
MAX_PARTICIPANTS = os.environ.get("CHRONOS2_ICLCOV_MAX_PARTICIPANTS")
MAX_PARTICIPANTS = int(MAX_PARTICIPANTS) if MAX_PARTICIPANTS else None

OUTPUT_DIR = os.environ.get(
    "CHRONOS2_ICLCOV_RESULTS_DIR",
    f"{REPO_ROOT}/chronos2_personalization_results/{SPLIT}/incontext_cov_{HISTORY_DAYS}d",
)


def load_eval_set():
    if not os.path.exists(EVAL_SET_PATH):
        raise SystemExit(f"{EVAL_SET_PATH} does not exist -- this script requires the "
                          f"covariate-eligible eval set (build_personalization_eval_set.py --cov "
                          f"{SPLIT}), it cannot fall back to the glucose-only eval set silently.")
    e = pd.read_csv(EVAL_SET_PATH, dtype={"person_id": str, "window_id": str})
    e["anchor"] = pd.to_datetime(e["anchor"], utc=True)
    history_col = f"history_start_{HISTORY_DAYS}d"
    if history_col not in e.columns:
        raise SystemExit(f"eval set has no {history_col}; "
                          f"CHRONOS2_ICLCOV_HISTORY_DAYS={HISTORY_DAYS} is not one of 1/3/7")
    e[history_col] = pd.to_datetime(e[history_col], utc=True)
    return e, history_col


def extended_grid(person_id, cov, start, end):
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


def build_window_frames(hist_grid, person_id, cov, ctx_start, horizon_bins, window_id):
    """One window: extended history ++ this window's own 144-bin context,
    same source (build_grid), for context_df; this window's own horizon
    rows for future_df (applicability channel only, real future value, a
    calendar fact, matching chronos2_zeroshot_eval.py's treatment).
    Returns (context_df, future_df_row, truth) or None."""
    ctx_end = ctx_start + pd.Timedelta(minutes=BIN_MINUTES) * CONTEXT_BINS
    horizon_end = ctx_end + pd.Timedelta(minutes=BIN_MINUTES) * horizon_bins
    win_grid = extended_grid(person_id, cov, ctx_start, horizon_end)
    if win_grid is None or len(win_grid) != CONTEXT_BINS + horizon_bins:
        return None

    win_context = win_grid.iloc[:CONTEXT_BINS]
    win_horizon = win_grid.iloc[CONTEXT_BINS:]

    full_index = pd.concat([hist_grid, win_context]).index
    n = len(full_index)
    last_real_ts = win_context.index[-1]
    # tz_localize(None): predict_df's normalize_df calls .to_numpy().view("int64")
    # on the timestamp column, which requires a plain datetime64[ns] array --
    # a tz-aware column comes back as an object array of Timestamp references
    # and crashes that cast. chronos2_incontext_personalization.py already
    # does this same strip; missed it here on first write.
    synthetic_index = pd.date_range(end=last_real_ts, periods=n, freq=f"{BIN_MINUTES}min",
                                     tz="UTC").tz_localize(None)

    frame = pd.DataFrame({"item_id": str(window_id), "timestamp": synthetic_index})
    frame["target"] = pd.concat([hist_grid["cgm_glucose"], win_context["cgm_glucose"]]).to_numpy(dtype=np.float32)

    for channel in COVARIATE_FEATURE_CHANNELS:
        combined = pd.concat([hist_grid[channel], win_context[channel]]).ffill().bfill()
        if combined.isna().all():
            combined = combined.fillna(0.0)
        frame[channel] = combined.to_numpy(dtype=np.float32)

    for channel in APPLICABILITY_CHANNELS:
        combined = pd.concat([hist_grid[channel], win_context[channel]]).fillna(0.0)
        frame[channel] = combined.to_numpy(dtype=np.float32)

    future_row = None
    if APPLICABILITY_CHANNELS:
        future_ts = pd.date_range(last_real_ts + pd.Timedelta(minutes=BIN_MINUTES),
                                   periods=horizon_bins, freq=f"{BIN_MINUTES}min",
                                   tz="UTC").tz_localize(None)
        future_row = pd.DataFrame({"item_id": str(window_id), "timestamp": future_ts})
        for channel in APPLICABILITY_CHANNELS:
            future_row[channel] = win_horizon[channel].fillna(0.0).to_numpy(dtype=np.float32)

    truth = win_horizon[["cgm_glucose"]].reset_index().rename(
        columns={"index": "bin_start", "cgm_glucose": "actual"})
    return frame, future_row, truth


def predict_participant(pipeline, person_id, cov, anchor, history_start, export_by_horizon,
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

        context_frames, future_frames, truths = [], [], []
        for wid, wdf in df.groupby("window_id"):
            ctx_start = wdf.loc[wdf["is_context"], "bin_start"].min()
            result = build_window_frames(hist_grid, person_id, cov, ctx_start, horizon_bins, wid)
            if result is None:
                continue
            frame, future_row, truth = result
            context_frames.append(frame)
            if future_row is not None:
                future_frames.append(future_row)
            truth["window_id"] = wid
            truths.append(truth)

        if not context_frames:
            continue

        context_df = pd.concat(context_frames, ignore_index=True)
        future_df = pd.concat(future_frames, ignore_index=True) if future_frames else None
        forecast = pipeline.predict_df(
            df=context_df,
            future_df=future_df,
            id_column="item_id",
            timestamp_column="timestamp",
            target="target",
            prediction_length=horizon_bins,
            context_length=MODEL_CONTEXT_LENGTH,
            freq=f"{BIN_MINUTES}min",
        )
        forecast = forecast.rename(columns={"item_id": "window_id"})

        rows = []
        for truth in truths:
            wid = truth["window_id"].iloc[0]
            fdf = forecast[forecast["window_id"] == str(wid)].reset_index(drop=True)
            if len(fdf) != len(truth):
                continue
            merged = truth.copy()
            merged["predictions"] = fdf["predictions"].to_numpy()
            rows.append(merged)
        if rows:
            merged_df = pd.concat(rows, ignore_index=True)
            merged_df["abs_error"] = (merged_df["predictions"] - merged_df["actual"]).abs()
            out[horizon] = merged_df
    return out


def load_export(horizon, keep_window_ids):
    eval_dir = (f"{REPO_ROOT}/model_input/zeroshot_eval" if SPLIT == "test"
                else f"{REPO_ROOT}/model_input/zeroshot_eval_{SPLIT}")
    path = f"{eval_dir}/zeroshot_{horizon}_{CONDITION}.csv"
    df = pd.read_csv(path, dtype={"window_id": str, "person_id": str})
    df = df[df["window_id"].isin(keep_window_ids)].copy()
    df["bin_start"] = pd.to_datetime(df["bin_start"], utc=True)
    df["is_context"] = df["is_context"].astype(bool)
    return df


def main():
    job_start = time.time()
    e, history_col = load_eval_set()
    all_covs = load_participants_covariates()

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
          f"split={SPLIT}, k={HISTORY_DAYS}d, model_context_length={MODEL_CONTEXT_LENGTH}",
          file=sys.stderr)

    load_start = time.time()
    pipeline = BaseChronosPipeline.from_pretrained(
        "amazon/chronos-2", device_map="cuda" if torch.cuda.is_available() else "cpu")
    model_load_seconds = round(time.time() - load_start, 1)
    print(f"model loaded in {model_load_seconds}s", file=sys.stderr)

    skipped = []
    n_done = 0
    for i, pid in enumerate(todo):
        anchor = anchors.loc[pid, "anchor"]
        history_start = anchors.loc[pid, "history_start"]
        cov = all_covs.get(str(pid), {})

        preds = predict_participant(pipeline, pid, cov, anchor, history_start, export_by_horizon,
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

    rows = []
    for horizon in HORIZON_BINS:
        path = f"{OUTPUT_DIR}/predictions_{horizon}_{CONDITION}.csv"
        if not os.path.exists(path):
            continue
        full = pd.read_csv(path, dtype={"window_id": str})
        scored = full.dropna(subset=["actual"])
        rows.append({
            "horizon": horizon, "split": SPLIT, "condition": CONDITION,
            "history_days": HISTORY_DAYS, "model_context_length": MODEL_CONTEXT_LENGTH,
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
