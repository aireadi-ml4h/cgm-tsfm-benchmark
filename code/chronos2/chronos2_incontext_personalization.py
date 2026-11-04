"""Chronos-2 in-context personalization: no fine-tuning, no LoRA. The
participant's own 1/3/7 days of prior glucose history is prepended directly
to the model's normal 144-bin window context and handed to predict_df in ONE
call -- purely a longer context, zero weight updates. Mirrors
code/timesfm/timesfm_incontext_personalization.py's design exactly (same eval set,
same anchors, same k=1/3/7-day history, same glucose-only scope), so
TimesFM's and Chronos-2's in-context arms are directly comparable, and this
new arm is directly comparable to chronos2_personalization.py's own LoRA
arm (ft_p{k}d) on the identical eval set E.

WHY THIS EXISTS ALONGSIDE chronos2_personalization.py, NOT INSTEAD OF IT
=========================================================================
chronos2_personalization.py answers "does adapting Chronos-2's WEIGHTS to
one person's history help." This answers a different, cheaper question:
"does just SHOWING the model more of this person's own data help, with no
training at all." Until now Chronos-2 had zero-shot and LoRA arms but no
in-context arm -- TimesFM had all three. This closes that gap so all
personalization-mechanism comparisons in this project can be made for both
foundation models, not just one.

REUSES chronos2_personalization.py's history_array() (COPIED, per that
module's own precedent for cross-file duplication of this exact function --
see timesfm_incontext_personalization.py's docstring for the general
reasoning: avoiding a cross-venv or unnecessary import-time dependency.
Here there is no cross-venv issue, both scripts share venv-chronos, but the
function is trivial (15 lines) and copying keeps this script fully
self-contained and independently readable).

BUG FOUND AND FIXED 2026-09-07 (caught by a collaborator's review, not by
this project's own testing): the original version of this script called
`history_array(person_id, history_start, ctx_start)` per window instead of
per participant -- since every window but a participant's FIRST has
`ctx_start > anchor` (build_personalization_eval_set.py's own documented,
deliberate conservative gap), this silently re-fetched the ENTIRE k-day
history PLUS an uncontrolled, window-dependent gap (mean 25.7h, max 137.5h,
measured directly against the real eval sets) every time, instead of just
the small anchor-to-ctx_start gap the code's own naming implied. Net effect:
k was not actually controlled -- k=1 fed a mean of ~2.1 real days, k=3 fed
~4.1, and EVERY k=7 window (not just a small fraction, as this docstring
previously claimed) hit the 2048-bin cap. Two literal dead lines (an
earlier, correct `history_values`+`ctx` concatenation, immediately
overwritten) were sitting right above the bug, unused.

FIXED to match timesfm_incontext_personalization.py's approach exactly:
the fed context is the participant's real k-day history CONCATENATED
DIRECTLY to this window's own 144-bin context, with NOTHING filling the gap
between anchor and ctx_start -- a real, deliberate time discontinuity, not
a bug. This is safe for Chronos-2, verified directly against the vendored
source before trusting it (not assumed): `predict_df`'s docstring claims
"time series ... must have regular timestamps with the same frequency (no
gaps)" (pipeline.py:854), but `validate_df` (chronos/df_utils.py:139-195)
performs NO check of this at all -- only column presence, dtype, and
future_df shape are validated. More fundamentally, `from_data_frame`
(chronos2/preprocess.py:115-210) builds the model's actual target array via
`df[target_columns].to_numpy(...).T` -- purely POSITIONAL, in
already-sorted row order; the timestamp column is never consulted again
after sorting except by `make_future_df` (chronos/df_utils.py:62-103),
which only reads each series' LAST timestamp + a `freq` offset to generate
forecast-horizon timestamps -- it does not inspect or require regularity of
the timestamps in between. So a fed series with a real calendar gap is
consumed identically to one without: the model sees N context positions in
row order, full stop. Synthetic, contiguous timestamps are assigned to the
concatenated context purely to satisfy predict_df's schema requirement (a
timestamp column must exist), not because the model uses them for anything
beyond sorting and forecasting the next `prediction_length` steps after the
last one.

This also fixes a real performance issue a collaborator's review flagged: the buggy
per-window `history_array()` call re-read each participant's full raw
glucose CSV from disk once per window; the fix calls it once per
participant (already how `main()` passed `history_values` in -- the
now-removed per-window call was ignoring that argument entirely).

REAL, DISCLOSED CONTEXT-LENGTH LIMIT, corrected math (verified directly
against the vendored source, not assumed): Chronos-2's predict_df accepts
an arbitrary-length long-format dataframe and a `context_length` parameter
controlling how much of the MOST RECENT part of it is actually used at
inference -- confirmed by reading the actual truncation line,
`pipeline.py:436: context = torch.cat([...], dim=-1)[...,
-self.model_context_length:]`, which keeps the TAIL (most recent), drops
the head (oldest) -- not merely inferred from the docstring's prose.
Chronos-2's own default/hard context_length is 2048 (pipeline.py:44,
`default_context_length: int = 2048`) -- confirmed directly: passing a
LARGER context_length than this triggers a warning and resets to 2048
(pipeline.py:612-617), it is NOT extended by request the way TimesFM's
max_context can be raised up to 16384. Now that the fetch bug above is
fixed, the fed length is EXACTLY k*288 + 144 bins, no gap inflation: k=1 is
432 bins, k=3 is 1008 bins (both comfortably under 2048, unaffected), k=7
is 2016+144=2160 bins -- exactly 112 bins (9.3 hours) OVER the cap, so
Chronos-2 will use only the most recent 2048 of those 2160 bins, dropping
the OLDEST 112 bins (9.3 hours) of the requested 7-day history. This
matches what this docstring claimed BEFORE the bug was found -- the
original math was right, only the actual fed length (corrupted by the
per-window refetch bug above) was wrong. context_length is passed
explicitly as 2048 (the model's real, verified cap) rather than a larger
number that would just be silently reset anyway, so the actual behavior is
not left implicit.

NaN IS LEFT RAW, NOT FILLED -- matches this project's own "cgm_glucose is
never imputed" rule (report.md Sec 13, Sec 15.7) and chronos2_personalization.py's
own history_array()/model.py's confirmed context-NaN masking (context_mask =
~isnan(context); a patch is attended only if at least one bin is observed).

GLUCOSE-ONLY, MATCHING chronos2_personalization.py's PHASE 1 AND
timesfm_incontext_personalization.py -- no covariates in this arm; combining
in-context history injection with covariates is a real follow-up experiment,
not this one.
"""
import os
import sys
import time

import numpy as np
import pandas as pd
import torch

REPO_ROOT = os.environ["AIREADI_ROOT"]
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
# Sibling modules live in adjacent topic directories under code/.
_CODE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _d in ("analysis", "baselines", "cgmlsm", "chronos2", "meal_detection", "model_input", "ssmcgm", "timesfm"):
    sys.path.insert(0, os.path.join(_CODE, _d))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "model_input"))
sys.path.insert(0, f"{REPO_ROOT}/code/external/chronos-forecasting/src")
from chronos import BaseChronosPipeline  # noqa: E402
from build_participant_grid import load_cleaned_glucose_binned  # noqa: E402
import chronos2_zeroshot_eval as zs  # noqa: E402

CONTEXT_BINS = 144  # the 144-bin window context every zeroshot_eval CSV window has
BIN_MINUTES = 5
MODEL_CONTEXT_LENGTH = 2048  # Chronos-2's real, verified hard cap -- see module docstring
HORIZON_BINS = {"30min": 6, "1h": 12, "2h": 24, "3h": 36, "4h": 48}
CONDITION = "glucose_only"

SPLIT = os.environ.get("CHRONOS2_ICL_SPLIT", "val")
EVAL_SET_PATH = f"{REPO_ROOT}/model_input/personalization/eval_set_{SPLIT}.csv"
EVAL_DIR = (f"{REPO_ROOT}/model_input/zeroshot_eval" if SPLIT == "test"
            else f"{REPO_ROOT}/model_input/zeroshot_eval_{SPLIT}")

HISTORY_DAYS = int(os.environ.get("CHRONOS2_ICL_HISTORY_DAYS", "7"))

SHARD = int(os.environ.get("CHRONOS2_ICL_SHARD", "0"))
NUM_SHARDS = int(os.environ.get("CHRONOS2_ICL_NUM_SHARDS", "1"))
MAX_PARTICIPANTS = os.environ.get("CHRONOS2_ICL_MAX_PARTICIPANTS")
MAX_PARTICIPANTS = int(MAX_PARTICIPANTS) if MAX_PARTICIPANTS else None

OUTPUT_DIR = os.environ.get(
    "CHRONOS2_ICL_RESULTS_DIR",
    f"{REPO_ROOT}/chronos2_personalization_results/{SPLIT}/icl_p{HISTORY_DAYS}d",
)


def history_array(person_id, history_start, anchor):
    """Copy of chronos2_personalization.py's function of the same name --
    see module docstring for why it's copied here rather than imported."""
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


def load_export(horizon, keep_window_ids):
    path = f"{EVAL_DIR}/zeroshot_{horizon}_{CONDITION}.csv"
    df = pd.read_csv(path, dtype={"window_id": str, "person_id": str})
    df = df[df["window_id"].isin(keep_window_ids)].copy()
    df["bin_start"] = pd.to_datetime(df["bin_start"], utc=True).dt.tz_localize(None)
    df["is_context"] = df["is_context"].astype(bool)
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


def predict_participant(pipeline, history_values, export_by_horizon,
                         window_ids_by_horizon, person_id):
    """For every eval window this participant has, at every horizon: build
    ONE long-format dataframe = [participant's real k-day history] ++
    [this window's own 144-bin context], glued DIRECTLY together (no gap
    fill -- see module docstring's "BUG FOUND AND FIXED" section for why
    this is correct and safe, matching timesfm_incontext_personalization.py's
    identical design), call predict_df once per window (predict_df's own
    item_id grouping means windows cannot share one call unless their
    concatenated series were identical length AND aligned, which they are
    not here since each window's own context_start differs) --
    context_length=MODEL_CONTEXT_LENGTH passed explicitly so the real
    2048-bin cap is used deliberately, not left to the vendor's own silent
    reset warning. `history_values` is computed ONCE per participant in
    main() and passed in unchanged -- no per-window refetch."""
    out = {}
    for horizon, horizon_bins in HORIZON_BINS.items():
        wanted = window_ids_by_horizon.get((person_id, horizon))
        if not wanted:
            continue
        df = export_by_horizon[horizon]
        df = df[df["window_id"].isin(wanted)]
        if df.empty:
            continue

        rows = []
        for wid, wdf in df.groupby("window_id"):
            wdf = wdf.sort_values("bin_start")
            ctx = wdf.loc[wdf["is_context"], "cgm_glucose"].to_numpy(dtype=np.float32)
            tgt = wdf.loc[~wdf["is_context"], ["bin_start", "cgm_glucose"]].reset_index(drop=True)
            if len(ctx) != CONTEXT_BINS:
                continue
            ctx_start = wdf.loc[wdf["is_context"], "bin_start"].min()
            # Direct concatenation, no gap fill -- exactly k*288 + 144 bins
            # every time, matching TimesFM's in-context design. Verified
            # safe: predict_df consumes the target column purely
            # positionally (see module docstring), so the timestamp column
            # only needs to exist and be sorted/regular for schema purposes
            # -- it is not used to detect or react to the real calendar gap
            # between the history's end (anchor) and this window's own
            # context_start. Synthetic timestamps anchored so the LAST bin
            # equals this window's true final context timestamp (keeps any
            # debug/print output interpretable); everything earlier is a
            # contiguous synthetic 5-minute grid, not real calendar time.
            full_values = np.concatenate([history_values, ctx])
            n = len(full_values)
            last_real_ts = pd.Timestamp(ctx_start, tz="UTC") + pd.Timedelta(minutes=BIN_MINUTES) * (CONTEXT_BINS - 1)
            full_index = pd.date_range(end=last_real_ts, periods=n,
                                        freq=f"{BIN_MINUTES}min", tz="UTC").tz_localize(None)
            context_df = pd.DataFrame({"item_id": str(wid), "timestamp": full_index, "target": full_values})
            rows.append((wid, context_df, tgt))

        if not rows:
            continue

        context_frames = pd.concat([r[1] for r in rows], ignore_index=True)
        forecast = pipeline.predict_df(
            df=context_frames,
            id_column="item_id",
            timestamp_column="timestamp",
            target="target",
            prediction_length=horizon_bins,
            context_length=MODEL_CONTEXT_LENGTH,
            freq=f"{BIN_MINUTES}min",
        )
        forecast = forecast.rename(columns={"item_id": "window_id"})

        merged_rows = []
        for wid, _, tgt in rows:
            fdf = forecast[forecast["window_id"] == str(wid)].reset_index(drop=True)
            if len(fdf) != len(tgt):
                continue
            merged = tgt.rename(columns={"cgm_glucose": "actual"}).copy()
            merged["window_id"] = wid
            merged["predictions"] = fdf["predictions"].to_numpy()
            merged_rows.append(merged)
        if merged_rows:
            merged_df = pd.concat(merged_rows, ignore_index=True)
            merged_df["abs_error"] = (merged_df["predictions"] - merged_df["actual"]).abs()
            out[horizon] = merged_df
    return out


def main():
    job_start = time.time()
    e = pd.read_csv(EVAL_SET_PATH, dtype={"person_id": str, "window_id": str})
    e["anchor"] = pd.to_datetime(e["anchor"], utc=True)
    history_col = f"history_start_{HISTORY_DAYS}d"
    if history_col not in e.columns:
        raise SystemExit(f"eval set has no {history_col}; "
                          f"CHRONOS2_ICL_HISTORY_DAYS={HISTORY_DAYS} is not one of 1/3/7")
    e[history_col] = pd.to_datetime(e[history_col], utc=True)

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

    expected_bins = HISTORY_DAYS * 24 * 60 // BIN_MINUTES
    skipped = []
    n_done = 0
    for i, pid in enumerate(todo):
        anchor = anchors.loc[pid, "anchor"]
        history_start = anchors.loc[pid, "history_start"]
        values = history_array(pid, history_start, anchor)
        if values is None or len(values) != expected_bins:
            skipped.append((pid, 0 if values is None else len(values)))
            continue

        preds = predict_participant(pipeline, values, export_by_horizon,
                                     window_ids_by_horizon, pid)
        for horizon, merged in preds.items():
            path = f"{OUTPUT_DIR}/predictions_{horizon}_{CONDITION}.csv"
            merged.to_csv(path, mode="a", header=not os.path.exists(path), index=False)

        pd.DataFrame([{"person_id": pid, "history_bins": len(values), "expected_bins": expected_bins,
                        "real_fraction": round(float(np.isfinite(values).mean()), 4)}]).to_csv(
            done_path, mode="a", header=not os.path.exists(done_path), index=False)
        n_done += 1

        if (i + 1) % 10 == 0 or i + 1 == len(todo):
            print(f"  {i + 1}/{len(todo)} participants, {time.time() - job_start:.0f}s elapsed",
                  file=sys.stderr)

    if skipped:
        print(f"WARNING: skipped {len(skipped)} participants with wrong-length history "
              f"(expected {expected_bins} bins): {skipped[:5]}", file=sys.stderr)
        pd.DataFrame(skipped, columns=["person_id", "history_bins"]).to_csv(
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
