"""TimesFM 2.5 in-context personalization: no fine-tuning, no LoRA. The
participant's own 1/3/7 days of prior glucose history is prepended directly
to the model's normal 144-bin context and handed to the SAME zero-shot
`model.forecast()` call every other TimesFM result in this project uses --
purely a longer context, zero weight updates.

WHY THIS EXISTS ALONGSIDE code/chronos2/chronos2_personalization.py, NOT INSTEAD OF IT
===============================================================================
chronos2_personalization.py (Stephanie's, active as of 2026-09-02) answers
"does adapting Chronos-2's WEIGHTS to one person's history help." This
answers a different, cheaper question for a different model: "does just
SHOWING the model more of this person's own data help, with no training at
all." TimesFM has no LoRA-compatible trainable path with covariate support
(see model_input/HANDOFF.md's 2026-09-02 section), and no personalization
work of any kind existed for it before this file -- this is not a duplicate
of anything, LoRA or otherwise, for either model.

REUSES, DOES NOT DUPLICATE, Stephanie's eval-set infrastructure
=================================================================
model_input/personalization/eval_set_{val,test}.csv (build_personalization_
eval_set.py) is model-agnostic by its own docstring ("nothing here touches a
model") -- the causal anchor T(p), the eligible window set E(p), and the
history_start_{1,3,7}d columns are pure function of window_index.csv and each
participant's own cleaned glucose, independent of which model consumes them.
This script reads that file directly, unmodified.

`history_array()` below is a direct copy of chronos2_personalization.py's
function of the same name (same logic, same docstring), NOT an import of it:
importing that module pulls in `from chronos import BaseChronosPipeline` at
module scope, which requires `einops` -- installed in venv-chronos, absent
from venv-timesfm (confirmed directly: smoke job 41803706 failed on exactly
this ModuleNotFoundError). The function itself only ever calls
build_participant_grid.load_cleaned_glucose_binned (plain pandas/numpy, no
Chronos dependency), so duplicating these ~15 lines avoids a cross-venv
dependency coupling with zero loss of behavior -- it is not a fork of any
Chronos-2-specific logic, since there isn't any here to begin with.

THE CONCATENATED CONTEXT, AND WHY THE ANCHOR STAYS FIXED PER PARTICIPANT
===========================================================================
For a window whose own 144-bin context starts at `context_start`, the full
input handed to `model.forecast()` is:
    [glucose over [history_start_kd, anchor)]  ++  [glucose over [context_start-144*5min, context_start)]
i.e. the participant's fixed k-day history, followed by that window's own
real context. Because E(p)'s anchor is a single fixed point per participant
(shared across every one of their eval windows, exactly as
chronos2_personalization.py uses it), windows later in a participant's
record have a temporal gap between the end of the injected history (at the
anchor) and the start of their own context (at context_start >= anchor).
This is the identical, already-disclosed conservative choice
build_personalization_eval_set.py documents for the LoRA arm ("for p's LATER
windows we withhold recent history a real deployment would have had ...
deliberate and conservative: it can only understate personalization, never
inflate it") -- kept identical here on purpose, not re-derived, so the two
personalization arms (fit-based, in-context) are answering their respective
questions on the literal same eval set and the same history definition, and
only the ADAPTATION MECHANISM differs between them.

NaN IS LEFT RAW, NOT FILLED -- matches timesfm_zeroshot_eval.py's glucose-
only path exactly (with_covariates fills because forecast_with_covariates's
xreg fit cannot tolerate NaN in the target; forecast() has no such
requirement and is already confirmed tolerant there). Filling here would be
inconsistent with this project's own "cgm_glucose is never imputed" rule
(report.md Sec 13, Sec 15.7) for no documented reason.

MAX_CONTEXT MUST BE RAISED FOR THIS SCRIPT SPECIFICALLY
==========================================================
timesfm_zeroshot_eval.py compiles TimesFM with max_context=1024 (1024 bins =
85.3h = 3.56 days), sized for its own 144-bin windows. A 7-day history
(2016 bins) plus a 144-bin window context is 2160 bins, which exceeds that.
This script compiles its own ForecastConfig with max_context=4096 (comfortably
above the largest arm's 2160, a power of two matching the existing 1024/256
convention, well under the model's real 16384-bin ceiling) -- confirmed via
web search against the model's own documentation, not assumed.

GLUCOSE-ONLY, MATCHING THE PROJECT'S OWN LORA DECISION
=========================================================
No covariates in this arm, for the same reason chronos2_personalization.py's
Phase 1 and the population-level LoRA decision are both glucose-only:
TimesFM's covariate mechanism (forecast_with_covariates's non-trainable xreg
residual regression) is architecturally orthogonal to in-context history
injection -- combining them is a real follow-up experiment, not this one.
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
from build_participant_grid import load_cleaned_glucose_binned  # noqa: E402

CONTEXT_BINS = 144
BIN_MINUTES = 5


def history_array(person_id, history_start, anchor):
    """Copy of chronos2_personalization.py's function of the same name --
    see this module's docstring for why it's copied, not imported. The
    participant's own glucose over [history_start, anchor), upper bound
    exclusive, reindexed onto a complete 5-minute grid so missing bins are
    explicit NaN at their true position rather than silently closing the
    gap. NaN is left in (see module docstring), matching forecast()'s
    already-confirmed tolerance of raw NaN in glucose-only context."""
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
HORIZON_BINS = {"30min": 6, "1h": 12, "2h": 24, "3h": 36, "4h": 48}
CONDITION = "glucose_only"

SPLIT = os.environ.get("TIMESFM_ICL_SPLIT", "val")
EVAL_SET_PATH = f"{REPO_ROOT}/model_input/personalization/eval_set_{SPLIT}.csv"
EVAL_DIR = (f"{REPO_ROOT}/model_input/zeroshot_eval" if SPLIT == "test"
            else f"{REPO_ROOT}/model_input/zeroshot_eval_{SPLIT}")

HISTORY_DAYS = int(os.environ.get("TIMESFM_ICL_HISTORY_DAYS", "7"))
MAX_CONTEXT = int(os.environ.get("TIMESFM_ICL_MAX_CONTEXT", "4096"))

SHARD = int(os.environ.get("TIMESFM_ICL_SHARD", "0"))
NUM_SHARDS = int(os.environ.get("TIMESFM_ICL_NUM_SHARDS", "1"))
MAX_PARTICIPANTS = os.environ.get("TIMESFM_ICL_MAX_PARTICIPANTS")
MAX_PARTICIPANTS = int(MAX_PARTICIPANTS) if MAX_PARTICIPANTS else None

OUTPUT_DIR = os.environ.get(
    "TIMESFM_ICL_RESULTS_DIR",
    f"{REPO_ROOT}/timesfm_personalization_results/{SPLIT}/incontext_{HISTORY_DAYS}d",
)


def load_eval_set():
    e = pd.read_csv(EVAL_SET_PATH, dtype={"person_id": str, "window_id": str})
    e["anchor"] = pd.to_datetime(e["anchor"], utc=True)
    history_col = f"history_start_{HISTORY_DAYS}d"
    if history_col not in e.columns:
        raise SystemExit(f"eval set has no {history_col}; "
                          f"TIMESFM_ICL_HISTORY_DAYS={HISTORY_DAYS} is not one of 1/3/7")
    e[history_col] = pd.to_datetime(e[history_col], utc=True)
    return e, history_col


def load_export(horizon, keep_window_ids):
    """Mirrors chronos2_personalization.py's load_export() exactly -- same
    eval directory, same restriction, same tz handling."""
    path = f"{EVAL_DIR}/zeroshot_{horizon}_{CONDITION}.csv"
    df = pd.read_csv(path, dtype={"window_id": str, "person_id": str})
    df = df[df["window_id"].isin(keep_window_ids)].copy()
    df["bin_start"] = pd.to_datetime(df["bin_start"], utc=True)
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


def predict_participant(model, history_values, export_by_horizon, window_ids_by_horizon, person_id):
    """For every eval window this participant has, at every horizon: build
    [history_values ++ this window's own 144-bin context], forecast once
    with TimesFM's plain zero-shot forecast(), score against truth. One
    model.forecast() call per horizon, batched across all of this
    participant's windows at that horizon (all inputs the same length --
    fixed history length + fixed 144-bin context -- so batching is safe)."""
    out = {}
    for horizon, horizon_bins in HORIZON_BINS.items():
        wanted = window_ids_by_horizon.get((person_id, horizon))
        if not wanted:
            continue
        df = export_by_horizon[horizon]
        df = df[df["window_id"].isin(wanted)]
        if df.empty:
            continue

        window_ids, inputs, truths = [], [], []
        for wid, wdf in df.groupby("window_id"):
            wdf = wdf.sort_values("bin_start")
            ctx = wdf.loc[wdf["is_context"], "cgm_glucose"].to_numpy(dtype=np.float32)
            tgt = wdf.loc[~wdf["is_context"], ["bin_start", "cgm_glucose"]].reset_index(drop=True)
            if len(ctx) != CONTEXT_BINS:
                continue
            full_context = np.concatenate([history_values, ctx])
            window_ids.append(wid)
            inputs.append(full_context)
            truths.append(tgt)

        if not inputs:
            continue

        point_forecast, _quantiles = model.forecast(horizon=horizon_bins, inputs=inputs)

        rows = []
        for wid, truth, pred in zip(window_ids, truths, point_forecast):
            merged = truth.rename(columns={"cgm_glucose": "actual"}).copy()
            merged["window_id"] = wid
            merged["predictions"] = np.asarray(pred)[:horizon_bins]
            rows.append(merged)
        merged_df = pd.concat(rows, ignore_index=True)
        merged_df["abs_error"] = (merged_df["predictions"] - merged_df["actual"]).abs()
        out[horizon] = merged_df
    return out


def main():
    job_start = time.time()
    e, history_col = load_eval_set()

    window_to_person = dict(zip(e["window_id"], e["person_id"]))
    window_ids_by_horizon = (e.groupby(["person_id", "horizon"])["window_id"]
                              .apply(set).to_dict())
    anchors = e.groupby("person_id").agg(anchor=("anchor", "first"),
                                          history_start=(history_col, "first"))

    all_participants = sorted(anchors.index)
    participants = [p for i, p in enumerate(all_participants) if i % NUM_SHARDS == SHARD]
    if MAX_PARTICIPANTS is not None:
        participants = participants[:MAX_PARTICIPANTS]

    os.makedirs(OUTPUT_DIR, exist_ok=True)
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
          f"({len(todo)} to do), split={SPLIT}, k={HISTORY_DAYS}d, max_context={MAX_CONTEXT}",
          file=sys.stderr)

    load_start = time.time()
    model = timesfm.TimesFM_2p5_200M_torch.from_pretrained("google/timesfm-2.5-200m-pytorch")
    model.compile(timesfm.ForecastConfig(
        max_context=MAX_CONTEXT, max_horizon=256, normalize_inputs=True,
        use_continuous_quantile_head=True, fix_quantile_crossing=True,
    ))
    model_load_seconds = round(time.time() - load_start, 1)
    print(f"model loaded + compiled (max_context={MAX_CONTEXT}) in {model_load_seconds}s",
          file=sys.stderr)

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

        preds = predict_participant(model, values, export_by_horizon, window_ids_by_horizon, pid)
        for horizon, merged in preds.items():
            path = f"{OUTPUT_DIR}/predictions_{horizon}_{CONDITION}.csv"
            merged.to_csv(path, mode="a", header=not os.path.exists(path), index=False)

        pd.DataFrame([{"person_id": pid, "history_bins": len(values),
                        "expected_bins": expected_bins,
                        "real_fraction": round(float(np.isfinite(values).mean()), 4)}]).to_csv(
            done_path, mode="a", header=not os.path.exists(done_path), index=False)
        n_done += 1

        if (i + 1) % 10 == 0 or i + 1 == len(todo):
            print(f"  {i + 1}/{len(todo)} participants, "
                  f"{time.time() - job_start:.0f}s elapsed", file=sys.stderr)

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
            "history_days": HISTORY_DAYS, "max_context": MAX_CONTEXT,
            "mae_mgdl": scored["abs_error"].mean(),
            "rmse_mgdl": float(np.sqrt((scored["abs_error"] ** 2).mean())),
            "n_scored": len(scored), "n_total": len(full),
            "n_windows": full["window_id"].nunique(),
            "n_participants_this_shard": n_done,
            "model_load_seconds": model_load_seconds,
            "cumulative_job_seconds": round(time.time() - job_start, 1),
        })
    if rows:
        summary = pd.DataFrame(rows)
        summary.to_csv(f"{OUTPUT_DIR}/summary_shard{SHARD}.csv", index=False)
        print(summary.to_string(index=False), file=sys.stderr)


if __name__ == "__main__":
    main()
