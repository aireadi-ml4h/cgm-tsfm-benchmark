"""Evaluates the trained CGM-LSM checkpoint (code/cgmlsm/cgmlsm_train.py, Slurm job
41045299, $SCRATCH/aireadi_cgmlsm/train_v1/final) over
model_input/cgmlsm/cgmlsm_test.csv, autoregressively forecasting each
window's horizon from its 144-bin (12h) context via model.generate() and
scoring MAE/RMSE against the real cgm_glucose ground truth -- the trained-
model analog of code/chronos2/chronos2_zeroshot_eval.py's zero-shot scoring, same
chunked/resumable/checkpointed pattern.

Same windows as Chronos-2's glucose_only eval, not just comparable ones:
cgmlsm_test.csv was built directly from window_index.csv's own
(person_id, context_start) test-split rows -- the exact pairs
export_zeroshot_windows.py used for zeroshot_<horizon>_glucose_only.csv --
and this was verified by an exact row-level merge before training started
(0 mismatches on a sample, see report.md Sec 15.6 / model_input/HANDOFF.md).
This script does not re-derive or re-filter those windows in any way; it
only reads cgmlsm_test.csv as-is, so whatever population that file has is
exactly what gets scored here.

Tokens outside the CGMValue vocab's numeric range (10-400) -- PAD, UNKNOWN,
MASK, Other_0..6 -- decode to NaN mg/dL, not a fabricated number: a model
predicting "UNKNOWN" at a horizon step means "I don't have a real forecast
here," and forcing that into a number would understate the model's own
uncertainty rather than report it. These NaN predictions are dropped at
scoring time via the same dropna-before-MAE/RMSE convention
chronos2_zeroshot_eval.py already uses for missing ground truth, applied
here to predictions instead (ground truth cgm_glucose in the test file has
its own, separate, much rarer residual-NaN case, handled the same way).
"""
import os
import sys
import time

import numpy as np
import pandas as pd
import torch

REPO_ROOT = "/path/to/aireadi-collab"
sys.path.append(os.path.join(REPO_ROOT, "code/external/cgmlsm/code/pipeline"))
# Sibling modules live in adjacent topic directories under code/.
_CODE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _d in ("analysis", "baselines", "cgmlsm", "chronos2", "meal_detection", "model_input", "ssmcgm", "timesfm"):
    sys.path.insert(0, os.path.join(_CODE, _d))
sys.path.append(os.path.join(REPO_ROOT, "code/model_input"))

from nn.cgmlsm.modeling_cgmlsm import CgmLsmLMHeadModel  # noqa: E402
from cgmlsm_vocab import CGM_IDX2TKN, CGM_TKN2IDX, CGM_PAD_IDX  # noqa: E402
from windows import CONTEXT_BINS, HORIZONS_BINS  # noqa: E402

# Overridable so a fine-tuning recipe can be SELECTED on val windows
# (cgmlsm_val_windows.csv) and only then REPORTED on test windows -- the
# original sweeps selected on cgmlsm_test.csv, i.e. on the reporting set.
TEST_PATH = os.environ.get(
    "CGMLSM_EVAL_INPUT", os.path.join(REPO_ROOT, "model_input", "cgmlsm", "cgmlsm_test.csv")
)
CHECKPOINT_DIR = os.environ.get(
    "CGMLSM_CHECKPOINT_DIR",
    os.path.join(os.environ["SCRATCH"], "aireadi_cgmlsm", "train_v1", "final"),
)
OUTPUT_DIR = os.environ.get("CGMLSM_EVAL_OUTPUT_DIR", os.path.join(REPO_ROOT, "cgmlsm_results", "from_scratch"))
CHUNK_SIZE = 500  # windows per generate() batch / per checkpoint, same convention as chronos2_zeroshot_eval.py

EVAL_HORIZONS = {k: v for k, v in HORIZONS_BINS.items() if k in ("30min", "1h", "2h")}


def token_to_mgdl(token):
    """Inverse of cgmlsm_vocab.tokenize_glucose: numeric CGMValue tokens ->
    float mg/dL, everything else (PAD/UNKNOWN/MASK/Other_*) -> NaN."""
    return float(token) if token.isdigit() else np.nan


def already_done_window_ids(output_path):
    if not os.path.exists(output_path):
        return set()
    return set(pd.read_csv(output_path, usecols=["window_id"], dtype={"window_id": str})["window_id"])


def run_chunk(model, device, df_chunk, horizon_bins):
    context = df_chunk[df_chunk["is_context"]].sort_values(["window_id", "bin_start"])
    horizon = df_chunk[~df_chunk["is_context"]].sort_values(["window_id", "bin_start"])

    window_ids = context["window_id"].unique()
    ctx_ids = context["cgm_token"].astype(str).map(CGM_TKN2IDX)
    if ctx_ids.isna().any():
        raise ValueError("found cgm_token value(s) outside CGMValue vocab in context")
    input_ids = torch.tensor(
        ctx_ids.to_numpy(dtype=np.int64).reshape(len(window_ids), CONTEXT_BINS), device=device
    )

    with torch.no_grad():
        generated = model.generate(
            input_ids=input_ids, max_new_tokens=horizon_bins, do_sample=False, pad_token_id=CGM_PAD_IDX,
        )
    forecast_ids = generated[:, CONTEXT_BINS:].cpu().numpy()

    rows = []
    for i, window_id in enumerate(window_ids):
        w_horizon = horizon[horizon["window_id"] == window_id]
        for j, (_, r) in enumerate(w_horizon.iterrows()):
            token = CGM_IDX2TKN[forecast_ids[i, j]]
            rows.append({
                "window_id": window_id, "bin_start": r["bin_start"],
                "prediction_token": token, "predictions": token_to_mgdl(token),
                "actual": r["cgm_glucose"],
            })
    merged = pd.DataFrame(rows)
    merged["abs_error"] = (merged["predictions"] - merged["actual"]).abs()
    return merged


def run_one(model, device, df, horizon_name, horizon_bins):
    combo_start = time.time()
    output_path = os.path.join(OUTPUT_DIR, f"predictions_{horizon_name}.csv")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    df_h = df[df["horizon"] == horizon_name]
    all_window_ids = df_h["window_id"].unique()
    if len(all_window_ids) == 0:
        # Only expected when a caller's window subsample (CGMLSM_MAX_WINDOWS)
        # happens to not include this horizon at all -- the full test file
        # always has windows for every EVAL_HORIZONS entry (build_cgmlsm_input.py
        # would have logged a WARNING and skipped writing the file otherwise).
        print(f"{horizon_name}: 0 windows in this subsample, skipping", file=sys.stderr)
        return {"horizon": horizon_name, "mae_mgdl": float("nan"), "rmse_mgdl": float("nan"),
                "n_scored": 0, "n_total": 0, "n_nonnumeric_predictions": 0,
                "elapsed_seconds_this_run": 0.0, "n_windows_processed_this_run": 0}

    done = already_done_window_ids(output_path)
    remaining = [w for w in all_window_ids if w not in done]
    if done:
        print(f"{horizon_name}: resuming, {len(done)}/{len(all_window_ids)} windows already done", file=sys.stderr)

    for start in range(0, len(remaining), CHUNK_SIZE):
        chunk_ids = set(remaining[start:start + CHUNK_SIZE])
        df_chunk = df_h[df_h["window_id"].isin(chunk_ids)]
        merged = run_chunk(model, device, df_chunk, horizon_bins)
        header = not os.path.exists(output_path)
        merged.to_csv(output_path, mode="a", header=header, index=False)
        print(f"{horizon_name}: checkpointed {start + len(chunk_ids)}/{len(remaining)} remaining windows "
              f"({len(done) + start + len(chunk_ids)}/{len(all_window_ids)} total)", file=sys.stderr)

    full = pd.read_csv(output_path, dtype={"window_id": str})
    scored = full.dropna(subset=["actual", "predictions"])
    n_total, n_scored = len(full), len(scored)
    mae = scored["abs_error"].mean()
    rmse = float(np.sqrt((scored["abs_error"] ** 2).mean()))
    n_nonnumeric_pred = full["predictions"].isna().sum()
    elapsed = time.time() - combo_start
    print(
        f"{horizon_name}: DONE MAE={mae:.3f} mg/dL RMSE={rmse:.3f} mg/dL "
        f"(scored {n_scored}/{n_total} horizon bins, {n_nonnumeric_pred} predictions were non-numeric tokens) "
        f"in {elapsed:.1f}s this run ({len(remaining)} windows newly processed, {len(done)} already done)",
        file=sys.stderr,
    )
    return {
        "horizon": horizon_name, "mae_mgdl": mae, "rmse_mgdl": rmse,
        "n_scored": n_scored, "n_total": n_total, "n_nonnumeric_predictions": int(n_nonnumeric_pred),
        "elapsed_seconds_this_run": round(elapsed, 1), "n_windows_processed_this_run": len(remaining),
    }


def main():
    max_windows = os.environ.get("CGMLSM_MAX_WINDOWS")
    max_windows = int(max_windows) if max_windows else None

    job_start = time.time()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = CgmLsmLMHeadModel.from_pretrained(CHECKPOINT_DIR).to(device)
    model.eval()
    model_load_seconds = round(time.time() - job_start, 1)
    print(f"model loaded in {model_load_seconds}s on {device}", file=sys.stderr)

    df = pd.read_csv(TEST_PATH, dtype={"window_id": str, "person_id": str})
    df["bin_start"] = pd.to_datetime(df["bin_start"], utc=True)
    df["is_context"] = df["is_context"].astype(bool)
    if max_windows is not None:
        # Capped per horizon, not globally -- a global head() cut would only
        # ever keep the first horizon's windows, since cgmlsm_test.csv is
        # written horizon-by-horizon (build_cgmlsm_input.py), silently
        # starving every other horizon's smoke-test coverage.
        keep = pd.concat([g["window_id"].drop_duplicates().head(max_windows)
                           for _, g in df.groupby("horizon")])
        df = df[df["window_id"].isin(keep)]

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    summary_path = os.path.join(OUTPUT_DIR, "summary.csv")

    rows = []
    for horizon_name, horizon_bins in EVAL_HORIZONS.items():
        row = run_one(model, device, df, horizon_name, horizon_bins)
        row["model_load_seconds"] = model_load_seconds
        row["cumulative_job_seconds"] = round(time.time() - job_start, 1)
        rows.append(row)
        pd.DataFrame(rows).to_csv(summary_path, index=False)

    print(pd.DataFrame(rows).to_string(index=False), file=sys.stderr)


if __name__ == "__main__":
    main()
