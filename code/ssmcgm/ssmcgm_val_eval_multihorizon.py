"""VAL-ONLY multi-horizon scoring for SSM-CGM's original disjoint baseline
checkpoint (best-epoch=0-val_loss=12.3597.ckpt). Does NOT touch the test set
-- ssmcgm_val.csv only, mirroring ssmcgm_test_eval.py's structure exactly but
scored on the model's own validation split, since the real test-set
evaluation (job 42290315) is deliberately held pending further debugging.

WHY 30MIN NEEDS NO RETRAINING: the model's decoder produces all HORIZON=12
steps (1h) in a single forward pass (see ssmcgm_test_eval.py's
`quantile_preds` shape (n_samples, HORIZON, 3)). The first 6 of those 12
already-computed steps ARE a genuine 30-minute-ahead forecast -- scoring is
just a different slice of output that already exists, not a new model
capability. This script reports both 30min (horizon_bin < 6) and 1h
(horizon_bin < 12) from the ONE forward pass, to confirm that slicing
produces sane numbers (30min MAE should be clearly lower than 1h MAE,
matching every other model in this project) before anyone trusts this
approach for a real test-set run later.

2h is NOT attempted here -- the model's decoder was trained with
max_prediction_length=HORIZON=12, hard-baked into the architecture; there is
no way to extract 24 steps from a 12-step decoder without retraining at
HORIZON=24 (see ssmcgm_train_real.py's SSMCGM_HORIZON override, added
alongside this script, and the corresponding new training run).
"""
import os
import sys
import types

import numpy as np
import pandas as pd
import torch

google_mod = types.ModuleType("google")
cloud_mod = types.ModuleType("google.cloud")
storage_mod = types.ModuleType("google.cloud.storage")


class _StubGCSClient:
    def __init__(self, *args, **kwargs):
        pass

    def bucket(self, *args, **kwargs):
        raise RuntimeError("GCS is stubbed out; no checkpoint/data GCS calls are exercised.")


storage_mod.Client = _StubGCSClient
cloud_mod.storage = storage_mod
google_mod.cloud = cloud_mod
sys.modules["google"] = google_mod
sys.modules["google.cloud"] = cloud_mod
sys.modules["google.cloud.storage"] = storage_mod

REPO_ROOT = "/path/to/aireadi-collab"
sys.path.insert(0, os.path.join(REPO_ROOT, "code/external/SSM-CGM"))
# Sibling modules live in adjacent topic directories under code/.
_CODE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _d in ("analysis", "baselines", "cgmlsm", "chronos2", "meal_detection", "model_input", "ssmcgm", "timesfm"):
    sys.path.insert(0, os.path.join(_CODE, _d))
sys.path.insert(0, os.path.join(REPO_ROOT, "code/model_input"))
import SSM_CGM  # noqa: E402
from pytorch_forecasting import TimeSeriesDataSet, GroupNormalizer  # noqa: E402
from pytorch_forecasting.data.encoders import NaNLabelEncoder  # noqa: E402
from build_ssmcgm_input import (  # noqa: E402
    STATIC_CATEGORICALS, STATIC_REALS, TIME_VARYING_KNOWN_REALS,
    TIME_VARYING_UNKNOWN_CATEGORICALS, TIME_VARYING_UNKNOWN_REALS,
    TIME_VARYING_UNKNOWN_REALS_CGM_ONLY,
)

# Must match whatever the loaded checkpoint was actually trained with --
# ssmcgm_train_real.py's SSMCGM_NO_COVARIATES=1 ablation arm swaps this same
# list; scoring a no-covariates checkpoint against the full-covariate schema
# would silently mismatch the TimeSeriesDataSet the checkpoint expects.
if os.environ.get("SSMCGM_VAL_EVAL_NO_COVARIATES", "0") == "1":
    TIME_VARYING_UNKNOWN_REALS = TIME_VARYING_UNKNOWN_REALS_CGM_ONLY

TRAIN_CSV = os.environ.get("SSMCGM_VAL_EVAL_TRAIN_CSV", f"{REPO_ROOT}/model_input/ssmcgm/ssmcgm_train.csv")
VAL_CSV = os.environ.get("SSMCGM_VAL_EVAL_VAL_CSV", f"{REPO_ROOT}/model_input/ssmcgm/ssmcgm_val.csv")  # VAL ONLY, never test
CKPT_PATH = os.environ.get(
    "SSMCGM_VAL_EVAL_CKPT",
    f"{REPO_ROOT}/ssmcgm_train_results/12h_batch128/best-epoch=0-val_loss=12.3597.ckpt",
)
OUTPUT_DIR = os.environ.get("SSMCGM_VAL_EVAL_OUTPUT_DIR", f"{REPO_ROOT}/ssmcgm_train_results/val_eval_multihorizon")

CONTEXT_LEN = 144
HORIZON = 12  # the checkpoint's own trained decoder length; unchanged
BATCH_SIZE = 128
NUM_WORKERS = 2
MIN_SEGMENT_LEN = CONTEXT_LEN + HORIZON

os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_and_type(path):
    df = pd.read_csv(path)
    df["participant_id"] = df["participant_id"].astype(str)
    seg_lens = df.groupby("series_id").size()
    qualifying = seg_lens[seg_lens >= MIN_SEGMENT_LEN].index
    n_before = df["series_id"].nunique()
    df = df[df["series_id"].isin(qualifying)].copy()
    print(f"  {path}: kept {len(qualifying)}/{n_before} segments >= {MIN_SEGMENT_LEN} rows", file=sys.stderr)
    return df


print("loading real train/val data (VAL ONLY -- test set is not touched by this script)...", file=sys.stderr)
train_df = load_and_type(TRAIN_CSV)
val_df = load_and_type(VAL_CSV)
print(f"train: {len(train_df)} rows, {train_df['participant_id'].nunique()} participants, "
      f"{train_df['series_id'].nunique()} segments", file=sys.stderr)
print(f"val:   {len(val_df)} rows, {val_df['participant_id'].nunique()} participants, "
      f"{val_df['series_id'].nunique()} segments", file=sys.stderr)

training = TimeSeriesDataSet(
    train_df,
    time_idx="ds",
    target="cgm_glucose",
    group_ids=["series_id"],
    max_encoder_length=CONTEXT_LEN,
    max_prediction_length=HORIZON,
    min_encoder_length=CONTEXT_LEN,
    min_prediction_length=HORIZON,
    static_categoricals=STATIC_CATEGORICALS,
    static_reals=STATIC_REALS,
    time_varying_known_reals=TIME_VARYING_KNOWN_REALS,
    time_varying_unknown_categoricals=TIME_VARYING_UNKNOWN_CATEGORICALS,
    time_varying_unknown_reals=TIME_VARYING_UNKNOWN_REALS,
    target_normalizer=GroupNormalizer(groups=["series_id"]),
    categorical_encoders={
        "series_id": NaNLabelEncoder(add_nan=True),
        "participant_id": NaNLabelEncoder(add_nan=True),
    },
    add_relative_time_idx=True,
    add_target_scales=True,
    add_encoder_length=True,
)
validation = TimeSeriesDataSet.from_dataset(training, val_df, stop_randomization=True)
print(f"TimeSeriesDataSet built: {len(training)} train samples, {len(validation)} val samples", file=sys.stderr)

val_dataloader = validation.to_dataloader(train=False, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)

print(f"loading checkpoint: {CKPT_PATH}", file=sys.stderr)
tft = SSM_CGM.MambaTFT.load_from_checkpoint(CKPT_PATH)
tft.eval()
device = "cuda" if torch.cuda.is_available() else "cpu"
tft.to(device)

print("running val-set forecast (ONE forward pass per batch, scored at both 30min and 1h, "
      "streamed to disk incrementally)...", file=sys.stderr)
# FIX 2026-09-07: tft.predict(val_dataloader, return_x=True, return_index=True)
# buffers EVERY window's full input/output/index in memory for the WHOLE
# dataloader before returning anything -- fine for the original ~1,752-window
# val set this pattern was copied from (ssmcgm_test_eval.py), but this run's
# val set is ~22,736 windows (13x larger, since the paper-mode fix widened
# the withheld region -- see report.md Sec 11), and that one call OOM'd twice
# (32GB then 96GB) before this was caught. Fixed by predicting and writing
# ONE BATCH AT A TIME instead of materializing the whole dataset in memory --
# same underlying model.predict_step, just not buffered across the full loop.
out_path = f"{OUTPUT_DIR}/predictions_val_multihorizon.csv"
header_written = False
n_rows_written = 0
with torch.no_grad():
    for batch in val_dataloader:
        x, _ = batch
        x = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in x.items()}
        out = tft(x)
        quantile_preds = tft.loss.to_quantiles(out["prediction"])  # (batch, HORIZON, n_quantiles)
        median_idx = tft.loss.quantiles.index(0.5)
        median_preds = quantile_preds[:, :, median_idx].cpu().numpy()
        actuals = x["decoder_target"].cpu().numpy()
        decoder_lengths = x["decoder_lengths"].cpu().numpy()

        rows = []
        for i in range(median_preds.shape[0]):
            dec_len = int(decoder_lengths[i])
            for h in range(dec_len):
                rows.append({
                    "sample_idx": n_rows_written + i,
                    "horizon_bin": h,  # 0-indexed 5-min bin: bins 0-5 = 30min, bins 0-11 = 1h
                    "predictions": float(median_preds[i, h]),
                    "actual": float(actuals[i, h]),
                })
        batch_df = pd.DataFrame(rows)
        batch_df["abs_error"] = (batch_df["predictions"] - batch_df["actual"]).abs()
        batch_df.to_csv(out_path, mode="a", header=not header_written, index=False)
        header_written = True
        n_rows_written += median_preds.shape[0]

pred_df = pd.read_csv(out_path)

summaries = []
for label, max_bin in [("30min", 6), ("1h", 12)]:
    sl = pred_df[pred_df["horizon_bin"] < max_bin]
    scored = sl.dropna(subset=["actual"])
    mae = scored["abs_error"].mean()
    rmse = float(np.sqrt((scored["abs_error"] ** 2).mean()))
    summaries.append({
        "horizon": label, "checkpoint": CKPT_PATH, "split": "val",
        "mae_mgdl": mae, "rmse_mgdl": rmse, "n_scored": len(scored), "n_total": len(sl),
    })

summary = pd.DataFrame(summaries)
summary.to_csv(f"{OUTPUT_DIR}/summary_val_multihorizon.csv", index=False)
print(summary.to_string(index=False), file=sys.stderr)
print("SSMCGM_VAL_EVAL_MULTIHORIZON_OK", file=sys.stderr)
