"""Real test-set forecasting evaluation for SSM-CGM, on the true best
checkpoint from the real training run (ssmcgm_train_real.py).

USES best-epoch=0-val_loss=12.3597.ckpt, NOT the checkpoint the training
script itself printed as "best model" at the end of its run. The
ModelCheckpoint(save_top_k=3) callback's in-memory ranking resets every time
the Slurm job was requeued/resumed (three times across this run's real
history), so after the final resume it only ever compared epochs 8-15
against each other and had no memory that epoch 0's val_loss=12.3597 (from
before the first resume) was ever computed -- it reported
best-epoch=6-val_loss=13.6278 as "best," which is worse (13.63 > 12.36) than
a checkpoint still sitting on disk. Verified directly: `ls
ssmcgm_train_results/12h_batch128/*.ckpt` still shows the epoch-0 file with
the lower loss. This script hardcodes the true global-minimum checkpoint
rather than trusting the training run's own final printout.

HORIZON: 12 (1h, default) scores 30min+1h from one forward pass (30min is
just bins 0-5 of the same 12-bin decoder output -- no second pass needed).
Set SSMCGM_TEST_HORIZON=24 to score 2h instead, using the separate h24
retrain's checkpoint (SSMCGM_TEST_CKPT must be set to that checkpoint's
path when doing so) -- this reuses the SAME ssmcgm_train.csv/ssmcgm_test.csv
population, only the decoder length and checkpoint differ. 3h/4h have no
SSM-CGM equivalent; the model has never been trained past HORIZON=24 (2h).

Mirrors ssmcgm_train_real.py's exact TimeSeriesDataSet construction for
`training` (same schema, same categorical_encoders) because
TimeSeriesDataSet.from_dataset(training, test_df, ...) requires the
identical fitted training object to share its encoders/normalizers with the
held-out test set -- rebuilding `training` here from ssmcgm_train.csv is not
redundant work, it is the only correct way to get a test dataset scored on
the same target scale the checkpoint was trained on.
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

REPO_ROOT = os.environ["AIREADI_ROOT"]
sys.path.insert(0, os.path.join(REPO_ROOT, "code/external/SSM-CGM"))
# Sibling modules live in adjacent topic directories under code/.
_CODE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _d in ("analysis", "baselines", "cgmlsm", "chronos2", "meal_detection", "model_input", "ssmcgm", "timesfm"):
    sys.path.insert(0, os.path.join(_CODE, _d))
sys.path.insert(0, os.path.join(REPO_ROOT, "code/model_input"))
import SSM_CGM  # noqa: E402
from pytorch_forecasting import TimeSeriesDataSet, GroupNormalizer  # noqa: E402
from pytorch_forecasting.data.encoders import NaNLabelEncoder  # noqa: E402
from pytorch_forecasting.metrics import QuantileLoss  # noqa: E402
from build_ssmcgm_input import (  # noqa: E402
    STATIC_CATEGORICALS, STATIC_REALS, TIME_VARYING_KNOWN_REALS,
    TIME_VARYING_UNKNOWN_CATEGORICALS, TIME_VARYING_UNKNOWN_REALS,
    TIME_VARYING_UNKNOWN_REALS_CGM_ONLY,
)

TRAIN_CSV = os.environ.get("SSMCGM_TRAIN_CSV",
                           f"{REPO_ROOT}/model_input/ssmcgm/ssmcgm_train.csv")
TEST_CSV = os.environ.get("SSMCGM_TEST_CSV", f"{REPO_ROOT}/model_input/ssmcgm/ssmcgm_test.csv")
CKPT_PATH = os.environ.get(
    "SSMCGM_TEST_CKPT",
    f"{REPO_ROOT}/ssmcgm_train_results/12h_batch128/best-epoch=0-val_loss=12.3597.ckpt",
)
OUTPUT_DIR = os.environ.get("SSMCGM_TEST_OUTPUT_DIR", f"{REPO_ROOT}/ssmcgm_train_results/test_eval")

# SSMCGM_NO_COVARIATES=1 -- must mirror ssmcgm_train_real.py's stripping
# EXACTLY, or the TimeSeriesDataSet built here will not match the schema the
# ablation checkpoints were trained against and load_from_checkpoint will
# either error on a shape mismatch or, worse, score against the wrong inputs.
# Copied verbatim from that script's block; static identity fields stay in
# both arms because they are who the person is, not a wearable signal.
NO_COVARIATES = os.environ.get("SSMCGM_NO_COVARIATES", "0") == "1"
if NO_COVARIATES:
    TIME_VARYING_KNOWN_REALS = [c for c in TIME_VARYING_KNOWN_REALS
                                if c != "oxygen_saturation_applicable"]
    TIME_VARYING_UNKNOWN_CATEGORICALS = []          # drops sleep_stage
    TIME_VARYING_UNKNOWN_REALS = TIME_VARYING_UNKNOWN_REALS_CGM_ONLY
print(f"NO_COVARIATES={NO_COVARIATES}", file=sys.stderr)

CONTEXT_LEN = 144
HORIZON = int(os.environ.get("SSMCGM_TEST_HORIZON", 12))  # 12=1h (default), 24=2h
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


print("loading real train/test data...", file=sys.stderr)
train_df = load_and_type(TRAIN_CSV)
test_df = load_and_type(TEST_CSV)
print(f"train: {len(train_df)} rows, {train_df['participant_id'].nunique()} participants, "
      f"{train_df['series_id'].nunique()} segments", file=sys.stderr)
print(f"test:  {len(test_df)} rows, {test_df['participant_id'].nunique()} participants, "
      f"{test_df['series_id'].nunique()} segments", file=sys.stderr)

# Rebuild `training` EXACTLY as ssmcgm_train_real.py did -- only used to
# donate its fitted encoders/normalizers to the test TimeSeriesDataSet
# below; never trained on again here.
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
testing = TimeSeriesDataSet.from_dataset(training, test_df, stop_randomization=True)
print(f"TimeSeriesDataSet built: {len(training)} train samples, {len(testing)} test samples", file=sys.stderr)

test_dataloader = testing.to_dataloader(
    train=False, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS,
)

print(f"loading checkpoint: {CKPT_PATH}", file=sys.stderr)
tft = SSM_CGM.MambaTFT.load_from_checkpoint(CKPT_PATH)
tft.eval()
device = "cuda" if torch.cuda.is_available() else "cpu"
tft.to(device)

print("running test-set forecast (ONE forward pass per batch, streamed to disk "
      "incrementally)...", file=sys.stderr)
# FIX 2026-09-08: tft.predict(test_dataloader, return_x=True, return_index=True)
# buffers EVERY window's full input/output/index in memory for the WHOLE
# dataloader before returning anything (verified directly against
# pytorch_forecasting's PredictCallback source: self._output/_x_list/_index
# are plain lists appended once per batch, never flushed, only returned at
# the end via predict_callback.result). This is the exact pattern that caused
# two real OOM crashes in ssmcgm_val_eval_multihorizon.py once its val set
# reached ~22,736 windows; this test set has a comparable number of raw rows
# (763,799) and this job requests only 32GB, so the same crash is a real risk
# here that was never exercised before (this script had never successfully
# run to completion). Replaced with the same manual batch-streaming pattern
# already verified correct there: same underlying tft(x) forward pass, same
# tft.loss.to_quantiles() conversion (confirmed to reduce to an identical
# call as the official pl_module.to_quantiles() path), and the official
# per-sample NaN-mask-by-decoder_length is reproduced by only ever emitting
# rows for h < dec_len rather than masking-then-dropping later.
import re
_ckpt_loss_match = re.search(r"val_loss=([\d.]+)\.ckpt", CKPT_PATH)
CKPT_VAL_LOSS = float(_ckpt_loss_match.group(1)) if _ckpt_loss_match else None
RAW_HORIZON_LABEL = {12: "1h", 24: "2h"}.get(HORIZON, f"{HORIZON}bins")

median_idx = None
out_path = f"{OUTPUT_DIR}/predictions_{RAW_HORIZON_LABEL}.csv"
if os.path.exists(out_path):
    os.remove(out_path)  # mode="a" below must start from a clean file, not a stale/failed run's leftovers
header_written = False
n_rows_written = 0
with torch.no_grad():
    for batch in test_dataloader:
        x, _ = batch
        x = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in x.items()}
        out = tft(x)
        quantile_preds = tft.loss.to_quantiles(out["prediction"])  # (batch, HORIZON, n_quantiles)
        if median_idx is None:
            median_idx = tft.loss.quantiles.index(0.5)
        median_preds = quantile_preds[:, :, median_idx].cpu().numpy()
        actuals = x["decoder_target"].cpu().numpy()
        decoder_lengths = x["decoder_lengths"].cpu().numpy()
        batch_index = testing.x_to_index(x)

        rows = []
        for i in range(median_preds.shape[0]):
            dec_len = int(decoder_lengths[i])
            series_id = batch_index.iloc[i]["series_id"] if "series_id" in batch_index.columns else None
            for h in range(dec_len):
                rows.append({
                    "series_id": series_id,
                    "sample_idx": n_rows_written + i,
                    "horizon_bin": h,
                    "predictions": float(median_preds[i, h]),
                    "actual": float(actuals[i, h]),
                })
        n_rows_written += median_preds.shape[0]
        if not rows:
            continue  # this batch's windows were all zero-length; nothing to write
        batch_df = pd.DataFrame(rows)
        batch_df["abs_error"] = (batch_df["predictions"] - batch_df["actual"]).abs()
        batch_df.to_csv(out_path, mode="a", header=not header_written, index=False)
        header_written = True

pred_df = pd.read_csv(out_path)
pred_df["abs_error"] = (pred_df["predictions"] - pred_df["actual"]).abs()
pred_df.to_csv(out_path, index=False)

# Sub-horizon slices are all scored from the SAME decoder output already
# written above -- e.g. bins 0-5 of a 24-bin (2h) forecast ARE the 30min
# forecast, bins 0-11 ARE the 1h forecast, no second forward pass needed.
# Mirrors ssmcgm_val_eval_multihorizon.py's identical slicing.
SUB_HORIZONS = [("30min", 6), ("1h", 12), ("2h", 24)]
summary_rows = []
for horizon, bins in SUB_HORIZONS:
    if bins > HORIZON:
        continue  # this checkpoint's decoder doesn't reach this horizon
    hdf = pred_df[pred_df["horizon_bin"] < bins]
    if bins < HORIZON:
        hdf.to_csv(f"{OUTPUT_DIR}/predictions_{horizon}.csv", index=False)
    scored = hdf.dropna(subset=["actual"])
    mae = scored["abs_error"].mean()
    rmse = float(np.sqrt((scored["abs_error"] ** 2).mean()))
    n_series = hdf["series_id"].nunique() if "series_id" in hdf.columns else None
    summary_rows.append({
        "horizon": horizon, "checkpoint": CKPT_PATH, "val_loss_at_checkpoint": CKPT_VAL_LOSS,
        "mae_mgdl": mae, "rmse_mgdl": rmse, "n_scored": len(scored), "n_total": len(hdf),
        "n_test_series": n_series,
    })

summary = pd.DataFrame(summary_rows)
summary.to_csv(f"{OUTPUT_DIR}/summary.csv", index=False)
print(summary.to_string(index=False), file=sys.stderr)
print("SSMCGM_TEST_EVAL_OK", file=sys.stderr)
