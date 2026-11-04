"""Persistence baseline computed on SSM-CGM's OWN evaluation windows.

WHY THIS EXISTS. T9 shows SSM-CGM's test MAE (10.63/14.32/18.98 at
30min/1h/2h) sitting above the persistence baseline (8.26/12.36/17.04). That
comparison is NOT valid as it stands: the naive baselines are scored on the
shared glucose-only window set (~4,680 non-overlapping windows at 30min),
while SSM-CGM is scored on ~662k stride-1 overlapping windows built by
pytorch_forecasting's TimeSeriesDataSet. Different window sets, different
window counts, different overlap structure. a collaborator's alignment audit
(NAIVE_ALIGNMENT_CHECK_FULL_*.md) excluded SSM-CGM for exactly this reason.

So this script does not try to reconcile the two keyings. It recomputes
persistence *inside* SSM-CGM's own pipeline: same TimeSeriesDataSet, same
segment filter, same batching, same `x` dict. For every window the prediction
is simply the last observed context glucose, `x["encoder_target"][:, -1]`,
broadcast across the forecast horizon.

Both the persistence prediction and the ground truth are pulled from the SAME
`x` dict that ssmcgm_test_eval.py takes `actual` from
(`x["decoder_target"]`), so they are on the same scale by construction and no
denormalization assumption is needed. That is the whole point: the comparison
is internally consistent even if the absolute scaling convention were
misunderstood.

NO MODEL IS LOADED. This is CPU-only and touches no checkpoint.

Set SSMCGM_HORIZON=24 (default) to match the headline test_eval_h24 run, or
12 to match the original test_eval run. MIN_SEGMENT_LEN follows automatically
so the window set matches whichever run is being compared against.
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
from pytorch_forecasting import TimeSeriesDataSet, GroupNormalizer  # noqa: E402
from pytorch_forecasting.data.encoders import NaNLabelEncoder  # noqa: E402
from build_ssmcgm_input import (  # noqa: E402
    STATIC_CATEGORICALS, STATIC_REALS, TIME_VARYING_KNOWN_REALS,
    TIME_VARYING_UNKNOWN_CATEGORICALS, TIME_VARYING_UNKNOWN_REALS,
)

TRAIN_CSV = f"{REPO_ROOT}/model_input/ssmcgm/ssmcgm_train.csv"
TEST_CSV = f"{REPO_ROOT}/model_input/ssmcgm/ssmcgm_test.csv"
CONTEXT_LEN = 144
HORIZON = int(os.environ.get("SSMCGM_HORIZON", "24"))
BATCH_SIZE = 128
NUM_WORKERS = 2
MIN_SEGMENT_LEN = CONTEXT_LEN + HORIZON
OUTPUT_DIR = os.environ.get(
    "SSMCGM_PERSISTENCE_OUTPUT_DIR",
    f"{REPO_ROOT}/ssmcgm_train_results/persistence_matched_h{HORIZON}")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_and_type(path):
    df = pd.read_csv(path)
    df["participant_id"] = df["participant_id"].astype(str)
    seg_lens = df.groupby("series_id").size()
    qualifying = seg_lens[seg_lens >= MIN_SEGMENT_LEN].index
    n_before = df["series_id"].nunique()
    df = df[df["series_id"].isin(qualifying)].copy()
    print(f"  {path}: kept {len(qualifying)}/{n_before} segments >= {MIN_SEGMENT_LEN} rows",
          file=sys.stderr)
    return df


print(f"HORIZON={HORIZON}, MIN_SEGMENT_LEN={MIN_SEGMENT_LEN}", file=sys.stderr)
train_df = load_and_type(TRAIN_CSV)
test_df = load_and_type(TEST_CSV)

# Identical construction to ssmcgm_test_eval.py. `training` exists only to
# donate its fitted encoders/normalizers, exactly as there.
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
print(f"test samples: {len(testing)}", file=sys.stderr)
test_dataloader = testing.to_dataloader(train=False, batch_size=BATCH_SIZE,
                                        num_workers=NUM_WORKERS)

out_path = f"{OUTPUT_DIR}/predictions_h{HORIZON}.csv"
if os.path.exists(out_path):
    os.remove(out_path)
header_written = False
n_windows = 0

with torch.no_grad():
    for batch in test_dataloader:
        x, _ = batch
        enc = x["encoder_target"].cpu().numpy()      # (batch, CONTEXT_LEN)
        actuals = x["decoder_target"].cpu().numpy()  # (batch, HORIZON)
        dec_len = x["decoder_lengths"].cpu().numpy()
        last_obs = enc[:, -1]                        # persistence prediction
        idx = testing.x_to_index(x)

        rows = []
        for i in range(actuals.shape[0]):
            L = int(dec_len[i])
            sid = idx.iloc[i]["series_id"] if "series_id" in idx.columns else None
            for h in range(L):
                rows.append({
                    "series_id": sid,
                    "sample_idx": n_windows + i,
                    "horizon_bin": h,
                    "persistence": float(last_obs[i]),
                    "actual": float(actuals[i, h]),
                })
        n_windows += actuals.shape[0]
        if not rows:
            continue
        bdf = pd.DataFrame(rows)
        bdf["abs_error"] = (bdf["persistence"] - bdf["actual"]).abs()
        bdf.to_csv(out_path, mode="a", header=not header_written, index=False)
        header_written = True

pred = pd.read_csv(out_path)
pred["abs_error"] = (pred["persistence"] - pred["actual"]).abs()

SUB = [("30min", 6), ("1h", 12), ("2h", 24)]
rows = []
for name, bins in SUB:
    if bins > HORIZON:
        continue
    sub = pred[pred["horizon_bin"] < bins].dropna(subset=["actual"])
    rows.append({
        "horizon": name,
        "method": "persistence_matched",
        "mae_mgdl": sub["abs_error"].mean(),
        "rmse_mgdl": float(np.sqrt((sub["abs_error"] ** 2).mean())),
        "n_scored": len(sub),
        "n_windows": n_windows,
        "n_test_series": pred["series_id"].nunique(),
    })
summary = pd.DataFrame(rows)
summary.to_csv(f"{OUTPUT_DIR}/summary.csv", index=False)
print(summary.to_string(index=False), file=sys.stderr)

# Side-by-side against SSM-CGM's own numbers on the same window construction.
model_summary = (f"{REPO_ROOT}/ssmcgm_train_results/"
                 f"{'test_eval_h24' if HORIZON == 24 else 'test_eval'}/summary.csv")
if os.path.exists(model_summary):
    m = pd.read_csv(model_summary)
    mc = next((c for c in m.columns if "mae" in c.lower()), None)
    print("\n=== MATCHED COMPARISON (same windows) ===", file=sys.stderr)
    print(f"{'horizon':<8}{'SSM-CGM':>10}{'persistence':>14}{'delta':>10}  verdict", file=sys.stderr)
    comp = []
    for _, r in summary.iterrows():
        mr = m[m["horizon"] == r["horizon"]]
        if mr.empty or not mc:
            continue
        mv = float(mr.iloc[0][mc])
        d = mv - r["mae_mgdl"]
        verdict = "SSM-CGM WORSE than persistence" if d > 0 else "SSM-CGM better"
        print(f"{r['horizon']:<8}{mv:>10.3f}{r['mae_mgdl']:>14.3f}{d:>+10.3f}  {verdict}",
              file=sys.stderr)
        comp.append({"horizon": r["horizon"], "ssmcgm_mae": mv,
                     "persistence_mae": r["mae_mgdl"], "delta_mae": d,
                     "ssmcgm_beats_persistence": bool(d < 0),
                     "n_scored": int(r["n_scored"])})
    if comp:
        pd.DataFrame(comp).to_csv(f"{OUTPUT_DIR}/matched_comparison.csv", index=False)
        print(f"\nwrote {OUTPUT_DIR}/matched_comparison.csv", file=sys.stderr)

print("SSMCGM_PERSISTENCE_MATCHED_OK", file=sys.stderr)
