"""Persistence and linear-extrapolation naive baselines for SSM-CGM, on the
IDENTICAL val-set windows/iteration order used by
ssmcgm_val_eval_multihorizon.py (best-epoch=0-val_loss=12.3597.ckpt real
result: 30min MAE=14.438, 1h MAE=17.523). VAL ONLY -- the held test set
(job 42290315) is not touched by this script.

Does not import SSM_CGM / load any checkpoint: only pytorch_forecasting's
TimeSeriesDataSet is needed to reconstruct the same encoder/decoder windows,
so this runs CPU-only, unlike every script that imports SSM_CGM (which
requires a GPU just to import mamba_ssm's Triton kernels -- confirmed
twice this session). Rebuilds the same `training`/`validation`
TimeSeriesDataSet objects, byte-for-byte the same construction as
ssmcgm_val_eval_multihorizon.py, then iterates val_dataloader (same
batch_size, shuffle=False) to pull x["encoder_target"]/x["encoder_lengths"]
(the real, un-normalized context values) and x["decoder_target"]/
x["decoder_lengths"] (actuals) directly -- no model forward pass needed for
a naive baseline. Iteration order is deterministic and matches the real
eval script's row order exactly (same dataset, same batch_size, shuffle=
False; num_workers only parallelizes __getitem__, not batch order).
"""
import os
import sys

import numpy as np
import pandas as pd

REPO_ROOT = "/path/to/aireadi-collab"
sys.path.insert(0, os.path.join(REPO_ROOT, "code/model_input"))
# Sibling modules live in adjacent topic directories under code/.
_CODE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _d in ("analysis", "baselines", "cgmlsm", "chronos2", "meal_detection", "model_input", "ssmcgm", "timesfm"):
    sys.path.insert(0, os.path.join(_CODE, _d))
from pytorch_forecasting import TimeSeriesDataSet, GroupNormalizer  # noqa: E402
from pytorch_forecasting.data.encoders import NaNLabelEncoder  # noqa: E402
from build_ssmcgm_input import (  # noqa: E402
    STATIC_CATEGORICALS, STATIC_REALS, TIME_VARYING_KNOWN_REALS,
    TIME_VARYING_UNKNOWN_CATEGORICALS, TIME_VARYING_UNKNOWN_REALS,
)

TRAIN_CSV = f"{REPO_ROOT}/model_input/ssmcgm/ssmcgm_train.csv"
VAL_CSV = f"{REPO_ROOT}/model_input/ssmcgm/ssmcgm_val.csv"
OUTPUT_DIR = f"{REPO_ROOT}/ssmcgm_train_results/val_eval_multihorizon"

CONTEXT_LEN = 144
HORIZON = 12
BATCH_SIZE = 128
MIN_SEGMENT_LEN = CONTEXT_LEN + HORIZON
LINEAR_FIT_BINS = 6  # 30 min, matching the other two naive-baseline scripts


def load_and_type(path):
    df = pd.read_csv(path)
    df["participant_id"] = df["participant_id"].astype(str)
    seg_lens = df.groupby("series_id").size()
    qualifying = seg_lens[seg_lens >= MIN_SEGMENT_LEN].index
    df = df[df["series_id"].isin(qualifying)].copy()
    return df


print("loading real train/val data (VAL ONLY -- test set is not touched)...", file=sys.stderr)
train_df = load_and_type(TRAIN_CSV)
val_df = load_and_type(VAL_CSV)

training = TimeSeriesDataSet(
    train_df, time_idx="ds", target="cgm_glucose", group_ids=["series_id"],
    max_encoder_length=CONTEXT_LEN, max_prediction_length=HORIZON,
    min_encoder_length=CONTEXT_LEN, min_prediction_length=HORIZON,
    static_categoricals=STATIC_CATEGORICALS, static_reals=STATIC_REALS,
    time_varying_known_reals=TIME_VARYING_KNOWN_REALS,
    time_varying_unknown_categoricals=TIME_VARYING_UNKNOWN_CATEGORICALS,
    time_varying_unknown_reals=TIME_VARYING_UNKNOWN_REALS,
    target_normalizer=GroupNormalizer(groups=["series_id"]),
    categorical_encoders={"series_id": NaNLabelEncoder(add_nan=True),
                           "participant_id": NaNLabelEncoder(add_nan=True)},
    add_relative_time_idx=True, add_target_scales=True, add_encoder_length=True,
)
validation = TimeSeriesDataSet.from_dataset(training, val_df, stop_randomization=True)
val_dataloader = validation.to_dataloader(train=False, batch_size=BATCH_SIZE, num_workers=0)
print(f"TimeSeriesDataSet built: {len(validation)} val samples", file=sys.stderr)

rows = []
n_seen = 0
for x, _ in val_dataloader:
    enc_target = x["encoder_target"].numpy()      # (batch, encoder_len)
    enc_lengths = x["encoder_lengths"].numpy()     # (batch,) real (unpadded) length
    dec_target = x["decoder_target"].numpy()       # (batch, HORIZON)
    dec_lengths = x["decoder_lengths"].numpy()

    for i in range(enc_target.shape[0]):
        elen = int(enc_lengths[i])
        # No padding is actually possible here: min_encoder_length ==
        # max_encoder_length == CONTEXT_LEN (144) in this TimeSeriesDataSet,
        # so every real sample has elen == 144, the full array -- confirmed
        # against the vendored source (rnn.pad_sequence defaults to
        # right-padding, i.e. real values first; a slice from either end
        # is identical whenever elen equals the full array length, which is
        # always true here). enc_target[i, :elen] used defensively in case
        # this script is ever reused with a variable-length encoder.
        real_ctx = enc_target[i, :elen] if elen > 0 else enc_target[i, :0]
        valid = real_ctx[~np.isnan(real_ctx)]
        if len(valid) == 0:
            n_seen += 1
            continue
        anchor = valid[-1]
        tail = real_ctx[-LINEAR_FIT_BINS:]
        idx = np.arange(len(tail))[~np.isnan(tail)]
        slope = np.polyfit(idx, tail[~np.isnan(tail)], 1)[0] if len(idx) >= 2 else 0.0

        dlen = int(dec_lengths[i])
        for h in range(dlen):
            rows.append({
                "sample_idx": n_seen, "horizon_bin": h,
                "actual": float(dec_target[i, h]),
                "persistence": float(anchor),
                "linear": float(anchor + slope * (h + 1)),
            })
        n_seen += 1

naive = pd.DataFrame(rows)
naive.to_csv(f"{OUTPUT_DIR}/naive_baselines_val.csv", index=False)
print(f"wrote {len(naive)} rows, {n_seen} samples", file=sys.stderr)

real_pred = pd.read_csv(f"{OUTPUT_DIR}/predictions_val_multihorizon.csv")

out_rows = []
for horizon_name, max_bin in [("30min", 6), ("1h", 12)]:
    real_h = real_pred[real_pred["horizon_bin"] < max_bin].dropna(subset=["actual"])
    naive_h = naive[naive["horizon_bin"] < max_bin].dropna(subset=["actual"])

    real_mae = (real_h["predictions"] - real_h["actual"]).abs().mean()
    real_rmse = float(np.sqrt(((real_h["predictions"] - real_h["actual"]) ** 2).mean()))
    pers_mae = (naive_h["persistence"] - naive_h["actual"]).abs().mean()
    pers_rmse = float(np.sqrt(((naive_h["persistence"] - naive_h["actual"]) ** 2).mean()))
    lin_mae = (naive_h["linear"] - naive_h["actual"]).abs().mean()
    lin_rmse = float(np.sqrt(((naive_h["linear"] - naive_h["actual"]) ** 2).mean()))

    for method, mae, rmse, n in [
        ("model", real_mae, real_rmse, len(real_h)),
        ("persistence", pers_mae, pers_rmse, len(naive_h)),
        ("linear_extrap", lin_mae, lin_rmse, len(naive_h)),
    ]:
        out_rows.append({"model": "ssmcgm", "horizon": horizon_name, "method": method,
                          "mae": round(float(mae), 4), "rmse": round(float(rmse), 4), "n_bins": n})

    # Paired cluster-by-participant test, model vs persistence, on matched
    # (sample_idx, horizon_bin) pairs -- sample_idx maps 1:1 to a val window,
    # not directly to participant_id here (not carried through the
    # dataloader batch), so this reports the per-sample (window-level)
    # paired test rather than a cluster-by-participant one; documented as a
    # real limitation, not silently presented as equivalent to the
    # cluster-by-participant tests used elsewhere in this project.
    merged = real_h.merge(naive_h, on=["sample_idx", "horizon_bin"], suffixes=("_real", "_naive"))
    d = (merged["predictions"] - merged["actual_real"]).abs() - (merged["persistence"] - merged["actual_naive"]).abs()
    from scipy import stats
    t_stat, p_val = stats.ttest_1samp(d.to_numpy(), 0.0)
    out_rows.append({"model": "ssmcgm", "horizon": horizon_name, "method": "sig_vs_persistence",
                      "mae": None, "rmse": None, "n_bins": len(merged),
                      "paired_t": round(float(t_stat), 4), "paired_p": float(p_val)})

pd.DataFrame(out_rows).to_csv(f"{OUTPUT_DIR}/naive_vs_model_ssmcgm.csv", index=False)
print(pd.DataFrame(out_rows).to_string(index=False))
print("\nNAIVE_BASELINES_SSMCGM_OK")
