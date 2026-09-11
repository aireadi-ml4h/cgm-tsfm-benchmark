"""Variable-importance extraction for SSM-CGM checkpoints, on VAL data only.

Does not touch the held-out test set (job 42290315 stays held; this script
never reads a *_test.csv file).

Uses pytorch_forecasting's own TemporalFusionTransformer.interpret_output()
(venv-ssmcgm/lib/python3.12/site-packages/pytorch_forecasting/models/
temporal_fusion_transformer/_tft.py:699), inherited unmodified by MambaTFT.
forward() (code/external/SSM-CGM/SSM_CGM.py:707-713) returns static_variables,
encoder_variables, decoder_variables -- exactly what interpret_output()
consumes. reduction="sum" returns unnormalized per-variable sums across the
batch; dividing by the total gives the same 0-1 proportions the SSM-CGM
paper reports in its own Table 5 / Fig 2A.

Variable name order comes from the model's own self.static_variables /
self.encoder_variables / self.decoder_variables (set by pytorch_forecasting's
TFT base class from the TimeSeriesDataSet's own categoricals/reals lists at
from_dataset() time) -- read directly from the loaded checkpoint's
hyperparameters, not reconstructed from build_ssmcgm_input.py's lists, so
this can't drift out of sync with what the checkpoint actually has.
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
)

CONTEXT_LEN = 144
HORIZON = 12
MIN_SEGMENT_LEN = CONTEXT_LEN + HORIZON
BATCH_SIZE = 128
NUM_WORKERS = 2

# Two checkpoints, VAL data only for each -- matched pairs, never test.
RUNS = [
    dict(
        name="disjoint_with_cov",
        ckpt=f"{REPO_ROOT}/ssmcgm_train_results/12h_batch128/best-epoch=0-val_loss=12.3597.ckpt",
        train_csv=f"{REPO_ROOT}/model_input/ssmcgm/ssmcgm_train.csv",
        val_csv=f"{REPO_ROOT}/model_input/ssmcgm/ssmcgm_val.csv",
    ),
    dict(
        name="proportional_with_cov",
        ckpt=f"{REPO_ROOT}/ssmcgm_train_results/proportional_mode/best-epoch=0-val_loss=14.3673.ckpt",
        train_csv=f"{REPO_ROOT}/model_input/ssmcgm/ssmcgm_train_proportional.csv",
        val_csv=f"{REPO_ROOT}/model_input/ssmcgm/ssmcgm_val_proportional.csv",
    ),
]

OUTPUT_DIR = f"{REPO_ROOT}/ssmcgm_train_results/variable_importance"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_and_type(path):
    df = pd.read_csv(path)
    df["participant_id"] = df["participant_id"].astype(str)
    seg_lens = df.groupby("series_id").size()
    qualifying = seg_lens[seg_lens >= MIN_SEGMENT_LEN].index
    df = df[df["series_id"].isin(qualifying)].copy()
    return df


def build_training_dataset(train_df):
    return TimeSeriesDataSet(
        train_df, time_idx="ds", target="cgm_glucose", group_ids=["series_id"],
        max_encoder_length=CONTEXT_LEN, max_prediction_length=HORIZON,
        min_encoder_length=CONTEXT_LEN, min_prediction_length=HORIZON,
        static_categoricals=STATIC_CATEGORICALS, static_reals=STATIC_REALS,
        time_varying_known_reals=TIME_VARYING_KNOWN_REALS,
        time_varying_unknown_categoricals=TIME_VARYING_UNKNOWN_CATEGORICALS,
        time_varying_unknown_reals=TIME_VARYING_UNKNOWN_REALS,
        target_normalizer=GroupNormalizer(groups=["series_id"]),
        categorical_encoders={
            "series_id": NaNLabelEncoder(add_nan=True),
            "participant_id": NaNLabelEncoder(add_nan=True),
        },
        add_relative_time_idx=True, add_target_scales=True, add_encoder_length=True,
    )


def run_one(cfg, device):
    print(f"\n=== {cfg['name']} ===", file=sys.stderr)
    train_df = load_and_type(cfg["train_csv"])
    val_df = load_and_type(cfg["val_csv"])
    training = build_training_dataset(train_df)
    validation = TimeSeriesDataSet.from_dataset(training, val_df, stop_randomization=True)
    val_dl = validation.to_dataloader(train=False, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)

    tft = SSM_CGM.MambaTFT.load_from_checkpoint(cfg["ckpt"])
    tft.eval()
    tft.to(device)

    static_names = list(tft.static_variables)
    encoder_names = list(tft.encoder_variables)
    decoder_names = list(tft.decoder_variables)

    static_sum = torch.zeros(len(static_names))
    encoder_sum = torch.zeros(len(encoder_names))
    decoder_sum = torch.zeros(len(decoder_names))
    n_batches = 0

    with torch.no_grad():
        for batch in val_dl:
            x, _ = batch
            x = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in x.items()}
            out = tft(x)
            interp = tft.interpret_output(out, reduction="sum")
            static_sum += interp["static_variables"].cpu()
            encoder_sum += interp["encoder_variables"].cpu()
            decoder_sum += interp["decoder_variables"].cpu()
            n_batches += 1
            if n_batches % 200 == 0:
                print(f"  {n_batches} batches", file=sys.stderr)

    static_frac = (static_sum / static_sum.sum()).numpy()
    encoder_frac = (encoder_sum / encoder_sum.sum()).numpy()
    decoder_frac = (decoder_sum / decoder_sum.sum()).numpy()

    rows = []
    for n, w in zip(static_names, static_frac):
        rows.append({"run": cfg["name"], "group": "static", "variable": n, "importance": float(w)})
    for n, w in zip(encoder_names, encoder_frac):
        rows.append({"run": cfg["name"], "group": "encoder", "variable": n, "importance": float(w)})
    for n, w in zip(decoder_names, decoder_frac):
        rows.append({"run": cfg["name"], "group": "decoder", "variable": n, "importance": float(w)})
    return pd.DataFrame(rows)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    all_rows = []
    for cfg in RUNS:
        df = run_one(cfg, device)
        all_rows.append(df)
        out_path = f"{OUTPUT_DIR}/{cfg['name']}_importance.csv"
        df.sort_values(["group", "importance"], ascending=[True, False]).to_csv(out_path, index=False)
        print(df.sort_values(["group", "importance"], ascending=[True, False]).to_string(index=False), file=sys.stderr)

    combined = pd.concat(all_rows, ignore_index=True)
    combined.to_csv(f"{OUTPUT_DIR}/all_runs_importance.csv", index=False)
    print("SSMCGM_VARIABLE_IMPORTANCE_OK", file=sys.stderr)


if __name__ == "__main__":
    main()
