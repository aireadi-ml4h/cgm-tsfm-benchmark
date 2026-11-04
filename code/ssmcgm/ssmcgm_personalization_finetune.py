"""SSM-CGM personalization: isolated per-participant fine-tuning of SSM-CGM's
SHARED (non-identity) weights, on top of the frozen population-trained base
checkpoint (best-epoch=0-val_loss=12.3597.ckpt) -- mirrors
code/chronos2/chronos2_personalization.py (Chronos-2 per-participant LoRA) and
code/timesfm/timesfm_lora_personalization.py (TimesFM per-participant LoRA): same
personalization eval set, same anchors, same k=1/3/7-day history convention,
same "one adaptation per person, base model reused untouched for the next
person" isolation guarantee -- so all three arms are directly comparable.

WHY NOT THE EMBEDDING-TABLE-EXPANSION APPROACH
================================================
SSM-CGM's participant_id embedding table (self.input_embeddings.embeddings[
"participant_id"], a pytorch_forecasting MultiEmbedding -> nn.Embedding,
confirmed via source read of temporal_fusion_transformer/_tft.py:206 and
models/nn/embeddings.py) is sized to the ORIGINAL disjoint training
population only. A personalization-eval participant is unseen by that
population, so (verified directly against pytorch_forecasting/data/
encoders.py's NaNLabelEncoder.transform, add_nan=True branch: `encoded =
[self.classes_.get(v, 0) for v in y]`) they are automatically mapped to class
0 -- the shared "unknown" embedding row -- by TimeSeriesDataSet.from_dataset(),
with a UserWarning, no crash, and no resizing. This is what makes fine-tuning
SSM-CGM on a new person possible at all without the embedding-table surgery
this project's earlier personalization attempt (report.md Sec 7) found too
risky to get right silently. It is also not a new risk: ssmcgm_test_eval.py's
own disjoint test set is, by construction, entirely unseen by training too,
and scores fine through this exact mechanism.

THE ONE THING THIS DESIGN DEPENDS ON: EVERY unseen participant maps to the
SAME shared row. If that row's weights were fine-tuned along with everything
else, adapting the model to one person would corrupt the "unknown" bucket for
whoever is fine-tuned next -- invisible cross-person leakage through a shared
parameter, worse than doing nothing. So the participant_id embedding is
FROZEN (requires_grad=False) before every participant's fine-tune, and the
freeze is verified, not assumed: this script clones that embedding's weight
tensor before fine-tuning and asserts torch.equal() against the clone
afterward, every single time, for every participant. Everything else in the
model (Mamba blocks, TFT variable-selection, all other embeddings) is fine-
tuned normally.

HORIZON: 1h ONLY. SSM-CGM was trained to forecast 1h ahead (HORIZON=12 bins),
not the 30min-4h suite the zero-shot/Chronos-2/TimesFM arms cover -- a real,
disclosed scope limit inherited from ssmcgm_test_eval.py's module docstring,
not something this script works around.

ISOLATION: a fresh copy of the frozen base checkpoint is loaded for every
(participant, k) pair; nothing is carried over between participants or
between k values for the same participant.

EVAL WINDOWS COME FROM build_participant_series, NOT THE ZEROSHOT EXPORT
CSVS. The zeroshot_eval export CSVs (used by Chronos-2/TimesFM) carry raw
glucose + raw covariates only -- SSM-CGM's schema needs the engineered
cgm_lag_*/cgm_rolling_*/cgm_diff_* columns build_ssmcgm_input.py computes.
build_participant_series computes those over the participant's FULL raw
history BEFORE any time_range slicing (verified by reading the function:
lag/rolling assignment happens before the cutoff/time_range filter), so
slicing to exactly [context_start, context_start + MIN_SEGMENT_LEN bins) for
one eval window yields correctly-computed features for that window, no
separate code path.

BOTH from_dataset() CALLS (FINE-TUNE AND EVAL) DONATE BASE_TRAINING'S FITTED
ENCODERS/NORMALIZERS, stop_randomization=True IN BOTH -- this only affects
whether categorical target-encoding is REFIT (it is not, in either case; both
paths reuse encoders/scales fitted once on the real training population,
mirroring ssmcgm_test_eval.py's own testing = TimeSeriesDataSet.from_dataset(
training, test_df, stop_randomization=True) exactly, whose test participants
are equally unseen by training). Training-vs-eval behavior is controlled by
to_dataloader(train=True/False), not by this flag.
"""
import os
import sys
import time
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
import lightning.pytorch as pl  # noqa: E402
from pytorch_forecasting import TimeSeriesDataSet, GroupNormalizer  # noqa: E402
from pytorch_forecasting.data.encoders import NaNLabelEncoder  # noqa: E402
from build_ssmcgm_input import (  # noqa: E402
    STATIC_CATEGORICALS, STATIC_REALS, TIME_VARYING_KNOWN_REALS,
    TIME_VARYING_UNKNOWN_CATEGORICALS, TIME_VARYING_UNKNOWN_REALS,
    build_participant_series,
)
from build_participant_grid import load_participants_covariates  # noqa: E402

CONTEXT_LEN = 144
HORIZON = 12  # 1h, the only horizon SSM-CGM was trained to forecast
MIN_SEGMENT_LEN = CONTEXT_LEN + HORIZON
BIN_MINUTES = 5

TRAIN_CSV = f"{REPO_ROOT}/model_input/ssmcgm/ssmcgm_train.csv"
BASE_CKPT = os.environ.get(
    "SSMCGM_PERS_BASE_CKPT",
    f"{REPO_ROOT}/ssmcgm_train_results/12h_batch128/best-epoch=0-val_loss=12.3597.ckpt",
)
SPLIT = os.environ.get("SSMCGM_PERS_SPLIT", "test")
EVAL_SET_PATH = f"{REPO_ROOT}/model_input/personalization/eval_set_{SPLIT}.csv"

HISTORY_DAYS_LIST = [int(k) for k in os.environ.get("SSMCGM_PERS_HISTORY_DAYS", "1,3,7").split(",")]
LEARNING_RATE = float(os.environ.get("SSMCGM_PERS_LR", "1e-4"))
NUM_STEPS = int(os.environ.get("SSMCGM_PERS_NUM_STEPS", "50"))
BATCH_SIZE = int(os.environ.get("SSMCGM_PERS_BATCH_SIZE", "16"))
SEED = int(os.environ.get("SSMCGM_PERS_SEED", "42"))

SHARD = int(os.environ.get("SSMCGM_PERS_SHARD", "0"))
NUM_SHARDS = int(os.environ.get("SSMCGM_PERS_NUM_SHARDS", "1"))
MAX_PARTICIPANTS = os.environ.get("SSMCGM_PERS_MAX_PARTICIPANTS")
MAX_PARTICIPANTS = int(MAX_PARTICIPANTS) if MAX_PARTICIPANTS else None

OUTPUT_DIR = os.environ.get("SSMCGM_PERS_OUTPUT_DIR", f"{REPO_ROOT}/ssmcgm_personalization_results/{SPLIT}")

BASE_TRAINING = None  # set once in main()


def load_and_type(path):
    df = pd.read_csv(path)
    df["participant_id"] = df["participant_id"].astype(str)
    return df


def build_base_training():
    """Rebuilds the SAME TimeSeriesDataSet ssmcgm_train_real.py /
    ssmcgm_test_eval.py construct from ssmcgm_train.csv -- donates its fitted
    encoders/normalizers (including the participant_id NaNLabelEncoder,
    add_nan=True) to every personalization participant's dataset below, so an
    unseen participant is handled exactly the way ssmcgm_test_eval.py's real
    disjoint-test-set participants already are (also all unseen -- SSM-CGM's
    disjoint test split was never in its own training population either)."""
    train_df = load_and_type(TRAIN_CSV)
    seg_lens = train_df.groupby("series_id").size()
    qualifying = seg_lens[seg_lens >= MIN_SEGMENT_LEN].index
    train_df = train_df[train_df["series_id"].isin(qualifying)].copy()
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


def participant_id_embedding(tft):
    return tft.input_embeddings.embeddings["participant_id"]


def freeze_participant_embedding(tft):
    """Freezes participant_id's embedding weights in place; everything else
    in the model keeps requires_grad=True. Returns a detached clone of the
    pre-freeze weights, for the post-fit bit-exact verification."""
    emb = participant_id_embedding(tft)
    before = emb.weight.detach().clone()
    for param in emb.parameters():
        param.requires_grad = False
    assert not emb.weight.requires_grad, "participant_id embedding did not freeze"
    return before


def verify_embedding_unchanged(tft, before):
    emb = participant_id_embedding(tft)
    after = emb.weight.detach()
    if not torch.equal(before.to(after.device), after):
        raise RuntimeError(
            "participant_id embedding CHANGED during fine-tuning despite being "
            "frozen -- cross-person leakage risk, aborting rather than scoring "
            "a silently-corrupted arm.")


def finetune_for_participant(history_df, device):
    """One isolated fine-tune: fresh copy of the frozen base checkpoint,
    participant_id embedding frozen, everything else trained for NUM_STEPS
    steps on this participant's own history slice only."""
    ft_dataset = TimeSeriesDataSet.from_dataset(BASE_TRAINING, history_df, stop_randomization=True)
    if len(ft_dataset) == 0:
        return None, 0, 0.0
    dl = ft_dataset.to_dataloader(train=True, batch_size=BATCH_SIZE, num_workers=0)

    torch.manual_seed(SEED)
    tft = SSM_CGM.MambaTFT.load_from_checkpoint(BASE_CKPT, learning_rate=LEARNING_RATE)
    emb_before = freeze_participant_embedding(tft)

    trainer = pl.Trainer(
        max_steps=NUM_STEPS, max_epochs=-1,
        accelerator="gpu" if torch.cuda.is_available() else "cpu", devices=1,
        precision="32-true", logger=False, enable_checkpointing=False,
        enable_progress_bar=False, num_sanity_val_steps=0, gradient_clip_val=1.0,
    )
    fit_start = time.time()
    trainer.fit(tft, train_dataloaders=dl)
    fit_seconds = time.time() - fit_start

    verify_embedding_unchanged(tft, emb_before)
    tft.eval()
    tft.to(device)
    return tft, len(ft_dataset), fit_seconds


def predict_window(tft, window_df, device):
    """One (context, 1h-target) window -> 12-bin median forecast + actuals.
    Mirrors ssmcgm_test_eval.py's predict path exactly (mode='quantiles',
    index 1 = the q=0.5 median)."""
    ds = TimeSeriesDataSet.from_dataset(BASE_TRAINING, window_df, stop_randomization=True)
    if len(ds) == 0:
        return None
    dl = ds.to_dataloader(train=False, batch_size=1, num_workers=0)
    raw = tft.predict(dl, mode="quantiles", return_x=True, return_index=True,
                       trainer_kwargs={"accelerator": device})
    median = raw.output[:, :, 1].cpu().numpy()
    actual = raw.x["decoder_target"].cpu().numpy()
    dec_len = raw.x["decoder_lengths"].cpu().numpy()
    rows = []
    for i in range(median.shape[0]):
        for h in range(int(dec_len[i])):
            rows.append({"horizon_bin": h, "predictions": float(median[i, h]),
                         "actual": float(actual[i, h])})
    return pd.DataFrame(rows) if rows else None


def load_eval_set():
    e = pd.read_csv(EVAL_SET_PATH, dtype={"person_id": str, "window_id": str})
    e = e[e["horizon"] == "1h"].copy()  # HORIZON: 1h only, see module docstring
    for col in ["context_start", "t0", "anchor"] + [c for c in e.columns if c.startswith("history_start_")]:
        e[col] = pd.to_datetime(e[col], utc=True)
    return e


def done_key(path):
    if not os.path.exists(path):
        return set()
    d = pd.read_csv(path, dtype={"person_id": str})
    return set(zip(d["person_id"], d["history_days"].astype(int)))


def main():
    job_start = time.time()
    global BASE_TRAINING
    print("building base training TimeSeriesDataSet from ssmcgm_train.csv...", file=sys.stderr)
    BASE_TRAINING = build_base_training()

    e = load_eval_set()
    covs = load_participants_covariates()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Sanity check, printed not asserted: personalization-eval participants
    # should be disjoint from SSM-CGM's own training population (same as
    # every other model's held-out test split in this project). If this is
    # ever 0 overlap it confirms the "unseen -> shared unknown row" mechanism
    # is exercised for every participant scored here, not a mix.
    train_pids = set(load_and_type(TRAIN_CSV)["participant_id"].unique())
    eval_pids = set(e["person_id"].unique())
    print(f"personalization eval participants (1h horizon): {len(eval_pids)}, "
          f"overlap with SSM-CGM training population: {len(train_pids & eval_pids)}",
          file=sys.stderr)

    all_participants = sorted(eval_pids)
    participants = [p for i, p in enumerate(all_participants) if i % NUM_SHARDS == SHARD]
    if MAX_PARTICIPANTS is not None:
        participants = participants[:MAX_PARTICIPANTS]

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    done_path = f"{OUTPUT_DIR}/done_shard{SHARD}.csv"
    done = done_key(done_path)
    print(f"shard {SHARD}/{NUM_SHARDS}: {len(participants)} participants, "
          f"k in {HISTORY_DAYS_LIST}, split={SPLIT}, lr={LEARNING_RATE}, steps={NUM_STEPS}",
          file=sys.stderr)

    n_fitted, fit_seconds_total, skipped = 0, 0.0, []
    for i, pid in enumerate(participants):
        pe = e[e["person_id"] == pid]
        anchor = pe["anchor"].iloc[0]
        cov = covs.get(str(pid), {})

        for k in HISTORY_DAYS_LIST:
            if (pid, k) in done:
                continue
            hist_col = f"history_start_{k}d"
            if hist_col not in pe.columns:
                continue
            history_start = pe[hist_col].iloc[0]

            ft_df = build_participant_series(pid, cov, time_range=(history_start, anchor))
            if ft_df is None:
                skipped.append((pid, k, "no_history_data"))
                continue
            seg_lens = ft_df.groupby("series_id").size()
            if (seg_lens >= MIN_SEGMENT_LEN).sum() == 0:
                skipped.append((pid, k, "history_too_short"))
                continue

            tft, n_windows, fit_seconds = finetune_for_participant(ft_df, device)
            if tft is None:
                skipped.append((pid, k, "no_finetune_windows"))
                continue
            fit_seconds_total += fit_seconds
            n_fitted += 1

            pred_rows = []
            for row in pe.itertuples(index=False):
                # BUG FIX 2026-09-05: this used to slice to
                # (context_start, t0 + 1h). t0 is NOT this window's target
                # start -- it is the participant's own record start, used
                # only to compute offset_days in build_personalization_eval_set.py
                # (t0_by_pid there is a per-person constant, unrelated to any
                # specific window). Confirmed directly: t0 comes out EARLIER
                # than context_start for every row, giving a negative-length
                # slice, so every window failed the length check below and
                # zero windows were ever scored in the smoke test. The real
                # per-window target start is context_start + CONTEXT_LEN bins
                # (the context is exactly CONTEXT_LEN bins by construction) --
                # verified empirically to yield exactly MIN_SEGMENT_LEN clean
                # rows in one segment for a real smoke-test window.
                window_end = row.context_start + pd.Timedelta(minutes=5 * MIN_SEGMENT_LEN)
                w_df = build_participant_series(
                    pid, cov, time_range=(row.context_start, window_end))
                if w_df is None:
                    continue
                seg_lens_w = w_df.groupby("series_id").size()
                if (seg_lens_w == MIN_SEGMENT_LEN).sum() == 0:
                    continue  # a gap inside this specific window; cannot be scored
                out = predict_window(tft, w_df, device)
                if out is None:
                    continue
                out["window_id"] = row.window_id
                pred_rows.append(out)

            if pred_rows:
                merged = pd.concat(pred_rows, ignore_index=True)
                merged["person_id"] = pid
                merged["history_days"] = k
                merged["abs_error"] = (merged["predictions"] - merged["actual"]).abs()
                path = f"{OUTPUT_DIR}/predictions_1h.csv"
                merged.to_csv(path, mode="a", header=not os.path.exists(path), index=False)

            pd.DataFrame([{"person_id": pid, "history_days": k, "n_finetune_windows": n_windows,
                            "n_eval_windows_scored": len(pred_rows), "fit_seconds": round(fit_seconds, 1)}]
                         ).to_csv(done_path, mode="a", header=not os.path.exists(done_path), index=False)

            del tft
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        if (i + 1) % 10 == 0 or i + 1 == len(participants):
            print(f"  {i + 1}/{len(participants)} participants, "
                  f"{fit_seconds_total / max(n_fitted, 1):.1f}s/fit avg, "
                  f"{time.time() - job_start:.0f}s elapsed", file=sys.stderr)

    if skipped:
        pd.DataFrame(skipped, columns=["person_id", "history_days", "reason"]).to_csv(
            f"{OUTPUT_DIR}/skipped_shard{SHARD}.csv", index=False)
        print(f"WARNING: skipped {len(skipped)} (participant, k) pairs: {skipped[:5]}", file=sys.stderr)

    rows = []
    path = f"{OUTPUT_DIR}/predictions_1h.csv"
    if os.path.exists(path):
        full = pd.read_csv(path, dtype={"window_id": str, "person_id": str})
        for k in HISTORY_DAYS_LIST:
            sub = full[full["history_days"] == k]
            if sub.empty:
                continue
            scored = sub.dropna(subset=["actual"])
            rows.append({
                "horizon": "1h", "split": SPLIT, "history_days": k,
                "learning_rate": LEARNING_RATE, "num_steps": NUM_STEPS, "batch_size": BATCH_SIZE,
                "mae_mgdl": scored["abs_error"].mean(),
                "rmse_mgdl": float(np.sqrt((scored["abs_error"] ** 2).mean())),
                "n_scored": len(scored), "n_total": len(sub),
                "n_windows": sub["window_id"].nunique(), "n_participants": sub["person_id"].nunique(),
                "mean_fit_seconds": round(fit_seconds_total / max(n_fitted, 1), 1),
                "cumulative_job_seconds": round(time.time() - job_start, 1),
            })
    if rows:
        summary = pd.DataFrame(rows)
        summary.to_csv(f"{OUTPUT_DIR}/summary_shard{SHARD}.csv", index=False)
        print(summary.to_string(index=False), file=sys.stderr)
    print("SSMCGM_PERSONALIZATION_OK", file=sys.stderr)


if __name__ == "__main__":
    main()
