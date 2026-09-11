"""Real SSM-CGM training run (not a smoketest/timing sweep) on this
project's sentinel-fill input (build_ssmcgm_input.py: all 1,355 train
participants kept, real predmeal_flag, real sleep_regularity_index,
oxygen_saturation filled -1.0 + oxygen_saturation_applicable flag) --
build_ssmcgm_input.py's own corrected schema (predmeal_flag/wearables moved
to time_varying_unknown, avoiding the future-covariate oracle issue in
SSM_CGM.py's own create_tft_dataloaders, see that module's docstring), not
their original schema, and grouped by `series_id` (per-participant
contiguous segments), not raw `participant_id`, per the allow_missing_
timesteps fix in ssmcgm_input_smoketest.py.

Two config choices made pragmatically, per instruction, each disclosed:

1. Context length: 12h (144 bins), not the paper's own best-performing 48h
   (576 bins) -- chosen to get a first real result fast; 48h is a real,
   named follow-up (build_ssmcgm_input.py's data already supports it, this
   script's CONTEXT_LEN is the only thing that needs to change).

2. Batch size: 128, not the paper's own 32 -- based on real timing evidence
   (ssmcgm_batchsize_sweep_v2.py/_sentinel.py: batch 32 measured ~11.5-58.7
   GPU-hours/epoch depending on GPU, entirely a dataloader bottleneck fixed
   by num_workers=2, not a true compute limit; batch 128-256 measured
   ~2-3 GPU-hours/epoch on both the drop-participants and sentinel-fill
   variants). NOT validated for whether it changes final model quality --
   only that it trains faster. Confirming that would require fully training
   both, which defeats the point of avoiding the slow one; flagged here so
   this remains a disclosed, deliberate choice, not an assumption.

Real validation (missing from every smoketest/sweep script so far): built
via TimeSeriesDataSet.from_dataset(training, val_df, stop_randomization=True)
-- the standard pytorch_forecasting idiom for a held-out set that shares
the training set's own fitted encoders/normalizers, so val loss isn't
computed against a differently-scaled target.

Checkpointing: every epoch (last.ckpt) plus best-val-loss
(best-{epoch}-{val_loss}.ckpt), both under OUTPUT_DIR, so a cluster
walltime cutoff can resume via SSMCGM_RESUME_CKPT rather than restarting a
multi-day run from scratch.
"""
import os
import sys
import types

import pandas as pd
import torch
import lightning.pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger

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
from pytorch_forecasting.metrics import QuantileLoss  # noqa: E402
from build_ssmcgm_input import (  # noqa: E402
    STATIC_CATEGORICALS, STATIC_REALS, TIME_VARYING_KNOWN_REALS,
    TIME_VARYING_UNKNOWN_CATEGORICALS, TIME_VARYING_UNKNOWN_REALS,
    TIME_VARYING_UNKNOWN_REALS_CGM_ONLY,
)

# SSMCGM_NO_COVARIATES=1: covariate ablation arm (2026-09-05) -- strips every
# wearable/physiological covariate, keeping only glucose (+ its own lag/diff/
# rolling features), person identity/demographics (static), and calendar
# features (time_varying_known_reals minus oxygen_saturation_applicable,
# itself a covariate-availability flag). Static identity fields
# (participant_id/clinical_site/study_group/age) are NOT covariates in the
# sense being tested here -- they are who the person is, not a wearable
# signal -- so they stay in both arms; only the with-vs-without split is on
# the physiological/wearable channels. Default (unset) reproduces the
# original script's behavior exactly, unchanged.
NO_COVARIATES = os.environ.get("SSMCGM_NO_COVARIATES", "0") == "1"
if NO_COVARIATES:
    TIME_VARYING_KNOWN_REALS = [c for c in TIME_VARYING_KNOWN_REALS if c != "oxygen_saturation_applicable"]
    TIME_VARYING_UNKNOWN_CATEGORICALS = []  # drops sleep_stage
    TIME_VARYING_UNKNOWN_REALS = TIME_VARYING_UNKNOWN_REALS_CGM_ONLY

OUTPUT_DIR = os.environ.get("SSMCGM_OUTPUT_DIR", f"{REPO_ROOT}/ssmcgm_train_results/12h_batch128")
TRAIN_CSV = os.environ.get("SSMCGM_TRAIN_CSV") or f"{REPO_ROOT}/model_input/ssmcgm/ssmcgm_train.csv"
VAL_CSV = os.environ.get("SSMCGM_VAL_CSV") or f"{REPO_ROOT}/model_input/ssmcgm/ssmcgm_val.csv"
RESUME_CKPT = os.environ.get("SSMCGM_RESUME_CKPT")

CONTEXT_LEN = int(os.environ.get("SSMCGM_CONTEXT_LEN", "144"))  # 12h default, see module docstring
# SSMCGM_HORIZON: 12 (1h) default matches the paper's own forecasting
# objective (Sec 2, Eq. 1) and every existing checkpoint's trained decoder
# length. Override (e.g. 24 for 2h) trains a NEW decoder length from
# scratch -- the resulting checkpoint cannot be substituted for an existing
# HORIZON=12 checkpoint, and MIN_SEGMENT_LEN below grows accordingly, which
# may exclude some short segments an existing HORIZON=12 run included.
HORIZON = int(os.environ.get("SSMCGM_HORIZON", "12"))
BATCH_SIZE = int(os.environ.get("SSMCGM_BATCH_SIZE", "128"))  # see module docstring
NUM_WORKERS = 2  # real dataloader-bottleneck fix, see ssmcgm_batchsize_sweep_v2*.py
MAX_EPOCHS = int(os.environ.get("SSMCGM_MAX_EPOCHS", "30"))
EARLY_STOP_PATIENCE = 15
LR = float(os.environ.get("SSMCGM_LR", "1e-3"))

mb_cfg = dict(SSM_CGM.param_48["mamba_block"])
mt_cfg = SSM_CGM.param_48["mamba_tft_init"]
attention_heads = (mb_cfg["expand"] * mb_cfg["d_model"]) // mb_cfg["headdim"]

print(f"12h-context real training run: context={CONTEXT_LEN} horizon={HORIZON} batch_size={BATCH_SIZE} "
      f"d_model={mb_cfg['d_model']} mamba_depth={mt_cfg['mamba_depth']} num_workers={NUM_WORKERS} "
      f"NO_COVARIATES={NO_COVARIATES}",
      file=sys.stderr)

os.makedirs(OUTPUT_DIR, exist_ok=True)


MIN_SEGMENT_LEN = CONTEXT_LEN + HORIZON


def load_and_type(path):
    df = pd.read_csv(path)
    df["participant_id"] = df["participant_id"].astype(str)
    # Pre-filter segments too short to ever form a valid (context+horizon)
    # window, BEFORE TimeSeriesDataSet construction -- matching every
    # working MambaTFT script in this project (ssmcgm_batchsize_sweep_v2*.py
    # both explicitly sample only qualifying segments first). Passing every
    # segment (including some as short as 2 rows, 56 of which have zero-
    # variance targets from repeated identical readings) let
    # TimeSeriesDataSet's own min_encoder_length filtering silently exclude
    # them from actual windows AFTER GroupNormalizer had already fit a
    # degenerate (zero-scale) target normalizer on them -- the confirmed
    # difference between this script's crash (full data) and
    # ssmcgm_debug_val_bug.py's success (small sample, pre-filtered the
    # same way this now does).
    seg_lens = df.groupby("series_id").size()
    qualifying = seg_lens[seg_lens >= MIN_SEGMENT_LEN].index
    n_before = df["series_id"].nunique()
    df = df[df["series_id"].isin(qualifying)].copy()
    print(f"  {path}: kept {len(qualifying)}/{n_before} segments >= {MIN_SEGMENT_LEN} rows", file=sys.stderr)
    return df


print("loading real train/val data...", file=sys.stderr)
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
    # series_id/participant_id: val/test are ENTIRELY DIFFERENT participants
    # from train (this project's own deliberate cross-model comparability
    # choice, not SSM-CGM's own paper's per-participant time-based holdout,
    # where every test participant was also seen in train) -- so their IDs
    # are, by construction, categories training's encoder has never seen.
    # add_nan=True maps any unseen ID to an explicit "unknown" bucket at
    # inference, rather than crashing -- the honest behavior for a model
    # whose participant_id embedding (SSM-CGM's own #1 static feature,
    # arXiv:2510.04386 Table 5) has nothing trained for a genuinely new
    # person, matching how the other three models are evaluated on this
    # same held-out population. Caught directly via ssmcgm_train_real.py's
    # smoke test (KeyError: Unknown category '1028_0') before this was
    # fixed, not assumed in advance.
    categorical_encoders={
        "series_id": NaNLabelEncoder(add_nan=True),
        "participant_id": NaNLabelEncoder(add_nan=True),
    },
    add_relative_time_idx=True,
    add_target_scales=True,
    add_encoder_length=True,
)
# Shares training's fitted encoders/normalizers -- a val set built as its own
# independent TimeSeriesDataSet would fit its OWN categorical encoders/scalers,
# silently invalidating any comparison to the training-fitted model.
validation = TimeSeriesDataSet.from_dataset(training, val_df, stop_randomization=True)

print(f"TimeSeriesDataSet built: {len(training)} train samples, {len(validation)} val samples", file=sys.stderr)

train_dataloader = training.to_dataloader(
    train=True, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, persistent_workers=True,
)
val_dataloader = validation.to_dataloader(
    train=False, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, persistent_workers=True,
)

tft = SSM_CGM.MambaTFT.from_dataset(
    training,
    learning_rate=LR,
    hidden_size=mb_cfg["d_model"],
    attention_head_size=attention_heads,
    dropout=mt_cfg["mamba_dropout"],
    loss=QuantileLoss(quantiles=[0.1, 0.5, 0.9]),
    log_interval=0,
    log_val_interval=0,
    reduce_on_plateau_patience=4,
    mamba_depth=mt_cfg["mamba_depth"],
    mamba_dropout=mt_cfg["mamba_dropout"],
    mamba_checkpoint=False,
    mamba_block_config=mb_cfg,
)
n_params = sum(p.numel() for p in tft.parameters())
print(f"MambaTFT built: {n_params:,} params", file=sys.stderr)

logger = CSVLogger(OUTPUT_DIR, name="logs")
checkpoint_last = ModelCheckpoint(
    dirpath=OUTPUT_DIR, filename="last", save_last=True, every_n_epochs=1,
)
checkpoint_best = ModelCheckpoint(
    dirpath=OUTPUT_DIR, filename="best-{epoch}-{val_loss:.4f}", monitor="val_loss", mode="min", save_top_k=3,
)
early_stop = EarlyStopping(monitor="val_loss", mode="min", patience=EARLY_STOP_PATIENCE)

# SSMCGM_SMOKE=1: a few real train steps + real val batches, exactly the
# max_steps mechanism every previously-working MambaTFT run in this project
# uses (ssmcgm_real_model_smoketest.py, ssmcgm_batchsize_sweep_v2*.py) --
# NOT pl.Trainer's fast_dev_run=True, confirmed directly to break backward()
# on this model ("element 0 of tensors does not require grad and does not
# have a grad_fn", job 41375483) despite being a standard Lightning flag;
# a real, project-specific incompatibility, not a data issue.
SMOKE = os.environ.get("SSMCGM_SMOKE", "0") == "1"
# NUM_GPUS>1: DDP, matching the paper's own training setup (arXiv:2510.04386
# Appendix A.9: "4xA100 40GB (DDP)") -- not a shortcut introduced for this
# project, it's how the original was actually trained, just scaled to
# however many GPUs this job requests. "ddp" strategy only when >1 device;
# passing strategy="ddp" with devices=1 is harmless but pointless overhead.
NUM_GPUS = int(os.environ.get("SSMCGM_NUM_GPUS", "1"))
# Clamp to what's actually visible: job 41480351 requested -G 2 (SSMCGM_NUM_GPUS=2)
# and ran correctly for ~2h33m (through all of epoch 0 + part of epoch 1), then
# was requeued by Slurm (Reason=BeginTime) and on restart torch.cuda saw only 1
# device despite the same -G 2 allocation, crashing with "You requested gpu:
# [0, 1] But your machine only has: [0]" before any training happened this
# time. A prior occurrence of the identical error was a one-off that resolved
# itself on resubmission; this one didn't, so degrade gracefully instead of
# trusting the requested count matches what's visible.
_visible_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
if NUM_GPUS > _visible_gpus:
    print(f"WARNING: requested {NUM_GPUS} GPUs but only {_visible_gpus} visible -- "
          f"falling back to {_visible_gpus}", file=sys.stderr)
    NUM_GPUS = max(_visible_gpus, 1)
# bf16 mixed precision was tried and reverted: confirmed directly
# (RuntimeError: element 0 of tensors does not require grad and does not
# have a grad_fn, jobs 41376724/41377108) that MambaTFT's custom CUDA kernel
# (mamba_chunk_scan_combined) is not autocast-safe. Training stays full
# 32-bit precision; only DDP (NUM_GPUS>1) is used for speedup.
#
# The "element 0 of tensors does not require grad" crash (reproduced across
# jobs 41372284 through 41378606, every combination of fast_dev_run/bf16/DDP/
# num_sanity_val_steps tried) turned out to be none of those -- root cause
# was 56 segments in the raw train file as short as 2 rows (some with
# zero-variance targets from repeated identical readings), never filtered
# out before TimeSeriesDataSet construction the way every other script in
# this project does. TimeSeriesDataSet's own min_encoder_length filtering
# silently excludes them from actual windows, but GroupNormalizer had
# already fit a degenerate target scale on them first. Confirmed via
# ssmcgm_debug_val_bug.py (same encoder/validation mechanism, pre-filtered
# segments, ran cleanly) vs. this script's own crash before the
# load_and_type() pre-filter above was added. num_sanity_val_steps=0 is
# still set below since it's harmless and matches the other working
# scripts, but it was never the actual fix.
#
# limit_val_batches is NOT set to 0 here (unlike the sweep scripts, which
# had no real validation at all) -- this script needs genuine validation
# for early stopping to mean anything.
trainer = pl.Trainer(
    max_epochs=1 if SMOKE else MAX_EPOCHS,
    limit_train_batches=5 if SMOKE else 1.0,
    limit_val_batches=2 if SMOKE else 1.0,
    num_sanity_val_steps=0,
    accelerator="gpu" if torch.cuda.is_available() else "cpu",
    devices=NUM_GPUS,
    strategy="ddp" if NUM_GPUS > 1 else "auto",
    precision="32-true",
    logger=logger,
    callbacks=[checkpoint_last, checkpoint_best, early_stop],
    gradient_clip_val=1.0,
    enable_progress_bar=False,
)

print(f"starting training (resume={RESUME_CKPT})...", file=sys.stderr)
trainer.fit(tft, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader, ckpt_path=RESUME_CKPT)

print(f"best model: {checkpoint_best.best_model_path} (val_loss={checkpoint_best.best_model_score})", file=sys.stderr)
print("SSMCGM_TRAIN_REAL_OK", file=sys.stderr)
