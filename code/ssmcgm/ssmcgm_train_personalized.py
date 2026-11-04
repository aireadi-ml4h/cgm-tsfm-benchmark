"""SSM-CGM, paper-style personalization: fresh training run on the train
split PLUS each eval-set-E test participant's own pre-anchor history
(build_ssmcgm_personalization_train_input.py ->
ssmcgm_train_plus_testhistory.csv), so every test participant is partially
seen in training the same way SSM-CGM's own paper's per-participant
time-based holdout works -- not this project's own disjoint-participant
zero-shot condition (that's ssmcgm_train_real.py, already run, best
checkpoint best-epoch=0-val_loss=12.3597.ckpt).

FRESH TRAINING, NOT A WARM START FROM THAT CHECKPOINT -- see
build_ssmcgm_personalization_train_input.py's module docstring for why:
MambaTFT's participant_id/series_id embeddings are sized to whatever
vocabulary TimeSeriesDataSet.from_dataset() sees, and the augmented data
here contains genuinely new (test) participant IDs the old checkpoint's
embedding table has no rows for. A fresh `.from_dataset()` call sizes the
model correctly from the start with no embedding-surgery needed, and is
MORE faithful to their own methodology besides (they trained fresh under
their own split too).

MAX_EPOCHS/EARLY_STOP_PATIENCE lowered from ssmcgm_train_real.py's 30/15,
NOT re-derived from anything specific to this run -- ssmcgm_train_real.py's
own real result already showed this architecture/data combination reaches
its best generalization almost immediately (epoch 0 of 15 actually
completed) and overfits from there. Capping lower here is a direct,
disclosed response to that observed behavior, meant to save wall-clock
against the Sept 10 deadline, not a claim that this specific (different)
training set will behave identically.

Otherwise identical to ssmcgm_train_real.py: same architecture
(SSM_CGM.param_48), same context/horizon (144/12), same batch_size=128 +
num_workers=2 (the confirmed real dataloader-bottleneck fix, not a true
compute limit), same NUM_GPUS visible-device clamp, same checkpointing/
resume mechanism (proven robust across three real Slurm requeues on the
other run).

VAL_CSV is the ORIGINAL, unmodified ssmcgm_val.csv -- validation stays the
existing disjoint set so "which epoch is best" is judged by the same
criterion as before, uncontaminated by this experiment's own test
population (which never appears in validation, only in this run's training
data and, post-anchor, in the eventual test-time evaluation).
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
)

OUTPUT_DIR = os.environ.get("SSMCGM_PERS_OUTPUT_DIR", f"{REPO_ROOT}/ssmcgm_train_results/personalized_paperstyle")
TRAIN_CSV = f"{REPO_ROOT}/model_input/ssmcgm/ssmcgm_train_plus_testhistory.csv"
VAL_CSV = f"{REPO_ROOT}/model_input/ssmcgm/ssmcgm_val.csv"
RESUME_CKPT = os.environ.get("SSMCGM_PERS_RESUME_CKPT")

CONTEXT_LEN = 144
HORIZON = 12  # 1h, matches ssmcgm_train_real.py and the paper's own objective
BATCH_SIZE = int(os.environ.get("SSMCGM_PERS_BATCH_SIZE", "128"))
NUM_WORKERS = 2
MAX_EPOCHS = int(os.environ.get("SSMCGM_PERS_MAX_EPOCHS", "10"))
EARLY_STOP_PATIENCE = int(os.environ.get("SSMCGM_PERS_PATIENCE", "6"))
LR = 1e-3

mb_cfg = dict(SSM_CGM.param_48["mamba_block"])
mt_cfg = SSM_CGM.param_48["mamba_tft_init"]
attention_heads = (mb_cfg["expand"] * mb_cfg["d_model"]) // mb_cfg["headdim"]

print(f"paper-style personalization training run: context={CONTEXT_LEN} horizon={HORIZON} "
      f"batch_size={BATCH_SIZE} max_epochs={MAX_EPOCHS} patience={EARLY_STOP_PATIENCE}",
      file=sys.stderr)

os.makedirs(OUTPUT_DIR, exist_ok=True)

MIN_SEGMENT_LEN = CONTEXT_LEN + HORIZON


def load_and_type(path):
    df = pd.read_csv(path)
    df["participant_id"] = df["participant_id"].astype(str)
    seg_lens = df.groupby("series_id").size()
    qualifying = seg_lens[seg_lens >= MIN_SEGMENT_LEN].index
    n_before = df["series_id"].nunique()
    df = df[df["series_id"].isin(qualifying)].copy()
    print(f"  {path}: kept {len(qualifying)}/{n_before} segments >= {MIN_SEGMENT_LEN} rows", file=sys.stderr)
    return df


print("loading augmented train + real val data...", file=sys.stderr)
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
    # add_nan=True: val's participants are still entirely disjoint from this
    # run's (augmented) training population, same reasoning as
    # ssmcgm_train_real.py. Test participants are NO LONGER in this bucket
    # for this run specifically -- they now have real, dedicated embeddings
    # from being included in training -- but val's own population is
    # untouched and still needs the unknown-category fallback.
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

SMOKE = os.environ.get("SSMCGM_PERS_SMOKE", "0") == "1"
NUM_GPUS = int(os.environ.get("SSMCGM_PERS_NUM_GPUS", "1"))
_visible_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
if NUM_GPUS > _visible_gpus:
    print(f"WARNING: requested {NUM_GPUS} GPUs but only {_visible_gpus} visible -- "
          f"falling back to {_visible_gpus}", file=sys.stderr)
    NUM_GPUS = max(_visible_gpus, 1)

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

print(f"best model this process saw: {checkpoint_best.best_model_path} "
      f"(val_loss={checkpoint_best.best_model_score}) -- ONLY valid if this run was never "
      f"requeued/resumed; if it was, check all best-*.ckpt files on disk directly, same caveat "
      f"as ssmcgm_train_real.py's run.", file=sys.stderr)
print("SSMCGM_TRAIN_PERSONALIZED_OK", file=sys.stderr)
