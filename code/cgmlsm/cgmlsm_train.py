"""Trains CgmLsmLMHeadModel from scratch on this project's own
model_input/cgmlsm/cgmlsm_{train,val}.csv (code/model_input/build_cgmlsm_input.py),
adapting code/external/cgmlsm/1-CGMLSM/run_cgmlsm_pretrain.py's own Trainer/
CgmLsmLMHeadModel setup -- the authors' real pretraining script, not a toy --
to read this project's CSVs instead of their private WellDoc HF dataset.
Necessary because CGM-LSM ships no public checkpoint (report.md Sec 15.6):
this is a real from-scratch run, the same situation SSM-CGM is in.

Vocab: reuses code/model_input/cgmlsm_vocab.py's CGMValue mapping exactly.
HM5MinStep (the input CSVs' tod_token column) is NOT fed to the model here:
checked directly against modeling_cgmlsm.py's CgmLsmModel.forward(), the
only embeddings are wte (token) + wpe (learned position, default sequential
position_ids) -- run_cgmlsm_pretrain.py itself builds the HM5MinStep vocab
but never passes it to CgmLsmConfig or the model call, so it has no consumer
in the authors' own real training script either. This script matches that:
input_ids = cgm_token only, labels = input_ids (standard causal LM loss,
shifted internally by CgmLsmLMHeadModel.forward -- confirmed working in
code/cgmlsm_toy_pretrain_smoketest.py).

Architecture: deliberately NOT run_cgmlsm_pretrain.py's own defaults
verbatim. `CgmLsmConfig(vocab_size=...)` with every other default untouched
resolves to full GPT-2-base (n_positions=1024, n_embd=768, n_layer=12,
n_head=12, ~124M params) -- sized for their WellDoc pretraining set
("1.6M CGM recordings", Project_Brainstorm.pdf). This project's train split
is 22,783 chunks x 144 tokens = ~3.3M tokens total, roughly 3 orders of
magnitude smaller; keeping their config would guarantee severe overfitting
long before one epoch finishes. Scaled down to n_embd=256/n_layer=6/n_head=8
(~5M params) as a disclosed, reasoned starting point: larger than the pure
correctness-check config in cgmlsm_toy_pretrain_smoketest.py (128/4/4,
20 synthetic series, 10 steps) since this is a real run, far below the
authors' own WellDoc-scale config. n_positions is set to CONTEXT_BINS plus
the longest eval horizon (144 + 24 = 168), not their 1024 and not just 144:
training chunks are only ever 144 tokens long, but model.generate() at eval
time needs one learned position embedding per context+generated-token slot,
and the 2h eval horizon (24 bins) pushes that past 144 -- a first version of
this script used exactly 144 and hit
`indexSelectLargeIndex: srcIndex < srcSelectDimSize` (a CUDA assertion, not
a Python exception) partway through eval, since position ids 144-167 had no
row in the wpe table at all. Trained here even though no training chunk
actually reaches position >=144, so the embedding rows for 144-167 start
untrained (randomly initialized) and only ever get used at generation time
-- disclosed as a real limitation, not silently patched over: those late
positions' embeddings have no training signal behind them, unlike a
from-scratch run whose training data itself spanned the full context+horizon
length.
pad_token_id/bos_token_id are set to the vocab's own PAD index
(cgmlsm_vocab.CGM_PAD_IDX, following cgmlsm_toy_pretrain_smoketest.py's
precedent); eos_token_id is left None -- this vocab has no dedicated EOS
symbol, and no chunk needs one since every chunk is a fixed 144-token slice,
not a variable-length sequence ending in EOS.

Uses transformers.Trainer directly on a plain torch.utils.data.Dataset
(CgmLsmTokenDataset below), not the `datasets` library run_cgmlsm_pretrain.py
uses to load its own HF-format WellDoc dataset -- `datasets` is not
installed in venv-cgmlsm and Trainer works with any torch Dataset, so there
is no need to add it. Also does not reproduce the authors' fuller
compute_metrics (next_token_loss/perplexity/accuracy recomputed from raw
logits) -- Trainer's own eval_loss already tracks the causal LM loss the
model returns, which is enough for a first real run; not needed to add the
extra logits-memory overhead of a custom compute_metrics for this pass.
"""
import os
import sys

import numpy as np
import pandas as pd
import torch
from transformers import Trainer, TrainingArguments, default_data_collator

REPO_ROOT = "/path/to/aireadi-collab"
sys.path.append(os.path.join(REPO_ROOT, "code/external/cgmlsm/code/pipeline"))
# Sibling modules live in adjacent topic directories under code/.
_CODE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _d in ("analysis", "baselines", "cgmlsm", "chronos2", "meal_detection", "model_input", "ssmcgm", "timesfm"):
    sys.path.insert(0, os.path.join(_CODE, _d))
sys.path.append(os.path.join(REPO_ROOT, "code/model_input"))

from nn.cgmlsm.configuration_cgmlsm import CgmLsmConfig  # noqa: E402
from nn.cgmlsm.modeling_cgmlsm import CgmLsmLMHeadModel  # noqa: E402
from cgmlsm_vocab import CGM_IDX2TKN, CGM_TKN2IDX, CGM_PAD_IDX  # noqa: E402
from windows import CONTEXT_BINS, HORIZONS_BINS  # noqa: E402

CGMLSM_INPUT_DIR = os.path.join(REPO_ROOT, "model_input", "cgmlsm")
OUTPUT_DIR = os.path.join(os.environ["AIREADI_SCRATCH"], "aireadi_cgmlsm", "train_v1")

# Longest horizon actually evaluated (2h = 24 bins, cgmlsm_eval.py's
# EVAL_HORIZONS) -- see build_model()'s n_positions comment for why this
# must be added to CONTEXT_BINS, not just CONTEXT_BINS alone.
MAX_EVAL_HORIZON_BINS = max(v for k, v in HORIZONS_BINS.items() if k in ("30min", "1h", "2h"))

# Overridable for a quick smoke test against the real data before committing
# to a full multi-epoch job (mirrors this project's own smoke-test-before-
# full-run convention -- code/CHRONOS2_EVAL.md, ssmcgm_timing_dryrun.py).
MAX_STEPS = int(os.environ.get("CGMLSM_MAX_STEPS", "-1"))
NUM_EPOCHS = float(os.environ.get("CGMLSM_NUM_EPOCHS", "20"))


class CgmLsmTokenDataset(torch.utils.data.Dataset):
    """cgm_token strings -> CGMValue index tensors, reshaped assuming each
    file's rows are already grouped into contiguous CONTEXT_BINS-length
    chunks in chunk order (guaranteed by build_cgmlsm_input.py's own
    construction -- asserted here, not just assumed)."""

    def __init__(self, csv_path, context_bins=CONTEXT_BINS):
        df = pd.read_csv(csv_path, usecols=["chunk_id", "cgm_token"])
        if len(df) % context_bins != 0:
            raise ValueError(f"{csv_path}: {len(df)} rows not divisible by context_bins={context_bins}")
        n_chunks = len(df) // context_bins
        first_of_each_chunk = df["chunk_id"].to_numpy()[0::context_bins]
        if len(set(first_of_each_chunk.tolist())) != n_chunks:
            raise ValueError(f"{csv_path}: rows are not laid out as contiguous {context_bins}-row chunks")
        ids = df["cgm_token"].astype(str).map(CGM_TKN2IDX)
        if ids.isna().any():
            raise ValueError(f"{csv_path}: found cgm_token value(s) outside CGMValue vocab")
        self.input_ids = ids.to_numpy(dtype=np.int64).reshape(n_chunks, context_bins)

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        ids = torch.from_numpy(self.input_ids[idx].copy())
        return {"input_ids": ids, "labels": ids.clone()}


def build_model():
    config = CgmLsmConfig(
        vocab_size=len(CGM_IDX2TKN),
        n_positions=CONTEXT_BINS + MAX_EVAL_HORIZON_BINS,
        n_embd=256,
        n_layer=6,
        n_head=8,
        pad_token_id=CGM_PAD_IDX,
        bos_token_id=CGM_PAD_IDX,
        eos_token_id=None,
    )
    return CgmLsmLMHeadModel(config)


def run():
    print("cuda available:", torch.cuda.is_available())
    torch.manual_seed(42)

    train_ds = CgmLsmTokenDataset(os.path.join(CGMLSM_INPUT_DIR, "cgmlsm_train.csv"))
    val_ds = CgmLsmTokenDataset(os.path.join(CGMLSM_INPUT_DIR, "cgmlsm_val.csv"))
    print(f"train chunks: {len(train_ds)}, val chunks: {len(val_ds)}")

    model = build_model()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model params: {n_params:,}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        do_train=True,
        do_eval=True,
        num_train_epochs=NUM_EPOCHS,
        max_steps=MAX_STEPS,  # -1 means "use num_train_epochs instead", transformers' own convention
        per_device_train_batch_size=64,
        per_device_eval_batch_size=128,
        learning_rate=3e-4,
        weight_decay=0.01,
        warmup_steps=200,
        max_grad_norm=1.0,
        logging_steps=50,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=5,
        report_to=[],  # no wandb -- WANDB_PROJECT isn't configured for this run
        seed=42,
        dataloader_num_workers=4,
        fp16=torch.cuda.is_available(),
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=default_data_collator,
    )

    trainer.train()
    metrics = trainer.evaluate()
    print("final eval metrics:", metrics)

    trainer.save_model(os.path.join(OUTPUT_DIR, "final"))
    print("saved final checkpoint to", os.path.join(OUTPUT_DIR, "final"))
    print("CGMLSM_TRAIN_OK")


if __name__ == "__main__":
    run()
