"""Pretrains CgmLsmLMHeadModel from scratch on GlucoFM-Bench
(model_input/glucofmbench/glucofmbench_{train,held_out,temporal,internal}.csv,
code/model_input/build_glucofmbench_input.py) -- an 11-cohort, no-AI-READI-
participants corpus, distinct from code/cgmlsm/cgmlsm_train.py's AI-READI-only
from-scratch fit. This is the "pretraining" half of the pretrain-then-adapt
pattern the authors' own run_cgmlsm_pretrain.py follows (pretrain on a large
external corpus, apply to a smaller target population after); AI-READI
fine-tuning/eval from this checkpoint is a separate, later step, not done
by this script.

Reuses CgmLsmTokenDataset and build_model from cgmlsm_train.py unchanged --
same architecture (n_embd=256/n_layer=6/n_head=8, ~4.9M params;
n_positions=168), same reasoning: GlucoFM-Bench's ~2.2M total chunk-tokens
is the same order of magnitude as AI-READI's ~3.3M, not the "1.6M CGM
recordings"-scale corpus the authors' own GPT-2-base-sized config was built
for, so there is no more justification for a bigger model here than there
was for the AI-READI run.

Reports all three of the CGM-LSM paper's own held-in/held-out test
conditions (papers/2412.09727v3.pdf Methods), not just one: `held_out`
(subjects entirely excluded from training), `temporal` (held-in subjects'
own latest-in-time chunks), `internal` (held-in subjects' own random
held-out chunks) -- passed to Trainer as a dict eval_dataset, matching the
authors' own run_cgmlsm_pretrain.py eval_dict pattern (valid_t1d/valid_t2d/
testid_t1d/testid_t2d/testod_t1d/testod_t2d), which reports one loss per
named eval set per epoch rather than a single blended number.
"""
import os
import sys

import torch
from transformers import Trainer, TrainingArguments, default_data_collator

REPO_ROOT = "/path/to/aireadi-collab"
sys.path.append(os.path.dirname(__file__))
from cgmlsm_train import CgmLsmTokenDataset, build_model  # noqa: E402

GLUCOFMBENCH_INPUT_DIR = os.environ.get(
    "GLUCOFMBENCH_INPUT_DIR_OVERRIDE", os.path.join(REPO_ROOT, "model_input", "glucofmbench")
)
OUTPUT_DIR = os.environ.get(
    "CGMLSM_PRETRAIN_OUTPUT_DIR_OVERRIDE",
    os.path.join(os.environ["AIREADI_SCRATCH"], "aireadi_cgmlsm", "pretrain_glucofmbench_v2"),
)

MAX_STEPS = int(os.environ.get("CGMLSM_MAX_STEPS", "-1"))
NUM_EPOCHS = float(os.environ.get("CGMLSM_NUM_EPOCHS", "20"))


def run():
    print("cuda available:", torch.cuda.is_available())
    torch.manual_seed(42)

    train_ds = CgmLsmTokenDataset(os.path.join(GLUCOFMBENCH_INPUT_DIR, "glucofmbench_train.csv"))
    eval_datasets = {
        split: CgmLsmTokenDataset(os.path.join(GLUCOFMBENCH_INPUT_DIR, f"glucofmbench_{split}.csv"))
        for split in ("held_out", "temporal", "internal")
    }
    print(f"train chunks: {len(train_ds)}")
    for split, ds in eval_datasets.items():
        print(f"{split} chunks: {len(ds)}")

    model = build_model()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model params: {n_params:,}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        do_train=True,
        do_eval=True,
        num_train_epochs=NUM_EPOCHS,
        max_steps=MAX_STEPS,
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
        report_to=[],
        seed=42,
        dataloader_num_workers=4,
        fp16=torch.cuda.is_available(),
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=eval_datasets,
        data_collator=default_data_collator,
    )

    trainer.train()
    for split, ds in eval_datasets.items():
        metrics = trainer.evaluate(eval_dataset=ds, metric_key_prefix=split)
        print(f"final {split} eval metrics:", metrics)

    trainer.save_model(os.path.join(OUTPUT_DIR, "final"))
    print("saved final checkpoint to", os.path.join(OUTPUT_DIR, "final"))
    print("CGMLSM_PRETRAIN_GLUCOFMBENCH_OK")


if __name__ == "__main__":
    run()
