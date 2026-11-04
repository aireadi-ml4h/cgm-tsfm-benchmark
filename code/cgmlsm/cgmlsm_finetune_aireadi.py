"""Fine-tunes the GlucoFM-Bench-pretrained CgmLsmLMHeadModel checkpoint
(code/cgmlsm/cgmlsm_pretrain_glucofmbench.py) on AI-READI's own
cgmlsm_{train,val}.csv (code/model_input/build_cgmlsm_input.py) -- the
adapt-to-target-population step the AI-READI zero-shot check
(code/cgmlsm/cgmlsm_eval.py against the raw GlucoFM-Bench checkpoint) deliberately
did not include.

Full fine-tuning, not LoRA, by explicit choice (discussed with the user):
LoRA's value is avoiding the compute/memory cost of updating a large model's
full parameter set and/or keeping one frozen base model with several small
swappable adapters. Neither applies here -- this model is ~4.9M parameters,
and full training from scratch on data this size already takes ~100s on 1
GPU (code/cgmlsm/cgmlsm_train.py, job 41046607), so there is no cost problem for
LoRA to solve. More importantly, the zero-shot check just found a real
domain-shift gap (GlucoFM-Bench zero-shot MAE 20.2 vs. the AI-READI
from-scratch model's 13.0 at 30min) -- the model needs to adapt
substantially to AI-READI's population, and LoRA's whole design constrains
updates to a low-rank subspace specifically to *limit* how much a model
changes, which works against that goal. Full fine-tuning lets every
parameter move as needed.

Lower learning rate (5e-5 vs. the from-scratch run's 3e-4) and fewer epochs
(10 vs. 20) than code/cgmlsm/cgmlsm_train.py's from-scratch run -- standard
fine-tuning practice, not re-derived from scratch: the model already has
learned general glucose-dynamics structure from GlucoFM-Bench, so this is
adaptation, not learning from nothing, and a smaller/slower update reduces
the risk of catastrophically overwriting that with only ~3.3M AI-READI
training tokens. eval_dataset is AI-READI's own held-out val split
(code/cgmlsm/cgmlsm_train.py's CgmLsmTokenDataset on cgmlsm_val.csv), watched to
catch overfitting on this comparatively small target population.
"""
import os
import sys

import torch
from transformers import Trainer, TrainingArguments, default_data_collator

REPO_ROOT = "/path/to/aireadi-collab"
sys.path.append(os.path.dirname(__file__))
from cgmlsm_train import CgmLsmTokenDataset  # noqa: E402
from nn.cgmlsm.modeling_cgmlsm import CgmLsmLMHeadModel  # noqa: E402

CGMLSM_INPUT_DIR = os.path.join(REPO_ROOT, "model_input", "cgmlsm")
PRETRAINED_CHECKPOINT_DIR = os.environ.get(
    "CGMLSM_PRETRAINED_CHECKPOINT_DIR",
    "/path/to/work/aireadi_cgmlsm/pretrain_glucofmbench_v2/final",
)
OUTPUT_DIR = os.environ.get(
    "CGMLSM_FINETUNE_OUTPUT_DIR_OVERRIDE",
    os.path.join(os.environ["AIREADI_SCRATCH"], "aireadi_cgmlsm", "finetune_glucofmbench_to_aireadi_v1"),
)

MAX_STEPS = int(os.environ.get("CGMLSM_MAX_STEPS", "-1"))
NUM_EPOCHS = float(os.environ.get("CGMLSM_NUM_EPOCHS", "10"))
LEARNING_RATE = float(os.environ.get("CGMLSM_LEARNING_RATE", "5e-5"))
# Overridable so a recipe can be checked for SEED SENSITIVITY. This matters at
# the high learning rates sweeps 3-5 explored: lr=2e-2 destroys the model
# outright and lr=1e-2 starts degenerating with one extra epoch, so the chosen
# recipe sits next to a cliff and a single-seed result there is not evidence of
# a stable optimum.
SEED = int(os.environ.get("CGMLSM_SEED", "42"))


def run():
    print("cuda available:", torch.cuda.is_available())
    torch.manual_seed(SEED)

    train_ds = CgmLsmTokenDataset(os.path.join(CGMLSM_INPUT_DIR, "cgmlsm_train.csv"))
    val_ds = CgmLsmTokenDataset(os.path.join(CGMLSM_INPUT_DIR, "cgmlsm_val.csv"))
    print(f"train chunks: {len(train_ds)}, val chunks: {len(val_ds)}")

    print("loading pretrained checkpoint from", PRETRAINED_CHECKPOINT_DIR)
    model = CgmLsmLMHeadModel.from_pretrained(PRETRAINED_CHECKPOINT_DIR)
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
        learning_rate=LEARNING_RATE,
        weight_decay=0.01,
        warmup_steps=100,
        max_grad_norm=1.0,
        logging_steps=50,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=5,
        report_to=[],
        seed=SEED,
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
    print("CGMLSM_FINETUNE_AIREADI_OK")


if __name__ == "__main__":
    run()
