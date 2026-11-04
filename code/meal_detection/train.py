"""Trains model.MealDetector on cgmacros_dataset.build_dataset()'s real
CGMacros windows, tracks val AUPRC (not just loss -- positives are a
minority class, see pos_weight below), and checkpoints the best epoch.

pos_weight is SSM-CGM's own fixed alpha=5 (arXiv:2510.04386 Appendix A.2),
not empirically estimated -- with the real trapezoidal soft target
(cgmacros_dataset.trapezoidal_label, continuous in [0,1]), "empirical
positive rate" isn't the clean binary-class-balance concept an empirical
pos_weight assumes.

Architecture/training hyperparameters (c_out, h, dropout, LR, BATCH_SIZE)
are overridable via env vars -- see sweep_train.py, which trains a small
grid of these and reports the best by val AUPRC, matching the "sweep more
params" instruction directly, not just re-running the paper's exact
defaults once.

Saves: checkpoint.pt (model state_dict + the training-set glucose mean/std
used to normalize input -- inference must reuse these exact values, not
recompute new ones on AI-READI's own data, or the model sees an input
distribution it was never trained on) and metrics.json (val AUPRC/loss per
epoch, for a paper-ready training curve).
"""
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import average_precision_score
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, os.path.dirname(__file__))
from cgmacros_dataset import build_dataset  # noqa: E402
from model import MealDetector  # noqa: E402

REPO_ROOT = os.environ["AIREADI_ROOT"]
OUTPUT_DIR = os.environ.get(
    "MEAL_DETECTION_OUTPUT_DIR",
    f"{REPO_ROOT}/model_input/meal_detection",
)
EPOCHS = int(os.environ.get("MEAL_DETECTION_EPOCHS", "40"))
PATIENCE = int(os.environ.get("MEAL_DETECTION_PATIENCE", "6"))
BATCH_SIZE = int(os.environ.get("MEAL_DETECTION_BATCH_SIZE", "64"))
LR = float(os.environ.get("MEAL_DETECTION_LR", "1e-4"))
C_OUT = int(os.environ.get("MEAL_DETECTION_C_OUT", "16"))
HIDDEN = int(os.environ.get("MEAL_DETECTION_HIDDEN", "64"))
DROPOUT = float(os.environ.get("MEAL_DETECTION_DROPOUT", "0.4"))
SEED = 42


class WindowDataset(Dataset):
    def __init__(self, X, y, mean, std):
        self.X = torch.from_numpy((X - mean) / std).float().unsqueeze(-1)
        self.y = torch.from_numpy(y).float()

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        return self.X[i], self.y[i]


def run_epoch(model, dl, device, criterion, optimizer=None, scheduler=None):
    is_train = optimizer is not None
    model.train() if is_train else model.eval()
    total_loss, n_samples = 0.0, 0
    all_probs, all_labels = [], []
    with torch.set_grad_enabled(is_train):
        for xb, yb in dl:
            xb, yb = xb.to(device), yb.to(device)
            if is_train:
                optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            if is_train:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()
            total_loss += loss.item() * xb.size(0)
            n_samples += xb.size(0)
            all_probs.append(torch.sigmoid(logits).detach().cpu().numpy().ravel())
            all_labels.append(yb.detach().cpu().numpy().ravel())
    # average_precision_score requires binary y_true; the real trapezoidal
    # target (cgmacros_dataset.trapezoidal_label) is continuous in [0,1], so
    # it's binarized at 0.5 ONLY for this reported metric -- the actual loss
    # just above (BCEWithLogits) still trains against the full soft target,
    # unchanged. 0.5 matches the plateau region (distance <= P bins, target
    # exactly 1.0) vs. the ramp's outer half, a reasonable "was this really
    # a meal moment" cutoff for monitoring, not used anywhere in training.
    binary_labels = (np.concatenate(all_labels) >= 0.5).astype(int)
    auprc = average_precision_score(binary_labels, np.concatenate(all_probs))
    return total_loss / n_samples, auprc


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("building CGMacros dataset...", file=sys.stderr)
    ds = build_dataset()
    mean, std = ds["X_train"].mean(), ds["X_train"].std()
    print(f"train windows: {ds['X_train'].shape}, val windows: {ds['X_val'].shape}", file=sys.stderr)
    print(f"train positive rate: {ds['y_train'].mean():.4f}, val positive rate: {ds['y_val'].mean():.4f}", file=sys.stderr)
    print(f"glucose mean={mean:.2f} std={std:.2f} (train set)", file=sys.stderr)

    # SSM-CGM's own fixed value (arXiv:2510.04386 Appendix A.2: "alpha=5
    # upweights positives"), not an empirical estimate: with the real
    # trapezoidal soft target (cgmacros_dataset.trapezoidal_label, values in
    # [0,1]), "empirical positive rate" isn't the clean binary-class-balance
    # concept an empirical pos_weight assumes -- their own fixed value is
    # used directly now that it's available, rather than approximated.
    pos_weight = 5.0
    print(f"pos_weight={pos_weight:.3f} (fixed, matching SSM-CGM's own alpha=5), "
          f"train mean target={ds['y_train'].mean():.4f}", file=sys.stderr)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_dl = DataLoader(WindowDataset(ds["X_train"], ds["y_train"], mean, std), batch_size=BATCH_SIZE, shuffle=True)
    val_dl = DataLoader(WindowDataset(ds["X_val"], ds["y_val"], mean, std), batch_size=BATCH_SIZE, shuffle=False)

    model = MealDetector(c_out=C_OUT, h=HIDDEN, dropout=DROPOUT).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight, device=device, dtype=torch.float32))
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=LR, steps_per_epoch=len(train_dl), epochs=EPOCHS)

    best_val_auprc, best_epoch, epochs_since_best = -1.0, -1, 0
    history = []
    job_start = time.time()
    for epoch in range(1, EPOCHS + 1):
        train_loss, train_auprc = run_epoch(model, train_dl, device, criterion, optimizer, scheduler)
        val_loss, val_auprc = run_epoch(model, val_dl, device, criterion)
        history.append({"epoch": epoch, "train_loss": train_loss, "train_auprc": train_auprc,
                         "val_loss": val_loss, "val_auprc": val_auprc})
        print(f"epoch {epoch}/{EPOCHS}: train_loss={train_loss:.4f} train_auprc={train_auprc:.4f} "
              f"val_loss={val_loss:.4f} val_auprc={val_auprc:.4f} "
              f"({time.time() - job_start:.1f}s elapsed)", file=sys.stderr)

        if val_auprc > best_val_auprc:
            best_val_auprc, best_epoch, epochs_since_best = val_auprc, epoch, 0
            torch.save({
                "model_state_dict": model.state_dict(),
                "glucose_mean": float(mean),
                "glucose_std": float(std),
                "seq_len": ds["X_train"].shape[1],
                "epoch": epoch,
                "val_auprc": val_auprc,
            }, f"{OUTPUT_DIR}/checkpoint.pt")
        else:
            epochs_since_best += 1
            if epochs_since_best >= PATIENCE:
                print(f"early stopping at epoch {epoch} (best was epoch {best_epoch}, val_auprc={best_val_auprc:.4f})",
                      file=sys.stderr)
                break

        with open(f"{OUTPUT_DIR}/metrics.json", "w") as f:
            json.dump({"history": history, "best_epoch": best_epoch, "best_val_auprc": best_val_auprc,
                       "train_pids": ds["train_pids"], "val_pids": ds["val_pids"]}, f, indent=2)

    print(f"DONE best_epoch={best_epoch} best_val_auprc={best_val_auprc:.4f}", file=sys.stderr)


if __name__ == "__main__":
    main()
