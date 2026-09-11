"""Hyperparameter sweep for the meal detector, on the real (corrected)
trapezoidal-label task -- train.py's architecture/training hyperparameters
(c_out, h, dropout, lr; batch_size held fixed at 64 to bound sweep size) are
overridable via env vars specifically to support this. Selection criterion:
best validation AUPRC (same metric train.py already tracks for early
stopping/checkpointing) -- never selected using AI-READI or the eval-only
event-level Recall/Precision numbers, which would be circular (tuning on
the thing you're trying to report cleanly).

Each config trains as its own subprocess (cleanest way to reuse train.py
unchanged rather than refactoring its module-level constants into
parameters) writing to its own MEAL_DETECTION_OUTPUT_DIR under
model_input/meal_detection/sweep/<config_name>/, so no config's checkpoint
overwrites another's until a winner is explicitly copied to the shared
checkpoint.pt location.

Grid (16 configs): c_out in {16, 32}, hidden in {64, 128}, dropout in
{0.2, 0.4}, lr in {1e-4, 3e-4}. c_out=16/hidden=64/dropout=0.4/lr=1e-4 is
the paper's own architecture (arXiv:2510.04386 Appendix A.2's "Training
Setup"); the rest of the grid explores around it, not a replacement for it.
"""
import itertools
import json
import os
import subprocess
import sys

REPO_ROOT = "/path/to/aireadi-collab"
SWEEP_DIR = f"{REPO_ROOT}/model_input/meal_detection/sweep"
TRAIN_SCRIPT = f"{REPO_ROOT}/code/meal_detection/train.py"

C_OUT_GRID = [16, 32]
HIDDEN_GRID = [64, 128]
DROPOUT_GRID = [0.2, 0.4]
LR_GRID = [1e-4, 3e-4]


def run_config(c_out, hidden, dropout, lr):
    name = f"c{c_out}_h{hidden}_d{dropout}_lr{lr}"
    out_dir = f"{SWEEP_DIR}/{name}"
    os.makedirs(out_dir, exist_ok=True)
    env = os.environ.copy()
    env.update({
        "MEAL_DETECTION_OUTPUT_DIR": out_dir,
        "MEAL_DETECTION_C_OUT": str(c_out),
        "MEAL_DETECTION_HIDDEN": str(hidden),
        "MEAL_DETECTION_DROPOUT": str(dropout),
        "MEAL_DETECTION_LR": str(lr),
        "MEAL_DETECTION_EPOCHS": "40",
        "MEAL_DETECTION_PATIENCE": "6",
    })
    print(f"=== running {name} ===", file=sys.stderr)
    result = subprocess.run([sys.executable, TRAIN_SCRIPT], env=env, capture_output=True, text=True)
    print(result.stderr[-2000:], file=sys.stderr)  # tail of training log for this config
    if result.returncode != 0:
        print(f"  {name}: FAILED (exit {result.returncode})", file=sys.stderr)
        return name, None
    with open(f"{out_dir}/metrics.json") as f:
        metrics = json.load(f)
    return name, metrics["best_val_auprc"]


def main():
    results = []
    for c_out, hidden, dropout, lr in itertools.product(C_OUT_GRID, HIDDEN_GRID, DROPOUT_GRID, LR_GRID):
        name, best_val_auprc = run_config(c_out, hidden, dropout, lr)
        results.append((name, best_val_auprc, c_out, hidden, dropout, lr))
        print(f"{name}: best_val_auprc={best_val_auprc}", file=sys.stderr)

    valid = [r for r in results if r[1] is not None]
    valid.sort(key=lambda r: -r[1])
    print("\n" + "=" * 70, file=sys.stderr)
    print(f"{'config':40} {'best_val_auprc':>15}", file=sys.stderr)
    for name, auprc, *_ in valid:
        print(f"{name:40} {auprc:>15.4f}", file=sys.stderr)

    if valid:
        winner_name, winner_auprc, c_out, hidden, dropout, lr = valid[0]
        print(f"\nWINNER: {winner_name} (val_auprc={winner_auprc:.4f}) "
              f"c_out={c_out} hidden={hidden} dropout={dropout} lr={lr}", file=sys.stderr)
        print(f"checkpoint at: {SWEEP_DIR}/{winner_name}/checkpoint.pt", file=sys.stderr)

    print("\nSWEEP_TRAIN_OK", file=sys.stderr)


if __name__ == "__main__":
    main()
