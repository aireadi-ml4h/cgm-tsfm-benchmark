"""Applies the trained model_input/meal_detection/checkpoint.pt CAUSALLY to
every real AI-READI participant's own cgm_glucose (build_participant_grid.
build_grid's glucose column, never fabricated -- see that module's
docstring), writing one per-participant CSV
(model_input/meal_detection/predmeal_prob/<person_id>.csv, columns
bin_start,predmeal_prob) mirroring analysis_dataframe's own per-participant-
file convention.

This is a SEPARATE precompute step, not something build_participant_grid.py
calls inline: build_grid() is invoked in a tight loop by every model-input
builder (build_ssmcgm_input.py, export_zeroshot_windows.py), most of which
run in the project's lightweight CPU-only venv (aireadi-collab/venv, plain
pandas/numpy, no torch) -- loading a PyTorch checkpoint on every call, or
adding a torch dependency to every consumer's environment, would be both
slow and invasive to scripts this project doesn't own alone (Stephanie's).
Precomputing once here and having build_participant_grid.py just read the
resulting plain CSV (added in a follow-up edit) keeps every downstream
consumer's dependency footprint unchanged.

CAUSALITY: for a real deployment forecast at time T, only glucose up to and
including T is ever available -- mealdetection.ipynb's own architecture
(model.py) is bidirectional (BiLSTM + "same"-padded CNN) and was trained
with full bidirectional context per window (train.py, legitimately, since
that is supervised learning on a fixed dataset). Applying that same
bidirectional model causally at inference time uses the standard trick: feed
it a window ENDING at T (bins [T-71, ..., T], nothing after T), and read off
only the model's own output at the window's LAST position. The model's
backward LSTM pass for that final position has nothing later in the fed
sequence to look at, so its receptive field is bounded by [T-71, T] even
though the same model, fed a window centered on some other T', would use
bidirectional context around T'. This is the same idea already used
elsewhere in this project to make an otherwise oracle-shaped input safe
(SSM-CGM's own wearable covariates were moved from known-real to
unknown-real for the same reason, build_ssmcgm_input.py's docstring).

A position gets NaN, not a fabricated value, if bins [T-71, T] are not all
real (a gap or segment start) -- consistent with this project's
never-fabricate-cgm_glucose principle, and with treating this precomputed
column the same as any other real-but-sometimes-missing physiological
channel (its own LOCF policy is applied downstream by each consumer, not
baked in here, matching build_ssmcgm_input.py's existing treatment of
heart_rate/oxygen_saturation/etc.).
"""
import glob
import os
import sys

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(__file__))
# Sibling modules live in adjacent topic directories under code/.
_CODE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _d in ("analysis", "baselines", "cgmlsm", "chronos2", "meal_detection", "model_input", "ssmcgm", "timesfm"):
    sys.path.insert(0, os.path.join(_CODE, _d))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "model_input"))
from model import MealDetector  # noqa: E402
from build_participant_grid import build_grid, load_participants_covariates  # noqa: E402

REPO_ROOT = "/path/to/aireadi-collab"
ANALYSIS_DIR = os.environ.get("ANALYSIS_DIR_OVERRIDE", f"{REPO_ROOT}/analysis_dataframe")
# MEAL_DETECTION_OUTPUT_DIR: which trained detector to load. Default
# unchanged, so the reported predmeal_prob channel and every existing caller
# behave exactly as before. This exists because load_model() is imported by
# evaluate_event_level.py / evaluate_event_level_curve.py, and until now it
# hardcoded the default checkpoint -- so pointing those scripts at an
# alternate directory silently re-scored the DEFAULT model while reading the
# alternate directory's metrics.json for its val split. That is exactly the
# failure it caused on 2026-09-07 (job 42433651), which returned the reported
# model's numbers to three decimal places and looked like a real result.
MODEL_DIR = os.environ.get("MEAL_DETECTION_OUTPUT_DIR", f"{REPO_ROOT}/model_input/meal_detection")
CHECKPOINT_PATH = f"{MODEL_DIR}/checkpoint.pt"
# NOT redirected by MODEL_DIR: this is the generated covariate channel every
# downstream model consumes, and regenerating it is a promotion decision, not
# a side effect of evaluating some other checkpoint. An alternate detector
# that should replace the channel must set this explicitly.
OUTPUT_DIR = os.environ.get(
    "MEAL_DETECTION_PREDMEAL_DIR", f"{REPO_ROOT}/model_input/meal_detection/predmeal_prob")
BATCH_SIZE = 4096


def load_model(device):
    ckpt = torch.load(CHECKPOINT_PATH, map_location=device)
    model = MealDetector().to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, ckpt["glucose_mean"], ckpt["glucose_std"], ckpt["seq_len"]


def causal_meal_probability(glucose, model, mean, std, seq_len, device):
    """glucose: 1D numpy array (real grid order, may contain NaN). Returns an
    array of the same length: NaN for positions without a full real causal
    window, else the model's sigmoid output at that window's last position."""
    n = len(glucose)
    out = np.full(n, np.nan, dtype=np.float64)
    if n < seq_len:
        return out

    valid_starts = []
    windows = []
    for t in range(seq_len - 1, n):
        window = glucose[t - seq_len + 1: t + 1]
        if np.isnan(window).any():
            continue
        valid_starts.append(t)
        windows.append(window)
    if not windows:
        return out

    X = (np.stack(windows) - mean) / std
    with torch.no_grad():
        for start in range(0, len(X), BATCH_SIZE):
            chunk = torch.from_numpy(X[start:start + BATCH_SIZE]).float().unsqueeze(-1).to(device)
            logits = model(chunk)  # (B, seq_len)
            probs = torch.sigmoid(logits[:, -1]).cpu().numpy()  # last position only, see module docstring
            for i, p in enumerate(probs):
                out[valid_starts[start + i]] = p
    return out


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, mean, std, seq_len = load_model(device)
    print(f"model loaded (device={device}, glucose_mean={mean:.2f}, glucose_std={std:.2f}, seq_len={seq_len})",
          file=sys.stderr)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    covs = load_participants_covariates()
    paths = sorted(glob.glob(f"{ANALYSIS_DIR}/*.csv"))
    pids = [os.path.basename(p).replace(".csv", "") for p in paths]

    n_done, n_skipped = 0, 0
    for i, pid in enumerate(pids):
        out_path = f"{OUTPUT_DIR}/{pid}.csv"
        if os.path.exists(out_path):
            n_done += 1
            continue
        df = build_grid(pid, covs)
        if df is None or df.empty:
            n_skipped += 1
            continue
        probs = causal_meal_probability(df["cgm_glucose"].to_numpy(dtype=float), model, mean, std, seq_len, device)
        pd.DataFrame({"bin_start": df["bin_start"], "predmeal_prob": probs}).to_csv(out_path, index=False)
        n_done += 1
        if (i + 1) % 200 == 0:
            print(f"  processed {i + 1}/{len(pids)}", file=sys.stderr)

    print(f"DONE: {n_done} participants written, {n_skipped} skipped (no data)", file=sys.stderr)


if __name__ == "__main__":
    main()
