"""Builds a participant-stratified sliding-window meal-detection dataset from
the real CGMacros dataset (PhysioNet, DOI 10.13026/3z8q-x658, CC BY-NC-SA
4.0, freely downloadable via `aws s3 sync --no-sign-request
s3://physionet-open/cgmacros/1.0.0/ ...` -- confirmed open access, no
credentialing) -- SSM-CGM's own paper (arXiv:2510.04386) trains its meal
detector on this same dataset.

UPDATED 2026-08-28: the paper's own Appendix A.2 ("Meal Detection") gives
the real labeling formula and windowing, previously unavailable (the repo's
MealDetection/mealdetection.ipynb only loads pre-built .npy arrays, whose
generation script is absent). This replaces this module's earlier, disclosed
GUESS (a flat binary window from meal-time to +120min) with their actual
method:

1. Windowing: "5-min CGM sampled into 6-h windows... L=72 samples" --
   CONFIRMS this module's 5-minute-bin, 72-bin-window choice was exactly
   right already (it was reasoned as a deviation from an apparent 1-minute
   resolution guessed from the notebook's ambiguous "15-60 min" phrasing;
   the paper text shows there was no deviation at all -- 5-min/72-bin was
   their real setup).

2. Label: a TRAPEZOIDAL soft target per meal timestamp tau_k, not a flat
   window -- w(t; tau_k, P, R) = 0 for |t-tau_k| >= P+R, 1 for |t-tau_k| <= P
   (bins), linearly ramping down between. P=2, R=2 in their own 5-min-bin
   units (their stated ~10min plateau / ~10min ramp), summed over every
   overlapping meal and clipped to 1 for BCE. Implemented exactly as
   trapezoidal_label() below, replacing the earlier flat +120min window.

3. pos_weight: their fixed alpha=5 (not empirically recomputed) -- with this
   soft trapezoidal target, "empirical positive rate" isn't a clean concept
   the way it is for a flat binary window, so their own fixed value is used
   directly rather than approximated.

Device: Dexcom GL is used, not Libre GL, despite Libre having 100% coverage
in the sample checked vs. Dexcom's 96.8% -- for consistency with the CGM
device family AI-READI/GlyLLM/SSM-CGM's own forecasting task actually uses,
not because of any completeness advantage. (Not stated in the paper text,
which doesn't name a specific sensor for this step.)

Windowing: sliding windows of SEQ_LEN=72 bins, stride=6 (30min) across each
participant's own resampled series, kept only if every bin in the window has
a real (non-NaN) Dexcom reading -- never fabricating the model's own input,
same principle as this project's cgm_glucose-never-filled rule elsewhere
(build_participant_grid.py). CORRECTED 2026-09-07: this docstring previously said stride "is not stated
for their training windows specifically, so stride=6 here is this project's
own choice ... rather than to match an unstated detail." That is FALSE.
Appendix A.2's "Training Setup" line states it directly -- "Sampling 5-minute
CGM; window length L=72; stride s=1; CNN channels cout=16; ... batch size 16;
10 epochs" -- so stride=1 applies to training, not only to the inference-time
ratio-voting reconstruction. stride=6 is therefore a DEVIATION FROM A STATED
SETTING, made to reduce training-set redundancy/overlap, and it was never
compared against stride=1. Same for batch size (theirs 16, ours 64, excluded
from the sweep only to bound its cost) and epochs (theirs a fixed 10, ours
<=40 with early stopping on val AUPRC). Of the four deviations only the
learning rate was ever measured, and that measurement is confounded: a higher
lr is the expected compensation for having ~24x fewer optimizer steps per
epoch. MEAL_DETECTION_STRIDE makes the faithful arm runnable. Every position in a kept window gets its own label (matching the
original architecture's per-timestep output head), not just the window's
last position -- TRAINING labels may legitimately use bidirectional context
(this is supervised learning on a fixed dataset, not a live forecast); it is
only inference-time application to AI-READI (generate_meal_probability.py)
that is constrained to be causal.

Participant-level split (not window-level): 36 train / 9 val, fixed seed.
The paper's own split is "subject-wise... stratified by diabetes status
(healthy, prediabetes, diabetes)" -- CGMacros' bio.csv is used here (added
2026-08-28) to replicate the stratification, not just the subject-wise
axis.
"""
import glob
import os

import numpy as np
import pandas as pd

# Defaults to where slurm/cgmacros_download.sbatch extracts CGMacros 1.0.0.
CGMACROS_DIR = os.environ.get(
    "CGMACROS_DIR",
    os.path.join(os.environ["AIREADI_SCRATCH"], "cgmacros", "1.0.0", "extracted", "CGMacros"),
)
BIO_CSV_PATH = os.environ.get(
    "CGMACROS_BIO_CSV", os.path.join(CGMACROS_DIR, "bio.csv")
)
BIN_MINUTES = 5
SEQ_LEN = 72  # bin count, matches SSM-CGM paper's Appendix A.2 exactly (confirmed, not guessed)
# CORRECTED 2026-09-07: the source's training stride IS stated. Appendix A.2's
# "Training Setup" line reads "Sampling 5-minute CGM; window length L=72;
# stride s=1; ... batch size 16; 10 epochs", so stride=1 is a published
# training setting, not (as this module previously claimed) an unstated detail
# left to our discretion. The default stays 6 so every already-reported number
# reproduces unchanged, but it is now overridable, because the deviation needs
# to be measurable rather than merely disclosed: stride 6 with batch 64 gives
# ~24x fewer optimizer steps per epoch than stride 1 with batch 16, so the
# faithful arm and this one are not close to the same training budget.
# MEAL_DETECTION_STRIDE=1 reproduces the source protocol -- see
# slurm/meal_detection_faithful_arm.sbatch.
STRIDE = int(os.environ.get("MEAL_DETECTION_STRIDE", "6"))  # 6 = 30 min between window starts
# Trapezoidal label smoothing (arXiv:2510.04386 Appendix A.2, "Trapezoidal
# label smoothing"): plateau half-width P and ramp length R, both in 5-min
# bin units, matching their stated "P=2 (plateau length ~=10min), R=2 (ramp
# length ~=10min)" exactly.
TRAPEZOID_P_BINS = 2
TRAPEZOID_R_BINS = 2
VAL_FRACTION = 0.2
SPLIT_SEED = 42


def load_bio_diabetes_status():
    """Diabetes status per subject, derived from bio.csv's own 'A1c PDL
    (Lab)' column using standard ADA HbA1c thresholds (<5.7% normoglycemia,
    5.7-6.4% prediabetes, >=6.5% diabetes) -- bio.csv has no direct
    diabetes-status column, and the paper's own text doesn't say how they
    derived it either, so this is this project's own defensible derivation,
    not a recovered detail. Used only for participant-level split
    stratification (build_dataset), matching the paper's stated "subject-
    wise... stratified by diabetes status" split axis."""
    bio = pd.read_csv(BIO_CSV_PATH)
    a1c = pd.to_numeric(bio["A1c PDL (Lab)"], errors="coerce")
    status = pd.cut(a1c, bins=[-np.inf, 5.7, 6.5, np.inf], labels=["healthy", "prediabetes", "diabetes"], right=False)
    return dict(zip(bio["subject"].astype(str), status.astype(str)))


def trapezoidal_label(meal_times, index, p_bins=TRAPEZOID_P_BINS, r_bins=TRAPEZOID_R_BINS):
    """y_t = min(1, sum_k w(t; tau_k, P, R)) -- real formula, arXiv:2510.04386
    Appendix A.2. `index` is the resampled glucose series' own DatetimeIndex
    (5-min bins); meal_times are real timestamps from the Meal Type column.
    Distance is measured in BINS (via index.get_indexer), not minutes, since
    P/R are themselves defined in bin units in the paper."""
    n = len(index)
    total = np.zeros(n, dtype=float)
    for t in meal_times:
        center = index.get_indexer([t], method="nearest")[0]
        lo, hi = max(0, center - p_bins - r_bins), min(n, center + p_bins + r_bins + 1)
        dist = np.abs(np.arange(lo, hi) - center)
        w = np.where(dist <= p_bins, 1.0, np.clip(1.0 - (dist - p_bins) / r_bins, 0.0, 1.0))
        total[lo:hi] += w
    return np.minimum(1.0, total)


def load_participant_series(csv_path):
    df = pd.read_csv(csv_path)
    df["Timestamp"] = pd.to_datetime(df["Timestamp"])
    df = df.set_index("Timestamp").sort_index()

    glucose = df["Dexcom GL"].resample(f"{BIN_MINUTES}min").mean()
    meal_times = df.index[df["Meal Type"].notna()]
    labels = pd.Series(trapezoidal_label(meal_times, glucose.index), index=glucose.index)

    return glucose, labels


def build_windows(glucose, labels):
    """Returns (n_windows, SEQ_LEN) X and y arrays, dropping any window with
    a NaN Dexcom reading anywhere in it."""
    values = glucose.to_numpy()
    label_values = labels.to_numpy()
    n = len(values)
    xs, ys = [], []
    for start in range(0, n - SEQ_LEN + 1, STRIDE):
        window = values[start:start + SEQ_LEN]
        if np.isnan(window).any():
            continue
        xs.append(window)
        ys.append(label_values[start:start + SEQ_LEN])
    if not xs:
        return np.empty((0, SEQ_LEN)), np.empty((0, SEQ_LEN))
    return np.stack(xs), np.stack(ys)


def build_dataset():
    paths = sorted(glob.glob(f"{CGMACROS_DIR}/*/*.csv"))
    pids = [os.path.basename(p).replace(".csv", "") for p in paths]

    # Diabetes-stratified subject-wise split (arXiv:2510.04386 Appendix A.2:
    # "subject-wise splits (no leakage) stratified by diabetes status") --
    # split independently within each status group, not the whole cohort at
    # once, so e.g. the 14 diabetes-status subjects aren't accidentally all
    # placed in the same split by chance.
    status_by_subject = load_bio_diabetes_status()
    pids_by_status = {}
    for pid in pids:
        subject_num = pid.replace("CGMacros-", "").lstrip("0") or "0"
        status = status_by_subject.get(subject_num, "unknown")
        pids_by_status.setdefault(status, []).append(pid)

    rng = np.random.RandomState(SPLIT_SEED)
    val_pids, train_pids = set(), set()
    for status, group_pids in pids_by_status.items():
        shuffled = rng.permutation(group_pids)
        n_val = max(1, int(round(len(shuffled) * VAL_FRACTION))) if len(shuffled) > 1 else 0
        val_pids.update(shuffled[:n_val])
        train_pids.update(shuffled[n_val:])

    splits = {"train": {"X": [], "y": []}, "val": {"X": [], "y": []}}
    n_windows_by_pid = {}
    for path, pid in zip(paths, pids):
        glucose, labels = load_participant_series(path)
        X, y = build_windows(glucose, labels)
        n_windows_by_pid[pid] = len(X)
        if len(X) == 0:
            continue
        split = "val" if pid in val_pids else "train"
        splits[split]["X"].append(X)
        splits[split]["y"].append(y)

    out = {}
    for split in ("train", "val"):
        out[f"X_{split}"] = np.concatenate(splits[split]["X"], axis=0)
        out[f"y_{split}"] = np.concatenate(splits[split]["y"], axis=0)
    out["train_pids"] = sorted(train_pids)
    out["val_pids"] = sorted(val_pids)
    out["n_windows_by_pid"] = n_windows_by_pid
    return out


if __name__ == "__main__":
    ds = build_dataset()
    print(f"train: {ds['X_train'].shape}, positive rate {ds['y_train'].mean():.4f}")
    print(f"val:   {ds['X_val'].shape}, positive rate {ds['y_val'].mean():.4f}")
    print(f"train participants ({len(ds['train_pids'])}): {ds['train_pids']}")
    print(f"val participants ({len(ds['val_pids'])}): {ds['val_pids']}")
