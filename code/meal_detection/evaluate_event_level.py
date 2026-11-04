"""Event-level Recall/Precision for the trained meal detector, computed
exactly as SSM-CGM's paper defines it (arXiv:2510.04386 Appendix A.2:
"Inference & reconstruction (thresholding + ratio voting)" and "Event-level
metrics") -- the only way to get a number directly comparable to their
reported Recall=80.13%, Precision=72.08%, since AUPRC (this project's own
training-time metric) isn't something they reported at all.

Method, matching their text exactly:
1. Slide a SEQ_LEN=72-bin window with STRIDE=1 across each validation
   participant's own real (non-NaN) glucose span.
2. Per-window thresholding at THETA=0.36: sigmoid(logits) > THETA -> binary
   prediction at every position inside that window.
3. Ratio voting at every global timestep t: ratio(t) = (# covering windows
   predicting 1 at t) / (# windows covering t); final flag = 1 if
   ratio(t) >= RHO=0.77.
4. Ground truth T and predictions P are each converted to sets of maximal
   1-run intervals; Recall_evt = fraction of true intervals overlapping
   >=1 predicted interval, Precision_evt = fraction of predicted intervals
   overlapping >=1 true interval.

Ground truth binarization: the real label (cgmacros_dataset.trapezoidal_
label) is continuous in [0,1]; thresholded at 0.5 here (the plateau region,
distance <= P bins from a real meal) to get discrete truth events -- the
paper's own text doesn't state how they binarized ground truth for this
step, so this is disclosed as this project's own choice, kept consistent
with train.py's AUPRC binarization threshold.

THETA/RHO: used at the paper's own reported values (0.36, 0.77) for a
direct, same-operating-point comparison, not re-swept on this project's own
split/model -- their values were chosen via their own validation sweep,
which may not be optimal for a differently-split, retrained model, but
re-sweeping would no longer be comparing the same thing.
"""
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
from cgmacros_dataset import CGMACROS_DIR, SEQ_LEN, load_participant_series  # noqa: E402
from generate_meal_probability import load_model  # noqa: E402

REPO_ROOT = os.environ["AIREADI_ROOT"]
# MEAL_DETECTION_OUTPUT_DIR: the same env var train.py already uses, so a
# checkpoint trained into an alternate directory can be evaluated without
# touching the reported one. Default unchanged, so every existing invocation
# behaves exactly as before. Needed for the source-protocol reproduction arm
# (slurm/meal_detection_faithful_arm.sbatch): without it this module would
# silently score the DEFAULT checkpoint and, worse, the curve variant would
# overwrite the reported event_level_pr_curve.csv with the other model's.
OUTPUT_DIR = os.environ.get("MEAL_DETECTION_OUTPUT_DIR", f"{REPO_ROOT}/model_input/meal_detection")
# Inference stride is 1 REGARDLESS of the training stride -- this is the
# paper's ratio-voting reconstruction step, which is stride 1 in both arms.
STRIDE = 1
THETA = 0.36
RHO = 0.77
TRUTH_THRESHOLD = 0.5


def sliding_window_ratio_vote(model, glucose, mean, std, device, batch_size=512):
    """Returns (pred_flags, covered) both length-n arrays: pred_flags[t] is
    the ratio-voted binary prediction (only meaningful where covered[t]),
    covered[t] is whether >=1 valid window includes position t."""
    n = len(glucose)
    vote_sum = np.zeros(n, dtype=float)
    vote_count = np.zeros(n, dtype=float)

    starts = [s for s in range(0, n - SEQ_LEN + 1, STRIDE) if not np.isnan(glucose[s:s + SEQ_LEN]).any()]
    if not starts:
        return np.zeros(n, dtype=bool), np.zeros(n, dtype=bool)

    for b in range(0, len(starts), batch_size):
        batch_starts = starts[b:b + batch_size]
        X = np.stack([glucose[s:s + SEQ_LEN] for s in batch_starts])
        X = (X - mean) / std
        with torch.no_grad():
            logits = model(torch.from_numpy(X).float().unsqueeze(-1).to(device))
            probs = torch.sigmoid(logits).cpu().numpy()
        window_preds = (probs > THETA).astype(float)
        for i, s in enumerate(batch_starts):
            vote_sum[s:s + SEQ_LEN] += window_preds[i]
            vote_count[s:s + SEQ_LEN] += 1.0

    covered = vote_count > 0
    ratio = np.divide(vote_sum, vote_count, out=np.zeros_like(vote_sum), where=covered)
    pred_flags = covered & (ratio >= RHO)
    return pred_flags, covered


def to_intervals(binary_series, covered=None):
    """Maximal runs of True in binary_series, restricted to `covered`
    positions if given (uncovered positions never start/extend an interval,
    they just end whatever run preceded them)."""
    if covered is not None:
        binary_series = binary_series & covered
    intervals = []
    start = None
    for i, v in enumerate(binary_series):
        if v and start is None:
            start = i
        elif not v and start is not None:
            intervals.append((start, i))
            start = None
    if start is not None:
        intervals.append((start, len(binary_series)))
    return intervals


def overlaps(a, b):
    return a[0] < b[1] and b[0] < a[1]


def event_level_recall_precision(truth_intervals, pred_intervals):
    if not truth_intervals:
        recall = float("nan")
    else:
        recall = sum(any(overlaps(t, p) for p in pred_intervals) for t in truth_intervals) / len(truth_intervals)
    if not pred_intervals:
        precision = float("nan")
    else:
        precision = sum(any(overlaps(t, p) for t in truth_intervals) for p in pred_intervals) / len(pred_intervals)
    return recall, precision


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, mean, std, seq_len = load_model(device)
    assert seq_len == SEQ_LEN

    with open(f"{OUTPUT_DIR}/metrics.json") as f:
        metrics = json.load(f)
    val_pids = metrics["val_pids"]
    print(f"scoring {len(val_pids)} val participants: {val_pids}", file=sys.stderr)

    all_truth, all_pred = [], []
    n_truth_total, n_pred_total = 0, 0
    for pid in val_pids:
        glucose, labels = load_participant_series(f"{CGMACROS_DIR}/{pid}/{pid}.csv")
        glucose_arr = glucose.to_numpy()
        truth_binary = (labels.to_numpy() >= TRUTH_THRESHOLD)

        pred_flags, covered = sliding_window_ratio_vote(model, glucose_arr, mean, std, device)
        truth_intervals = to_intervals(truth_binary, covered)
        pred_intervals = to_intervals(pred_flags, covered)

        r, p = event_level_recall_precision(truth_intervals, pred_intervals)
        n_truth_total += len(truth_intervals)
        n_pred_total += len(pred_intervals)
        all_truth.append(truth_intervals)
        all_pred.append(pred_intervals)
        print(f"  {pid}: {len(truth_intervals)} true events, {len(pred_intervals)} pred events, "
              f"recall={r:.3f} precision={p:.3f}", file=sys.stderr)

    # pooled (all val participants' events combined), matching a single
    # reported percentage the way the paper's own Recall/Precision are
    # stated as one number, not per-participant then averaged.
    pooled_recall = sum(
        sum(any(overlaps(t, p) for p in preds) for t in truths)
        for truths, preds in zip(all_truth, all_pred)
    ) / n_truth_total
    pooled_precision = sum(
        sum(any(overlaps(t, p) for t in truths) for p in preds)
        for truths, preds in zip(all_truth, all_pred)
    ) / n_pred_total

    print(f"\nPOOLED event-level Recall={pooled_recall:.4f} ({pooled_recall*100:.2f}%), "
          f"Precision={pooled_precision:.4f} ({pooled_precision*100:.2f}%) "
          f"[theta={THETA}, rho={RHO}, n_true_events={n_truth_total}, n_pred_events={n_pred_total}]",
          file=sys.stderr)
    print(f"SSM-CGM paper reports: Recall=80.13%, Precision=72.08%", file=sys.stderr)


if __name__ == "__main__":
    main()
