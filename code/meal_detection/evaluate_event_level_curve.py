"""Real, swept event-level precision-recall curve for the meal detector,
replacing evaluate_event_level.py's 3 hand-transcribed operating points with
an actual curve. Reuses that script's exact inference and event-matching
logic verbatim (imported, not reimplemented) -- the only change is that
sliding_window_ratio_vote's internal continuous ratio(t) array is exposed
and thresholded at many RHO values, instead of only the paper's RHO=0.77,
which the original script hardcoded and immediately discarded.

Model inference (the sliding-window forward passes) runs exactly ONCE per
validation participant, same as the original script -- ratio(t) does not
depend on RHO, so sweeping RHO afterward costs nothing extra on the GPU,
only cheap CPU interval-matching per sweep point. THETA (window-level
sigmoid threshold, 0.36) is held fixed at the paper's value throughout, so
this is a genuine one-parameter operating-point sweep of the paper's own
ratio-voting threshold, not a different method.
"""
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
from cgmacros_dataset import CGMACROS_DIR, SEQ_LEN, load_participant_series  # noqa: E402
from generate_meal_probability import load_model  # noqa: E402
from evaluate_event_level import (  # noqa: E402
    to_intervals, overlaps, event_level_recall_precision, THETA, TRUTH_THRESHOLD, OUTPUT_DIR,
)

STRIDE = 1
RHO_GRID = np.round(np.arange(0.10, 0.96, 0.05), 2)


def sliding_window_ratio_continuous(model, glucose, mean, std, device, batch_size=512):
    """Same as evaluate_event_level.sliding_window_ratio_vote, but returns
    the continuous ratio(t) array (fraction of covering windows voting 1,
    at THETA's fixed window-level threshold) instead of a single RHO's
    already-thresholded boolean flags."""
    n = len(glucose)
    vote_sum = np.zeros(n, dtype=float)
    vote_count = np.zeros(n, dtype=float)
    starts = [s for s in range(0, n - SEQ_LEN + 1, STRIDE) if not np.isnan(glucose[s:s + SEQ_LEN]).any()]
    if not starts:
        return np.zeros(n, dtype=float), np.zeros(n, dtype=bool)
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
    return ratio, covered


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, mean, std, seq_len = load_model(device)
    assert seq_len == SEQ_LEN

    with open(f"{OUTPUT_DIR}/metrics.json") as f:
        metrics = json.load(f)
    val_pids = metrics["val_pids"]
    print(f"scoring {len(val_pids)} val participants: {val_pids}", file=sys.stderr)

    per_participant = []
    for pid in val_pids:
        glucose, labels = load_participant_series(f"{CGMACROS_DIR}/{pid}/{pid}.csv")
        glucose_arr = glucose.to_numpy()
        truth_binary = (labels.to_numpy() >= TRUTH_THRESHOLD)
        ratio, covered = sliding_window_ratio_continuous(model, glucose_arr, mean, std, device)
        truth_intervals = to_intervals(truth_binary, covered)
        per_participant.append((ratio, covered, truth_intervals))
        print(f"  {pid}: {len(truth_intervals)} true events, ratio range "
              f"[{ratio[covered].min():.3f}, {ratio[covered].max():.3f}]", file=sys.stderr)

    rows = []
    for rho in RHO_GRID:
        n_truth_total, n_pred_total = 0, 0
        all_truth, all_pred = [], []
        for ratio, covered, truth_intervals in per_participant:
            pred_flags = covered & (ratio >= rho)
            pred_intervals = to_intervals(pred_flags)
            all_truth.append(truth_intervals)
            all_pred.append(pred_intervals)
            n_truth_total += len(truth_intervals)
            n_pred_total += len(pred_intervals)
        if n_truth_total == 0 or n_pred_total == 0:
            rows.append({"rho": rho, "recall": np.nan, "precision": np.nan,
                         "n_true_events": n_truth_total, "n_pred_events": n_pred_total})
            continue
        pooled_recall = sum(
            sum(any(overlaps(t, p) for p in preds) for t in truths)
            for truths, preds in zip(all_truth, all_pred)
        ) / n_truth_total
        pooled_precision = sum(
            sum(any(overlaps(t, p) for t in truths) for p in preds)
            for truths, preds in zip(all_truth, all_pred)
        ) / n_pred_total
        rows.append({"rho": rho, "recall": pooled_recall, "precision": pooled_precision,
                     "n_true_events": n_truth_total, "n_pred_events": n_pred_total})
        print(f"rho={rho:.2f}: recall={pooled_recall:.4f} precision={pooled_precision:.4f} "
              f"n_true={n_truth_total} n_pred={n_pred_total}", file=sys.stderr)

    import pandas as pd
    out_path = f"{OUTPUT_DIR}/event_level_pr_curve.csv"
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"wrote {out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
