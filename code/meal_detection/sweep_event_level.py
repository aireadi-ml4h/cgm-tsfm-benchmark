"""Diagnostic: is the low precision (55.82% vs SSM-CGM's 72.08%) a mistuned
operating point (theta=0.36, rho=0.77 borrowed from their paper, not
re-tuned for this independently-retrained model), or something deeper?
Caches raw per-window probabilities ONCE per val participant, then sweeps a
small (theta, rho) grid cheaply against the cached probabilities -- avoids
re-running the model per grid point.
"""
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
from cgmacros_dataset import SEQ_LEN, load_participant_series  # noqa: E402
from generate_meal_probability import load_model  # noqa: E402
from evaluate_event_level import to_intervals, event_level_recall_precision  # noqa: E402

REPO_ROOT = "/path/to/aireadi-collab"
OUTPUT_DIR = f"{REPO_ROOT}/model_input/meal_detection"
STRIDE = 1
TRUTH_THRESHOLD = 0.5
THETA_GRID = [0.2, 0.3, 0.36, 0.4, 0.5, 0.6]
RHO_GRID = [0.3, 0.4, 0.5, 0.6, 0.7, 0.77, 0.85]


def cache_window_probs(model, glucose, mean, std, device, batch_size=512):
    n = len(glucose)
    starts = [s for s in range(0, n - SEQ_LEN + 1, STRIDE) if not np.isnan(glucose[s:s + SEQ_LEN]).any()]
    all_probs = np.zeros((len(starts), SEQ_LEN), dtype=float)
    for b in range(0, len(starts), batch_size):
        batch_starts = starts[b:b + batch_size]
        X = np.stack([glucose[s:s + SEQ_LEN] for s in batch_starts])
        X = (X - mean) / std
        with torch.no_grad():
            logits = model(torch.from_numpy(X).float().unsqueeze(-1).to(device))
            probs = torch.sigmoid(logits).cpu().numpy()
        all_probs[b:b + len(batch_starts)] = probs
    return starts, all_probs, n


def ratio_vote(starts, all_probs, n, theta, rho):
    vote_sum = np.zeros(n, dtype=float)
    vote_count = np.zeros(n, dtype=float)
    window_preds = (all_probs > theta).astype(float)
    for i, s in enumerate(starts):
        vote_sum[s:s + SEQ_LEN] += window_preds[i]
        vote_count[s:s + SEQ_LEN] += 1.0
    covered = vote_count > 0
    ratio = np.divide(vote_sum, vote_count, out=np.zeros_like(vote_sum), where=covered)
    return covered & (ratio >= rho), covered


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, mean, std, seq_len = load_model(device)
    assert seq_len == SEQ_LEN

    with open(f"{OUTPUT_DIR}/metrics.json") as f:
        val_pids = json.load(f)["val_pids"]

    cgmacros_dir = os.environ.get("CGMACROS_DIR", "/path/to/cgmacros/1.0.0/extracted/CGMacros")
    cached = []
    for pid in val_pids:
        glucose, labels = load_participant_series(f"{cgmacros_dir}/{pid}/{pid}.csv")
        glucose_arr = glucose.to_numpy()
        truth_binary = (labels.to_numpy() >= TRUTH_THRESHOLD)
        starts, probs, n = cache_window_probs(model, glucose_arr, mean, std, device)
        cached.append((starts, probs, n, truth_binary))
    print(f"cached probabilities for {len(val_pids)} val participants", file=sys.stderr)

    results = []
    for theta in THETA_GRID:
        for rho in RHO_GRID:
            n_truth_total, n_pred_total, n_r_hits, n_p_hits = 0, 0, 0, 0
            for starts, probs, n, truth_binary in cached:
                pred_flags, covered = ratio_vote(starts, probs, n, theta, rho)
                truth_intervals = to_intervals(truth_binary, covered)
                pred_intervals = to_intervals(pred_flags, covered)
                r, p = event_level_recall_precision(truth_intervals, pred_intervals)
                n_truth_total += len(truth_intervals)
                n_pred_total += len(pred_intervals)
                n_r_hits += 0 if len(truth_intervals) == 0 else round(r * len(truth_intervals))
                n_p_hits += 0 if len(pred_intervals) == 0 else round(p * len(pred_intervals))
            recall = n_r_hits / n_truth_total if n_truth_total else float("nan")
            precision = n_p_hits / n_pred_total if n_pred_total else float("nan")
            f1 = 2 * recall * precision / (recall + precision) if (recall + precision) > 0 else 0.0
            results.append((theta, rho, recall, precision, f1, n_pred_total))

    results.sort(key=lambda r: -r[4])
    print("theta  rho   recall  precision  f1     n_pred_events", file=sys.stderr)
    for theta, rho, r, p, f1, npred in results[:15]:
        print(f"{theta:.2f}  {rho:.2f}  {r:.4f}  {p:.4f}   {f1:.4f} {npred}", file=sys.stderr)
    print(f"\n[Their reported operating point: theta=0.36 rho=0.77]", file=sys.stderr)
    for theta, rho, r, p, f1, npred in results:
        if theta == 0.36 and rho == 0.77:
            print(f"  -> our model at their exact point: recall={r:.4f} precision={p:.4f} f1={f1:.4f}", file=sys.stderr)


if __name__ == "__main__":
    main()
