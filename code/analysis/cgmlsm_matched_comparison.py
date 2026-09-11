"""Matched arm-vs-arm comparison for CGM-LSM personalization arms -- the
CGM-LSM counterpart of code/analysis/chronos2_matched_comparison.py, and the ONLY
sanctioned way to compare two arms in this project.

NEVER DIFF summary.csv. Two arms can cover different windows (a participant
whose history yields no eligible fit chunks is skipped in one arm and present
in another) and, for CGM-LSM specifically, can also cover different HORIZON
BINS within the same window, because a non-numeric (`UNKNOWN`) prediction is
dropped before MAE. Either difference makes two summary MAEs averages over
different populations. This module intersects on (person_id, window_id,
bin_start) and additionally requires BOTH arms to have produced a numeric
prediction for a bin before that bin contributes to either arm's score.

WHY THE SECOND CONDITION MATTERS HERE AND NOT FOR CHRONOS-2: CGM-LSM can
decline to forecast. On the from-scratch checkpoint that was catastrophic
(16.2% of 2h bins, an absorbing collapse -- see
code/analysis/cgmlsm_arm_matched_comparison.py); on the settled checkpoint this arm
uses it is ~0, but "~0 today" is not a guarantee that a fine-tuned copy stays
at 0, and a personalized arm that merely collapsed less often would otherwise
look better for a reason that has nothing to do with forecast quality.
n_nonnumeric_a/b are returned on every comparison so that drift is visible
rather than silently absorbed into the denominator.

SIGN CONVENTION, matching chronos2_matched_comparison.compare_arms: the
differential is arm_a - arm_b, so POSITIVE delta_mae means arm_b is better.
Call with arm_a = the zero-shot/baseline arm and arm_b = the personalized arm,
and then "positive delta_mae" reads as "personalization helped".

CLUSTERING: the unit of independence is the PARTICIPANT, not the window. One
participant contributes many overlapping windows whose errors are strongly
dependent, so a window-level t-test would badly overstate significance. The
paired test here reduces each participant to their own mean squared-error
difference first, then tests across participants.

P-VALUES ARE FOR THE AUTHOR'S REFERENCE ONLY and must never be used to label a
result as helped/hurt in any reported table -- the project's standing reporting
rule (code/CHRONOS2_PERSONALIZATION.md). Report the estimate and its CI.
"""
import math
import os

import numpy as np
import pandas as pd

KEY_COLS = ["person_id", "window_id", "bin_start"]


def load_arm(arm_dir, horizon):
    """One arm's per-bin predictions at one horizon, keyed for matching.
    `is_numeric` distinguishes "the model produced a forecast" from "the model
    emitted PAD/UNKNOWN/MASK", which is not the same thing as missing ground
    truth and must not be conflated with it."""
    path = os.path.join(arm_dir, f"predictions_{horizon}.csv")
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path, dtype={"window_id": str, "person_id": str, "prediction_token": str})
    df["is_numeric"] = df["prediction_token"].str.fullmatch(r"\d+").fillna(False)
    df["has_truth"] = df["actual"].notna()
    return df[KEY_COLS + ["is_numeric", "has_truth", "predictions", "actual", "abs_error"]]


def _cluster_paired_test(per_participant_diff):
    """Paired t across participants on their own mean loss differences.
    Returns (t, p, mean, se, ci_lo, ci_hi) with a normal approximation for the
    CI -- with 80-264 participants the t/normal difference is immaterial, and
    scipy is not a dependency of venv-cgmlsm."""
    d = np.asarray(per_participant_diff, dtype=float)
    d = d[np.isfinite(d)]
    n = len(d)
    if n < 2:
        return (np.nan,) * 6
    mean = float(d.mean())
    se = float(d.std(ddof=1) / np.sqrt(n))
    if se == 0:
        return (np.nan, np.nan, mean, 0.0, mean, mean)
    t = mean / se
    # Two-sided normal-approximation p; author-reference only, never a label.
    p = float(2.0 * (1.0 - 0.5 * (1.0 + math.erf(abs(t) / math.sqrt(2.0)))))
    return t, p, mean, se, mean - 1.96 * se, mean + 1.96 * se


def compare_arms(arm_a_dir, arm_b_dir, horizon, restrict_keys=None, label_a="a", label_b="b"):
    """Matched comparison at one horizon. restrict_keys, if given, is a set of
    (person_id, window_id, bin_start) tuples to further restrict to -- used when
    several arms must all be scored on one common bin set rather than on each
    pair's own intersection, which is what makes a multi-arm table internally
    consistent."""
    a, b = load_arm(arm_a_dir, horizon), load_arm(arm_b_dir, horizon)
    if a is None or b is None:
        return None

    m = a.merge(b, on=KEY_COLS, how="inner", suffixes=("_a", "_b"))
    n_intersect = len(m)
    if restrict_keys is not None:
        keep = pd.MultiIndex.from_frame(m[KEY_COLS]).isin(restrict_keys)
        m = m[keep]

    usable = m[m["is_numeric_a"] & m["is_numeric_b"] & m["has_truth_a"] & m["has_truth_b"]].copy()
    if usable.empty:
        return None

    usable["sq_a"] = usable["abs_error_a"] ** 2
    usable["sq_b"] = usable["abs_error_b"] ** 2

    # TWO SEPARATE CLUSTERED TESTS ON TWO DIFFERENT LOSSES, AND THEY MUST NOT BE
    # MIXED IN ONE REPORTED ROW. An earlier version returned the CI from the MAE
    # contrast next to the p-value from the SQUARED-ERROR contrast, which
    # produced rows like "delta +1.49, CI [0.87, 2.40], p=0.85" -- a CI
    # excluding zero beside a p-value near 1. Both numbers were individually
    # correct and they were answering different questions. Each loss now carries
    # its own t/p/CI, named for the loss it belongs to.
    per_p_mae = usable.groupby("person_id").apply(
        lambda g: (g["abs_error_a"] - g["abs_error_b"]).mean())
    mae_t, mae_p, mae_mean_d, mae_se_d, mae_ci_lo, mae_ci_hi = _cluster_paired_test(
        per_p_mae.to_numpy())

    per_p = usable.groupby("person_id").apply(lambda g: (g["sq_a"] - g["sq_b"]).mean())
    t, p, mean_d, se_d, ci_lo, ci_hi = _cluster_paired_test(per_p.to_numpy())

    return {
        "horizon": horizon,
        "arm_a": label_a,
        "arm_b": label_b,
        "n_bins_intersect": n_intersect,
        "n_bins_scored_matched": len(usable),
        "n_windows_matched": usable["window_id"].nunique(),
        "n_participants": usable["person_id"].nunique(),
        f"mae_{label_a}": round(float(usable["abs_error_a"].mean()), 4),
        f"mae_{label_b}": round(float(usable["abs_error_b"].mean()), 4),
        f"rmse_{label_a}": round(float(np.sqrt(usable["sq_a"].mean())), 4),
        f"rmse_{label_b}": round(float(np.sqrt(usable["sq_b"].mean())), 4),
        # POSITIVE = arm_b better.
        # delta_mae is POOLED over bins; delta_mae_clustered_mean averages each
        # participant first and then across participants. They are different
        # estimands and can differ when participants contribute unequal numbers
        # of bins. The CI and p below belong to the CLUSTERED one -- quote that
        # pair together, not the pooled estimate with the clustered CI.
        "delta_mae": round(float(usable["abs_error_a"].mean() - usable["abs_error_b"].mean()), 4),
        "delta_mae_clustered_mean": round(mae_mean_d, 4),
        "delta_mae_clustered_se": round(mae_se_d, 4),
        "delta_mae_ci_lo": round(mae_ci_lo, 4),
        "delta_mae_ci_hi": round(mae_ci_hi, 4),
        "delta_mae_cluster_t": round(mae_t, 4) if np.isfinite(mae_t) else np.nan,
        "delta_mae_cluster_p": mae_p,
        # The squared-error (MSE) contrast, a Diebold-Mariano-style loss
        # differential. Separate question, separate row of statistics.
        "mean_loss_diff_mse": round(mean_d, 4),
        "mse_cluster_t": round(t, 4) if np.isfinite(t) else np.nan,
        "mse_cluster_p": p,
        # Win rate over participants: fraction for whom arm_b had lower MAE.
        "win_rate_b": round(float((per_p_mae > 0).mean()), 4),
        # Denominator-drift watchdogs; see module docstring.
        f"n_nonnumeric_{label_a}": int((~m["is_numeric_a"]).sum()),
        f"n_nonnumeric_{label_b}": int((~m["is_numeric_b"]).sum()),
    }


def common_scored_keys(arm_dirs, horizon):
    """Keys where EVERY arm in arm_dirs produced a numeric prediction and ground
    truth exists -- pass to compare_arms(restrict_keys=...) so a multi-arm table
    is built on one bin set instead of a different intersection per pair."""
    keys = None
    for d in arm_dirs:
        df = load_arm(d, horizon)
        if df is None:
            return None
        ok = df[df["is_numeric"] & df["has_truth"]]
        k = set(map(tuple, ok[KEY_COLS].to_numpy()))
        keys = k if keys is None else (keys & k)
    return keys
