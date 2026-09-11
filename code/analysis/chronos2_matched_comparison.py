"""Shared matched-window comparison for any two Chronos-2 result arms.

Comparing two arms' raw summary.csv MAE/RMSE is confounded whenever their
underlying window sets differ in size -- most notably glucose_only (only
cgm_glucose coverage required) vs any with_covariates arm, including every
LOO ablation arm, since with_covariates additionally requires heart_rate +
respiratory_rate coverage and is therefore a strict subset of glucose_only's
eligible windows at every horizon (confirmed in code/CHRONOS2_EVAL.md).
Comparing unrestricted numbers mixes the effect being tested (what a
covariate/ablation does to the forecast) with a second, unrelated effect
(the two arms scoring different, differently-sized populations). This
module is the one place that restriction logic lives -- every Chronos-2
arm-vs-arm comparison (glucose_only vs with_covariates, baseline vs a LOO
ablation arm, one LOO arm vs another) should call compare_arms() rather than
diffing summary.csv rows directly.

Originally written as chronos2_covariate_comparison.py's inline logic
(glucose_only vs with_covariates only, same results_dir for both); pulled
out here, unchanged in method, so LOO ablation arms (different results_dir
per arm, e.g. chronos2_zeroshot_results_loo_drop_oxygen_saturation/) can
reuse it.
"""
import numpy as np
import pandas as pd
from scipy import stats

REPO_ROOT = "/path/to/aireadi-collab"
DEFAULT_EVAL_DIR = f"{REPO_ROOT}/model_input/zeroshot_eval"


def load_window_keys(horizon_name, condition, eval_dir=DEFAULT_EVAL_DIR):
    """window_id -> (person_id, context_start) for one zeroshot_eval file.
    window_id is NOT comparable across arms/conditions (export_zeroshot_windows.py
    enumerates it independently per (horizon, with_covariates) subset, and an
    LOO ablation arm re-reads the same export as the with_covariates baseline
    -- see chronos2_zeroshot_eval.py's ACTIVE_COVARIATE_FEATURE_CHANNELS), so
    the real join key across arms has to be reconstructed from the underlying
    context, not read off window_id directly."""
    path = f"{eval_dir}/zeroshot_{horizon_name}_{condition}.csv"
    df = pd.read_csv(path, usecols=["window_id", "person_id", "bin_start", "is_context"],
                      dtype={"window_id": str, "person_id": str})
    df["bin_start"] = pd.to_datetime(df["bin_start"], utc=True)
    df["is_context"] = df["is_context"].astype(bool)
    ctx = df[df["is_context"]]
    keys = ctx.groupby("window_id").agg(person_id=("person_id", "first"), context_start=("bin_start", "min"))
    keys["key"] = keys["person_id"] + "_" + keys["context_start"].astype(str)
    return keys


def per_window_mse(pred_df, window_to_key):
    d = pred_df.dropna(subset=["actual"]).copy()
    d["sq_error"] = (d["predictions"] - d["actual"]) ** 2
    d["key"] = d["window_id"].map(window_to_key)
    return d.groupby("key")["sq_error"].mean()


def dm_hln_test(d, h):
    """Diebold-Mariano + Harvey-Leybourne-Newbold small-sample correction.
    d: 1D array of loss differentials. h: lag truncation (horizon_bins), per
    the DM/HLN fixed-bandwidth convention -- see chronos2_covariate_comparison.py's
    original docstring for why this doesn't perfectly fit this project's
    non-overlapping, multi-participant window design (reported anyway
    because it was specifically asked for)."""
    d = np.asarray(d, dtype=float)
    T = len(d)
    dbar = d.mean()
    dc = d - dbar
    gamma0 = float(np.mean(dc ** 2))
    var_sum = gamma0
    for k in range(1, min(h, T)):
        gamma_k = float(np.mean(dc[:-k] * dc[k:]))
        var_sum += 2 * gamma_k
    var_dbar = var_sum / T
    if var_dbar <= 0:
        return np.nan, np.nan, T
    dm = dbar / np.sqrt(var_dbar)
    correction = np.sqrt(max((T + 1 - 2 * h + h * (h - 1) / T) / T, 0))
    dm_star = dm * correction
    p_value = 2 * (1 - stats.t.cdf(abs(dm_star), df=T - 1))
    return dm_star, p_value, T


def compare_arms(horizon_name, horizon_bins, arm_a, arm_b, restrict_keys=None):
    """arm_a/arm_b: dict with keys label, results_dir, condition
    ("glucose_only" or "with_covariates"), and optional eval_dir (defaults to
    DEFAULT_EVAL_DIR -- correct for every arm here, since LOO ablation
    changes which columns chronos2_zeroshot_eval.py feeds the model, not the
    exported window content itself).

    Restricts both arms to their common (person_id, context_start) window
    set before computing matched MAE/RMSE and three paired tests on the
    per-window MSE differential (arm_a - arm_b; positive => arm_b has lower
    loss, i.e. arm_b is better).

    restrict_keys: optional iterable of `key` strings
    (f"{person_id}_{context_start}") to intersect with the common window set,
    for a STRATIFIED comparison -- same two arms, same tests, run separately
    on a subset of windows. Added 2026-09-01 for the sleep_regularity_index
    day-boundary test (chronos2_sri_dayboundary_test.py), which needs
    "constant-in-window" and "varying-in-window" windows compared
    independently. Passing None reproduces the original unstratified
    behavior exactly, so every existing caller is unaffected."""
    keys_a = load_window_keys(horizon_name, arm_a["condition"], arm_a.get("eval_dir", DEFAULT_EVAL_DIR))
    keys_b = load_window_keys(horizon_name, arm_b["condition"], arm_b.get("eval_dir", DEFAULT_EVAL_DIR))
    common_keys = set(keys_a["key"]) & set(keys_b["key"])
    if restrict_keys is not None:
        common_keys &= set(restrict_keys)

    pred_a = pd.read_csv(f"{arm_a['results_dir']}/predictions_{horizon_name}_{arm_a['condition']}.csv",
                          dtype={"window_id": str})
    pred_b = pd.read_csv(f"{arm_b['results_dir']}/predictions_{horizon_name}_{arm_b['condition']}.csv",
                          dtype={"window_id": str})

    win_to_key_a = keys_a["key"].to_dict()
    win_to_key_b = keys_b["key"].to_dict()
    key_to_person = keys_a.reset_index().set_index("key")["person_id"].to_dict()

    pred_a = pred_a.assign(key=pred_a["window_id"].map(win_to_key_a))
    pred_b = pred_b.assign(key=pred_b["window_id"].map(win_to_key_b))

    matched_a = pred_a[pred_a["key"].isin(common_keys)]
    matched_b = pred_b[pred_b["key"].isin(common_keys)]

    scored_a = matched_a.dropna(subset=["actual"])
    scored_b = matched_b.dropna(subset=["actual"])

    label_a, label_b = arm_a["label"], arm_b["label"]
    row = {
        "horizon": horizon_name,
        "arm_a": label_a, "arm_b": label_b,
        "n_matched_windows": len(common_keys),
        f"{label_a}_mae_matched": scored_a["abs_error"].mean(),
        f"{label_a}_rmse_matched": float(np.sqrt((scored_a["abs_error"] ** 2).mean())),
        f"{label_b}_mae_matched": scored_b["abs_error"].mean(),
        f"{label_b}_rmse_matched": float(np.sqrt((scored_b["abs_error"] ** 2).mean())),
    }

    loss_a = per_window_mse(matched_a, win_to_key_a)
    loss_b = per_window_mse(matched_b, win_to_key_b)
    paired_keys = loss_a.index.intersection(loss_b.index)
    d = loss_a.loc[paired_keys] - loss_b.loc[paired_keys]

    naive_t, naive_p = stats.ttest_1samp(d.to_numpy(), 0.0)
    dm_stat, dm_p, dm_T = dm_hln_test(d.to_numpy(), horizon_bins)

    person_of_key = pd.Series({k: key_to_person[k] for k in paired_keys})
    d_by_person = d.groupby(person_of_key).mean()
    clust_t, clust_p = stats.ttest_1samp(d_by_person.to_numpy(), 0.0)

    row.update({
        "n_paired_windows": len(paired_keys),
        "mean_loss_diff_mse": d.mean(),
        "naive_paired_t": naive_t, "naive_paired_p": naive_p,
        "dm_hln_stat": dm_stat, "dm_hln_p": dm_p, "dm_hln_T": dm_T,
        "n_participants_clustered": len(d_by_person),
        "cluster_by_participant_t": clust_t, "cluster_by_participant_p": clust_p,
    })
    return row
