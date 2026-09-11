"""Persistence and linear-extrapolation naive baselines for Chronos-2 and
TimesFM 2.5, scored on the identical eligible-window set used for their
already-published zero-shot glucose-only MAE/RMSE (report.md Sec 1).

Reuses the naive-forecast logic from code/cgmlsm_naive_baseline_check.py
(persistence = last non-missing context value; linear = slope over the last
LINEAR_FIT_BINS context bins, extrapolated forward), applied to
model_input/zeroshot_eval/zeroshot_<horizon>_glucose_only.csv -- the same
source file both code/chronos2/chronos2_zeroshot_eval.py and
code/timesfm/timesfm_zeroshot_eval.py read from (confirmed: both scripts' own
docstrings cite this exact path), so persistence/linear are computed once
per horizon and are valid baselines for both models without re-deriving
per-model window sets.

Writes persistence/linear predictions in the exact schema
chronos2_matched_comparison.compare_arms() expects
(window_id,bin_start,predictions,actual,abs_error), so significance testing
against each model's real zero-shot arm reuses that module exactly rather
than a separately hand-rolled cluster test.
"""
import os
import sys

import numpy as np
import pandas as pd

REPO_ROOT = "/path/to/aireadi-collab"
sys.path.append(os.path.join(REPO_ROOT, "code"))
# Sibling modules live in adjacent topic directories under code/.
_CODE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _d in ("analysis", "baselines", "cgmlsm", "chronos2", "meal_detection", "model_input", "ssmcgm", "timesfm"):
    sys.path.insert(0, os.path.join(_CODE, _d))
from chronos2_matched_comparison import compare_arms  # noqa: E402

EVAL_DIR = f"{REPO_ROOT}/model_input/zeroshot_eval"
OUT_ROOT = f"{REPO_ROOT}/naive_baselines_results"
HORIZONS = {"30min": 6, "1h": 12, "2h": 24, "3h": 36, "4h": 48}
LINEAR_FIT_BINS = 6  # 30 min of context, matching cgmlsm_naive_baseline_check.py

os.makedirs(f"{OUT_ROOT}/persistence", exist_ok=True)
os.makedirs(f"{OUT_ROOT}/linear_extrap", exist_ok=True)


def naive_forecasts(win):
    """Same method as cgmlsm_naive_baseline_check.py's naive_forecasts():
    anchor = last non-missing context value; linear = slope over the last
    LINEAR_FIT_BINS context bins, extrapolated forward from the anchor."""
    ctx = win[win["is_context"]].sort_values("bin_start")
    hor = win[~win["is_context"]].sort_values("bin_start")
    obs = ctx["cgm_glucose"].to_numpy(dtype=float)
    valid = obs[~np.isnan(obs)]
    if len(valid) == 0:
        return None
    anchor = valid[-1]

    tail = obs[-LINEAR_FIT_BINS:]
    idx = np.arange(len(tail))[~np.isnan(tail)]
    if len(idx) >= 2:
        slope = np.polyfit(idx, tail[~np.isnan(tail)], 1)[0]
    else:
        slope = 0.0
    steps = np.arange(1, len(hor) + 1)

    return pd.DataFrame({
        "window_id": win["window_id"].iloc[0],
        "bin_start": hor["bin_start"].to_numpy(),
        "actual": hor["cgm_glucose"].to_numpy(dtype=float),
        "persistence": anchor,
        "linear": anchor + slope * steps,
    })


rows = []
for horizon in HORIZONS:
    path = f"{EVAL_DIR}/zeroshot_{horizon}_glucose_only.csv"
    print(f"{horizon}: reading {path}", file=sys.stderr)
    w = pd.read_csv(path, dtype={"window_id": str, "person_id": str})
    w["is_context"] = w["is_context"].astype(bool)

    naive = pd.concat(
        [f for f in (naive_forecasts(g) for _, g in w.groupby("window_id")) if f is not None],
        ignore_index=True)

    for method, col in [("persistence", "persistence"), ("linear_extrap", "linear")]:
        out = naive[["window_id", "bin_start", "actual"]].copy()
        out["predictions"] = naive[col]
        out["abs_error"] = (out["predictions"] - out["actual"]).abs()
        out.to_csv(f"{OUT_ROOT}/{method}/predictions_{horizon}_glucose_only.csv", index=False)

    for method in ("persistence", "linear_extrap"):
        scored = pd.read_csv(f"{OUT_ROOT}/{method}/predictions_{horizon}_glucose_only.csv").dropna(subset=["actual"])
        mae = scored["abs_error"].mean()
        rmse = float(np.sqrt((scored["abs_error"] ** 2).mean()))
        rows.append({"model": method, "horizon": horizon, "method": method,
                     "mae": round(mae, 4), "rmse": round(rmse, 4), "n_bins": len(scored)})
    print(f"  {horizon}: wrote persistence + linear_extrap, {len(naive)} scored rows", file=sys.stderr)

naive_df = pd.DataFrame(rows)

# Real model numbers, from the already-published zero-shot summary files.
model_rows = []
sig_rows = []
for model, results_dir in [
    ("chronos2", f"{REPO_ROOT}/chronos2_zeroshot_results/full/imputed"),
    ("timesfm", f"{REPO_ROOT}/timesfm_zeroshot_results/full/imputed"),
]:
    summary = pd.read_csv(f"{results_dir}/summary.csv")
    for horizon, hbins in HORIZONS.items():
        m = summary[(summary["horizon"] == horizon) & (summary["with_covariates"] == False)]  # noqa: E712
        if m.empty:
            continue
        model_rows.append({"model": model, "horizon": horizon, "method": "model",
                            "mae": round(float(m["mae_mgdl"].iloc[0]), 4),
                            "rmse": round(float(m["rmse_mgdl"].iloc[0]), 4),
                            "n_bins": int(m["n_scored"].iloc[0])})
        arm_model = dict(label=model, results_dir=results_dir, condition="glucose_only")
        for baseline in ("persistence", "linear_extrap"):
            arm_base = dict(label=baseline, results_dir=f"{OUT_ROOT}/{baseline}", condition="glucose_only")
            res = compare_arms(horizon, hbins, arm_model, arm_base)
            sig_rows.append({
                "model": model, "horizon": horizon, "vs": baseline,
                "model_mae_matched": round(res[f"{model}_mae_matched"], 4),
                f"{baseline}_mae_matched": round(res[f"{baseline}_mae_matched"], 4),
                "n_paired_windows": res["n_paired_windows"],
                "n_participants_clustered": res["n_participants_clustered"],
                "cluster_by_participant_p": res["cluster_by_participant_p"],
                "model_beats_baseline": res[f"{model}_mae_matched"] < res[f"{baseline}_mae_matched"],
            })

all_df = pd.concat([naive_df, pd.DataFrame(model_rows)], ignore_index=True)
all_df.to_csv(f"{OUT_ROOT}/naive_vs_model_chronos_timesfm.csv", index=False)
sig_df = pd.DataFrame(sig_rows)
sig_df.to_csv(f"{OUT_ROOT}/naive_vs_model_significance.csv", index=False)

print("\n=== MAE, model vs naive baselines (Chronos-2 / TimesFM, glucose-only, test) ===")
print(all_df.pivot_table(index=["model", "horizon"], columns="method", values="mae").to_string())
print("\n=== significance: model vs persistence/linear (cluster-by-participant) ===")
print(sig_df.to_string(index=False))
print("\nNAIVE_BASELINES_CHRONOS_TIMESFM_OK")
