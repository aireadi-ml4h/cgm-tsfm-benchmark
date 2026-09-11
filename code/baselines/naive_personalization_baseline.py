"""Naive personalization baseline: does a trivial, per-person BIAS
CORRECTION (no retraining, no in-context extension -- just a scalar
calibration shift) capture most of the benefit that real personalization
(Chronos-2 LoRA, TimesFM in-context) provides?

MECHANISM. For each held-out test participant and each personalization
condition (k=1/3/7 days), reusing the EXACT SAME causal anchor and history
window already used by that person's real personalization arm
(`model_input/personalization/eval_set_test.csv` -- built by
`build_personalization_eval_set.py`, never recomputed differently here):

  1. Pull the zero-shot model's own predictions on that person's PRE-ANCHOR
     history, i.e. the same k-day window their real personalization arm
     fits on -- these are ordinary zero-shot windows that already exist in
     the full zero-shot prediction files (chronos2/timesfm), no new
     inference needed. Verified: the zero-shot eligible window grid for a
     participant spans their whole record (window_id 0, 1, 2, ... in
     calendar order), so every pre-anchor history window a real
     personalization arm uses was already scored zero-shot.
  2. bias(person, horizon, k) = mean(actual - predicted) over those
     pre-anchor windows only -- never touching the real evaluation windows.
  3. naive-personalized prediction on a REAL evaluation window (from
     eval_set_test.csv, i.e. the same windows the real p{k}d/icl_p{k}d arm
     is scored on) = zero-shot prediction + bias(person, horizon, k).
  4. Written out as a real predictions_{horizon}_glucose_only.csv file in
     the same schema as every other arm in this project, so
     chronos2_matched_comparison.compare_arms() -- the same
     significance-testing machinery used everywhere else in this project --
     can compare it against both zero-shot and the real personalization arm
     with no new logic.

CAUSALITY. bias is computed ONLY from windows with context_start in
[history_start_kd, anchor), i.e. strictly before anchor. Evaluation windows
scored are restricted to eval_set_test.csv's own key set (context_start >=
anchor). No overlap between the two by construction (same guarantee
build_personalization_eval_set.py already establishes for the real
personalization arms).

MODELS COVERED: Chronos-2 (vs. LoRA p1d/p3d/p7d) and TimesFM (vs. in-context
incontext_1d/3d/7d). CGM-LSM's own naive_baseline_check.py already asks a
related-but-different question (does fine-tuning beat literal persistence,
not does it beat a bias-corrected zero-shot) and was not extended here for
time. SSM-CGM is handled by a separate script
(ssmcgm_naive_personalization_baseline.py) since it has no comparable
zero-shot-window-reuse path.
"""
import os
import sys

import numpy as np
import pandas as pd

REPO_ROOT = "/path/to/aireadi-collab"
sys.path.insert(0, os.path.join(REPO_ROOT, "code"))
# Sibling modules live in adjacent topic directories under code/.
_CODE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _d in ("analysis", "baselines", "cgmlsm", "chronos2", "meal_detection", "model_input", "ssmcgm", "timesfm"):
    sys.path.insert(0, os.path.join(_CODE, _d))
from chronos2_matched_comparison import load_window_keys, compare_arms  # noqa: E402

EVAL_SET = pd.read_csv(f"{REPO_ROOT}/model_input/personalization/eval_set_test.csv",
                        dtype={"person_id": str})
for c in ["context_start", "anchor", "history_start_1d", "history_start_3d", "history_start_7d"]:
    EVAL_SET[c] = pd.to_datetime(EVAL_SET[c], utc=True)

HORIZONS = {"30min": 6, "1h": 12, "2h": 24, "3h": 36, "4h": 48}
FIT_FOR_HORIZON = {"30min": "fit6", "1h": "fit24", "2h": "fit24", "3h": "fit24", "4h": "fit24"}

MODELS = {
    "chronos2": {
        "zeroshot_dir": f"{REPO_ROOT}/chronos2_zeroshot_results/full/imputed",
        "real_pers_dir": lambda k, horizon: f"{REPO_ROOT}/chronos2_personalization_results/test/p{k}d/{FIT_FOR_HORIZON[horizon]}",
        "naive_out_dir": lambda k: f"{REPO_ROOT}/naive_personalization_results/chronos2/test/naive_p{k}d",
        "eval_dir": f"{REPO_ROOT}/model_input/zeroshot_eval",
    },
    "timesfm": {
        "zeroshot_dir": f"{REPO_ROOT}/timesfm_zeroshot_results/full/imputed",
        "real_pers_dir": lambda k, horizon: f"{REPO_ROOT}/timesfm_personalization_results/test/incontext_{k}d",
        "naive_out_dir": lambda k: f"{REPO_ROOT}/naive_personalization_results/timesfm/test/naive_incontext_{k}d",
        "eval_dir": f"{REPO_ROOT}/model_input/zeroshot_eval",
    },
}

results_rows = []

for model_name, cfg in MODELS.items():
    for horizon in HORIZONS:
        keys = load_window_keys(horizon, "glucose_only", cfg["eval_dir"])  # window_id -> person_id, context_start, key
        zs_pred = pd.read_csv(f"{cfg['zeroshot_dir']}/predictions_{horizon}_glucose_only.csv",
                               dtype={"window_id": str})
        zs_pred = zs_pred.merge(keys.reset_index(), on="window_id", how="left")

        eval_h = EVAL_SET[EVAL_SET["horizon"] == horizon]

        for k in (1, 3, 7):
            history_col = f"history_start_{k}d"
            # bias per person: mean(actual - predicted) over pre-anchor windows
            # [history_start_kd, anchor) -- strictly causal, never touches eval windows.
            anchors = eval_h[["person_id", "anchor", history_col]].drop_duplicates("person_id")
            pre = zs_pred.merge(anchors, on="person_id", how="inner")
            pre = pre[(pre["context_start"] >= pre[history_col]) & (pre["context_start"] < pre["anchor"])]
            pre = pre.dropna(subset=["actual"])
            bias = pre.groupby("person_id").apply(lambda g: (g["actual"] - g["predictions"]).mean())
            n_bias_windows = pre.groupby("person_id")["context_start"].nunique()

            # apply bias to the REAL evaluation windows (eval_set_test.csv's own keys)
            eval_keys = set(eval_h["key"])
            eval_scored = zs_pred[zs_pred["key"].isin(eval_keys)].copy()
            eval_scored["bias"] = eval_scored["person_id"].map(bias)
            n_missing_bias = eval_scored["bias"].isna().sum()
            eval_scored = eval_scored.dropna(subset=["bias"])  # participants with 0 pre-anchor windows for this k: excluded, can't be bias-corrected
            eval_scored["predictions"] = eval_scored["predictions"] + eval_scored["bias"]
            eval_scored["abs_error"] = (eval_scored["predictions"] - eval_scored["actual"]).abs()

            out_dir = cfg["naive_out_dir"](k)
            os.makedirs(out_dir, exist_ok=True)
            out_cols = ["window_id", "bin_start", "predictions", "actual", "abs_error"]
            eval_scored[out_cols].to_csv(f"{out_dir}/predictions_{horizon}_glucose_only.csv", index=False)

            results_rows.append({
                "model": model_name, "horizon": horizon, "k": k,
                "n_bias_windows_median": float(n_bias_windows.median()) if len(n_bias_windows) else np.nan,
                "n_participants_missing_bias": int(n_missing_bias),
                "n_eval_windows_scored": len(eval_scored),
                "naive_personalized_mae": float(eval_scored["abs_error"].mean()),
            })
            print(f"{model_name} {horizon} k={k}d: naive-personalized MAE={eval_scored['abs_error'].mean():.4f}, "
                  f"n_eval_bins={len(eval_scored)}, missing_bias_rows_dropped={n_missing_bias}", file=sys.stderr)

pd.DataFrame(results_rows).to_csv(f"{REPO_ROOT}/naive_personalization_results/naive_personalization_summary.csv", index=False)

# Now the real 3-way comparisons via compare_arms(): zero-shot vs naive vs real personalization.
comparison_rows = []
for model_name, cfg in MODELS.items():
    zeroshot_arm = {"label": "zeroshot", "results_dir": cfg["zeroshot_dir"], "condition": "glucose_only",
                     "eval_dir": cfg["eval_dir"]}
    for horizon, hbins in HORIZONS.items():
        eval_h = EVAL_SET[EVAL_SET["horizon"] == horizon]
        restrict_keys = set(eval_h["key"])
        for k in (1, 3, 7):
            naive_arm = {"label": f"naive_p{k}d", "results_dir": cfg["naive_out_dir"](k), "condition": "glucose_only",
                         "eval_dir": cfg["eval_dir"]}
            real_dir = cfg["real_pers_dir"](k, horizon)
            real_arm = {"label": f"real_p{k}d", "results_dir": real_dir, "condition": "glucose_only",
                        "eval_dir": cfg["eval_dir"]}
            if not os.path.exists(f"{real_dir}/predictions_{horizon}_glucose_only.csv"):
                print(f"SKIP {model_name} {horizon} k={k}d: no real personalization file at {real_dir}", file=sys.stderr)
                continue

            r1 = compare_arms(horizon, hbins, zeroshot_arm, naive_arm, restrict_keys=restrict_keys)
            r2 = compare_arms(horizon, hbins, naive_arm, real_arm, restrict_keys=restrict_keys)
            comparison_rows.append({
                "model": model_name, "horizon": horizon, "k": k,
                "zeroshot_mae": r1["zeroshot_mae_matched"], "naive_p_mae": r1["naive_p" + str(k) + "d_mae_matched"],
                "real_p_mae": r2["real_p" + str(k) + "d_mae_matched"],
                "n_matched_zeroshot_vs_naive": r1["n_matched_windows"],
                "n_matched_naive_vs_real": r2["n_matched_windows"],
                "naive_vs_real_cluster_p": r2.get("cluster_by_participant_p"),
                "naive_vs_real_cluster_t": r2.get("cluster_by_participant_t"),
            })

comp_df = pd.DataFrame(comparison_rows)
comp_df.to_csv(f"{REPO_ROOT}/naive_personalization_results/naive_vs_real_comparison.csv", index=False)
print(comp_df.to_string(index=False), file=sys.stderr)
print("NAIVE_PERSONALIZATION_BASELINE_OK", file=sys.stderr)
