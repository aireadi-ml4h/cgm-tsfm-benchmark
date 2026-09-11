"""LightGBM WITH-COVARIATES arm plus leave-one-channel-out ablations,
extending code/baselines/lightgbm_zeroshot_baseline.py (glucose-only, MAE reported in
report.md Sec 32/36). Reuses the same 6-channel set as Chronos-2/TimesFM's
own "with covariates" comparison (report.md Sec 1) and LSTM's covariate arm
(code/baselines/lstm_zeroshot_baseline_covariates.py): heart_rate, stress,
respiratory_rate, oxygen_saturation, physical_activity_steps,
physical_activity_calorie_rate.

Training data: code/model_input/build_participant_grid.py's build_grid()
(same source LSTM's covariate arm effectively draws on via analysis_dataframe/
<pid>.csv -- confirmed column names match COVARIATE_CHANNELS exactly).
Covariate features per channel: 1 lag (last real value) + rolling mean/std
over the trailing 12 bins (1h) -- kept smaller than glucose's own 9-lag/
4-rolling-window feature set since 6 channels x a full feature set would
explode dimensionality for comparatively little expected signal (this
project's own LOO findings for Chronos-2/TimesFM already show most channels
have no effect at all -- report.md Sec 2). Values are ffilled/bfilled within
each participant's own series before feature construction, falling back to
0.0 if unresolvable -- same imputation convention as chronos2_zeroshot_eval.py
and lstm_zeroshot_baseline_covariates.py, for apples-to-apples comparability.

Only ONE feature-building pass is done (all 6 covariates' features built
once per row); the 6 leave-one-out refits reuse the same feature table with
the relevant channel's 3 columns dropped, rather than rebuilding features
from scratch per ablation -- this is the expensive part (reading every
participant's raw wearable file) and should not be repeated 7 times.
"""
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

REPO_ROOT = "/path/to/aireadi-collab"
sys.path.insert(0, os.path.join(REPO_ROOT, "code/model_input"))
# Sibling modules live in adjacent topic directories under code/.
_CODE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _d in ("analysis", "baselines", "cgmlsm", "chronos2", "meal_detection", "model_input", "ssmcgm", "timesfm"):
    sys.path.insert(0, os.path.join(_CODE, _d))
from build_participant_grid import load_cleaned_glucose_binned, build_grid, load_participants_covariates  # noqa: E402

WINDOW_INDEX = f"{REPO_ROOT}/model_input/window_index.csv"
ZEROSHOT_EVAL_DIR = f"{REPO_ROOT}/model_input/zeroshot_eval"
CHRONOS2_DIR = f"{REPO_ROOT}/chronos2_zeroshot_results/full/imputed"
OUTPUT_DIR = f"{REPO_ROOT}/lightgbm_zeroshot_covariates_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

N_TRAIN_PARTICIPANTS = int(os.environ.get("LGBM_N_TRAIN_PARTICIPANTS", "300"))
TRAIN_STRIDE_BINS = int(os.environ.get("LGBM_TRAIN_STRIDE_BINS", "6"))
SEED = 42
MAX_STEPS_AHEAD = 48
LAGS = [1, 2, 3, 6, 12, 24, 36, 72, 144]
ROLL_WINDOWS = [6, 12, 36, 144]
HORIZONS = {"30min": 6, "1h": 12, "2h": 24}
COVARIATE_CHANNELS = ["heart_rate", "stress", "respiratory_rate", "oxygen_saturation",
                       "physical_activity_steps", "physical_activity_calorie_rate"]
COV_LAG = 12  # 1h trailing lag/rolling window for each covariate channel


def build_features_from_series(vals, t_idx, steps_ahead, cov_series=None):
    if t_idx < 0 or t_idx >= len(vals):
        return None
    feats = {"steps_ahead": steps_ahead}
    for lag in LAGS:
        i = t_idx - lag + 1
        if i < 0:
            return None
        v = vals[i]
        if np.isnan(v):
            return None
        feats[f"lag_{lag}"] = v
    for w in ROLL_WINDOWS:
        i0 = t_idx - w + 1
        if i0 < 0:
            return None
        window = vals[i0:t_idx + 1]
        if np.isnan(window).any():
            return None
        feats[f"roll_mean_{w}"] = float(window.mean())
        feats[f"roll_std_{w}"] = float(window.std())
    if cov_series is not None:
        i0 = t_idx - COV_LAG + 1
        if i0 < 0:
            return None
        for ch, cvals in cov_series.items():
            cwin = cvals[i0:t_idx + 1]
            if np.isnan(cwin).any():
                return None
            feats[f"cov_{ch}_lag1"] = cvals[t_idx]
            feats[f"cov_{ch}_rollmean"] = float(cwin.mean())
            feats[f"cov_{ch}_rollstd"] = float(cwin.std())
    return feats


def build_training_rows(pid, cov_by_pid):
    grid = build_grid(pid, cov_by_pid)
    if grid is None or len(grid) < 300:
        return []
    grid = grid.sort_values("bin_start").reset_index(drop=True)
    glu = grid["cgm_glucose"].to_numpy(dtype=np.float64)
    cov_series = {}
    for ch in COVARIATE_CHANNELS:
        s = grid[ch].ffill().bfill().fillna(0.0)
        cov_series[ch] = s.to_numpy(dtype=np.float64)
    rows = []
    min_lag_needed = max(LAGS + ROLL_WINDOWS + [COV_LAG])
    for t_idx in range(min_lag_needed, len(glu) - MAX_STEPS_AHEAD, TRAIN_STRIDE_BINS):
        for steps_ahead in (1, 3, 6, 9, 12, 18, 24, 36, 48):
            target_idx = t_idx + steps_ahead
            if target_idx >= len(glu):
                continue
            y = glu[target_idx]
            if np.isnan(y):
                continue
            feats = build_features_from_series(glu, t_idx, steps_ahead, cov_series)
            if feats is None:
                continue
            feats["y"] = y
            rows.append(feats)
    return rows


def score_arm(model, feature_cols, label, chronos2_cache):
    """Score one fitted model on the with_covariates test windows; return
    summary rows and significance rows (vs persistence, vs Chronos-2
    with-covariates, vs the LightGBM full-covariates arm if label != 'full')."""
    summary_rows, sig_rows = [], []
    for horizon_name, horizon_bins in HORIZONS.items():
        path = f"{ZEROSHOT_EVAL_DIR}/zeroshot_{horizon_name}_with_covariates.csv"
        df = pd.read_csv(path, dtype={"window_id": str, "person_id": str})
        df["bin_start"] = pd.to_datetime(df["bin_start"], utc=True)
        df["is_context"] = df["is_context"].astype(bool)

        pred_rows = []
        for wid, g in df.groupby("window_id", sort=False):
            g = g.sort_values("bin_start")
            ctx = g[g["is_context"]]
            hor = g[~g["is_context"]]
            if ctx.empty or hor.empty:
                continue
            glu_ctx = ctx["cgm_glucose"].to_numpy(dtype=np.float64)
            cov_ctx = {ch: ctx[ch].ffill().bfill().fillna(0.0).to_numpy(dtype=np.float64)
                       for ch in COVARIATE_CHANNELS}
            t_idx = len(glu_ctx) - 1
            anchor_time = ctx["bin_start"].iloc[-1]
            for _, row in hor.iterrows():
                steps_ahead = int(round((row["bin_start"] - anchor_time).total_seconds() / 300))
                feats = build_features_from_series(glu_ctx, t_idx, steps_ahead, cov_ctx)
                if feats is None:
                    continue
                pred_rows.append({"window_id": wid, "person_id": g["person_id"].iloc[0],
                                   "bin_start": row["bin_start"], "actual": row["cgm_glucose"], **feats})
        pred_df = pd.DataFrame(pred_rows)
        if pred_df.empty:
            print(f"  [{label}] {horizon_name}: NO SCORABLE ROWS", file=sys.stderr)
            continue
        pred_df["predictions"] = model.predict(pred_df[feature_cols])
        pred_df["abs_error"] = (pred_df["predictions"] - pred_df["actual"]).abs()
        pred_df.drop(columns=feature_cols).to_csv(f"{OUTPUT_DIR}/predictions_{label}_{horizon_name}.csv", index=False)

        scored = pred_df.dropna(subset=["actual"])
        mae = scored["abs_error"].mean()
        rmse = float(np.sqrt((scored["abs_error"] ** 2).mean()))
        summary_rows.append({"arm": label, "horizon": horizon_name, "mae": round(float(mae), 4),
                              "rmse": round(rmse, 4), "n_bins": len(scored),
                              "n_participants": scored["person_id"].nunique()})
        print(f"  [{label}] {horizon_name}: n={len(scored)}, MAE={mae:.4f}", file=sys.stderr)

        # persistence, recomputed inline (same definition used elsewhere)
        persist_rows = []
        for wid, g in df.groupby("window_id", sort=False):
            g = g.sort_values("bin_start")
            ctx = g[g["is_context"]]
            valid = ctx["cgm_glucose"].dropna()
            if valid.empty:
                continue
            anchor = valid.iloc[-1]
            for _, row in g[~g["is_context"]].iterrows():
                persist_rows.append({"window_id": wid, "bin_start": row["bin_start"], "persistence": anchor})
        persist_df = pd.DataFrame(persist_rows)
        persist_df["bin_start"] = pd.to_datetime(persist_df["bin_start"], utc=True)

        c2_path = f"{CHRONOS2_DIR}/predictions_{horizon_name}_with_covariates.csv"
        if os.path.exists(c2_path):
            c2 = pd.read_csv(c2_path, dtype={"window_id": str})
            c2["bin_start"] = pd.to_datetime(c2["bin_start"], utc=True)
        else:
            c2 = None

        m = pred_df.merge(persist_df, on=["window_id", "bin_start"], how="inner")
        m["lgbm_err"] = (m["predictions"] - m["actual"]).abs()
        m["persist_err"] = (m["persistence"] - m["actual"]).abs()
        refs = [("persistence", "persist_err")]
        if c2 is not None:
            m = m.merge(c2[["window_id", "bin_start", "predictions"]].rename(columns={"predictions": "chronos2_pred"}),
                        on=["window_id", "bin_start"], how="inner")
            m["c2_err"] = (m["chronos2_pred"] - m["actual"]).abs()
            refs.append(("chronos2_with_covariates", "c2_err"))
        m = m.dropna(subset=["actual"])
        for ref_name, ref_col in refs:
            d = m["lgbm_err"] - m[ref_col]
            per_person = d.groupby(m["person_id"]).mean()
            t_stat, p_val = stats.ttest_1samp(per_person.to_numpy(), 0.0)
            sig_rows.append({"arm": label, "horizon": horizon_name, "vs": ref_name,
                              "n_matched_bins": len(m), "n_participants_clustered": len(per_person),
                              "lgbm_mae_matched": round(float(m["lgbm_err"].mean()), 4),
                              "ref_mae_matched": round(float(m[ref_col].mean()), 4),
                              "cluster_t": round(float(t_stat), 4), "cluster_p": float(p_val)})
    return summary_rows, sig_rows


def main():
    print("loading window_index.csv for train population...", file=sys.stderr)
    widx = pd.read_csv(WINDOW_INDEX, dtype={"person_id": str})
    train_pids_all = sorted(widx.loc[widx["recommended_split"] == "train", "person_id"].unique())
    test_pids_all = set(widx.loc[widx["recommended_split"] == "test", "person_id"].unique())
    assert not (set(train_pids_all) & test_pids_all), "train/test participant overlap -- leakage"

    rng = np.random.default_rng(SEED)
    sample_pids = rng.choice(train_pids_all, size=min(N_TRAIN_PARTICIPANTS, len(train_pids_all)), replace=False)
    cov_by_pid = load_participants_covariates()

    print(f"building training rows (glucose + {len(COVARIATE_CHANNELS)} covariates) "
          f"from {len(sample_pids)} sampled train participants...", file=sys.stderr)
    all_rows = []
    for i, pid in enumerate(sample_pids):
        all_rows.extend(build_training_rows(pid, cov_by_pid))
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(sample_pids)} participants, {len(all_rows)} rows so far", file=sys.stderr)
    train_df = pd.DataFrame(all_rows)
    print(f"total training rows: {len(train_df)}", file=sys.stderr)

    import lightgbm as lgb
    all_cols = [c for c in train_df.columns if c != "y"]

    all_summary, all_sig = [], []

    # -- full covariate model --
    print("fitting FULL covariate model...", file=sys.stderr)
    model_full = lgb.LGBMRegressor(n_estimators=300, learning_rate=0.05, max_depth=6,
                                    num_leaves=31, random_state=SEED, n_jobs=4, verbosity=-1)
    model_full.fit(train_df[all_cols], train_df["y"])
    model_full.booster_.save_model(f"{OUTPUT_DIR}/lgbm_full.txt")
    s, g = score_arm(model_full, all_cols, "full", None)
    all_summary += s
    all_sig += g

    # -- LOO ablations: drop each channel's 3 feature columns, refit on remaining --
    for ch in COVARIATE_CHANNELS:
        drop_cols = [f"cov_{ch}_lag1", f"cov_{ch}_rollmean", f"cov_{ch}_rollstd"]
        loo_cols = [c for c in all_cols if c not in drop_cols]
        print(f"fitting LOO drop_{ch} model...", file=sys.stderr)
        model_loo = lgb.LGBMRegressor(n_estimators=300, learning_rate=0.05, max_depth=6,
                                       num_leaves=31, random_state=SEED, n_jobs=4, verbosity=-1)
        model_loo.fit(train_df[loo_cols], train_df["y"])
        s, g = score_arm(model_loo, loo_cols, f"drop_{ch}", None)
        all_summary += s
        all_sig += g

    # -- LOO vs FULL significance (the actual ablation question) --
    ablation_rows = []
    for ch in COVARIATE_CHANNELS:
        for horizon_name in HORIZONS:
            full_pred = pd.read_csv(f"{OUTPUT_DIR}/predictions_full_{horizon_name}.csv",
                                     dtype={"window_id": str, "person_id": str})
            loo_pred = pd.read_csv(f"{OUTPUT_DIR}/predictions_drop_{ch}_{horizon_name}.csv",
                                    dtype={"window_id": str, "person_id": str})
            full_pred["bin_start"] = pd.to_datetime(full_pred["bin_start"], utc=True)
            loo_pred["bin_start"] = pd.to_datetime(loo_pred["bin_start"], utc=True)
            m = full_pred.merge(loo_pred, on=["window_id", "bin_start", "person_id"],
                                 suffixes=("_full", "_loo"))
            m = m.dropna(subset=["actual_full"])
            m["err_full"] = (m["predictions_full"] - m["actual_full"]).abs()
            m["err_loo"] = (m["predictions_loo"] - m["actual_full"]).abs()
            d = (m["err_loo"] - m["err_full"]).groupby(m["person_id"]).mean()  # >0 => dropping ch hurt => helpful
            t_stat, p_val = stats.ttest_1samp(d.to_numpy(), 0.0)
            ablation_rows.append({
                "channel": ch, "horizon": horizon_name, "n_matched_bins": len(m),
                "n_participants_clustered": len(d), "mae_full": round(float(m["err_full"].mean()), 4),
                "mae_drop": round(float(m["err_loo"].mean()), 4), "cluster_t": round(float(t_stat), 4),
                "cluster_p": float(p_val),
                "direction": "helpful" if float(d.mean()) > 0 else "harmful",
                "significant": bool(p_val < 0.05),
            })

    pd.DataFrame(all_summary).to_csv(f"{OUTPUT_DIR}/summary.csv", index=False)
    pd.DataFrame(all_sig).to_csv(f"{OUTPUT_DIR}/significance_vs_baselines.csv", index=False)
    ablation_df = pd.DataFrame(ablation_rows)
    ablation_df.to_csv(f"{OUTPUT_DIR}/loo_ablation.csv", index=False)

    # -- gain-based feature importance from the full model, secondary signal only --
    importance = pd.DataFrame({
        "feature": all_cols, "gain": model_full.booster_.feature_importance(importance_type="gain"),
    }).sort_values("gain", ascending=False)
    importance.to_csv(f"{OUTPUT_DIR}/feature_importance_full_model.csv", index=False)

    print("\n=== LOO ablation (drop channel vs full model) ===", file=sys.stderr)
    print(ablation_df.to_string(index=False), file=sys.stderr)
    print("\n=== top 15 features by gain (full model) ===", file=sys.stderr)
    print(importance.head(15).to_string(index=False), file=sys.stderr)
    print("LIGHTGBM_COVARIATES_LOO_OK", file=sys.stderr)


if __name__ == "__main__":
    main()
