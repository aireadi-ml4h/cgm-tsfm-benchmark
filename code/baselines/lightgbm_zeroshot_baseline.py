"""LightGBM zero-shot glucose forecasting baseline, matching the original
SSM-CGM paper's own "Quick Comparisons" (Appendix A.3), which itself
benchmarks against LightGBM (MAE=9.70 mg/dL, their population/split).

Trained from scratch on a sample of the official AI-READI TRAIN split
(never touching test participants), scored on the EXACT SAME test windows
already used for Chronos-2/TimesFM's zero-shot glucose-only comparison
(model_input/zeroshot_eval/zeroshot_{h}_glucose_only.csv), so MAE is
directly comparable via matched window_id/bin_start.

Design: one model per horizon family is unnecessary -- a single LightGBM
regressor is trained with "steps_ahead" (bins into the future) as an
explicit feature, using training examples built from a sliding window over
each sampled train participant's own real glucose series
(final_cleaned_data/glucose/<pid>.csv via load_cleaned_glucose_binned,
the same real, un-imputed 5-min series every other model in this project
uses -- cgm_glucose is NEVER imputed here, matching Appendix B of the
methods draft). This lets one model serve 30min/1h/2h horizons.

Features, causal only (built from data at or before the context/anchor
time, matching every other model's causal boundary in this project):
lags at 1,2,3,6,12,24,36,72,144 bins (5,10,15,30,60,120,180,360,720 min),
rolling mean/std over trailing 6/12/36/144 bins, time-of-day sin/cos, and
steps_ahead itself. Rows with any NaN feature or NaN target are dropped
(no imputation anywhere, matching this project's established convention).
"""
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

REPO_ROOT = os.environ["AIREADI_ROOT"]
sys.path.insert(0, os.path.join(REPO_ROOT, "code/model_input"))
# Sibling modules live in adjacent topic directories under code/.
_CODE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _d in ("analysis", "baselines", "cgmlsm", "chronos2", "meal_detection", "model_input", "ssmcgm", "timesfm"):
    sys.path.insert(0, os.path.join(_CODE, _d))
from build_participant_grid import load_cleaned_glucose_binned  # noqa: E402

WINDOW_INDEX = f"{REPO_ROOT}/model_input/window_index.csv"
ZEROSHOT_EVAL_DIR = f"{REPO_ROOT}/model_input/zeroshot_eval"
CHRONOS2_DIR = f"{REPO_ROOT}/chronos2_zeroshot_results/full/imputed"
# LGBM_OBJECTIVE / LGBM_OUTPUT_SUFFIX: added 2026-09-14 to run an L1-objective
# control. The objective was originally never chosen -- LGBMRegressor's default
# is `regression` (L2), while the LSTM and Mamba baselines explicitly use
# nn.L1Loss() to match the MAE eval metric. That mismatch was a candidate
# explanation for LightGBM losing to Chronos-2 at 30 min but winning at 2 h, so
# it needed isolating rather than assuming. Everything else -- features, seed,
# n_estimators, learning rate, depth, leaves, training participants, stride --
# is held identical, so the only difference between the two runs is the loss.
# The paper reports the L1 model, so `regression_l1` is the default here. For
# the original squared-error model, set LGBM_OBJECTIVE=regression (and e.g.
# LGBM_OUTPUT_SUFFIX=_l2 so it doesn't overwrite the default output).
OBJECTIVE = os.environ.get("LGBM_OBJECTIVE", "regression_l1")
OUTPUT_SUFFIX = os.environ.get("LGBM_OUTPUT_SUFFIX", "")
OUTPUT_DIR = f"{REPO_ROOT}/lightgbm_zeroshot_results{OUTPUT_SUFFIX}"
os.makedirs(OUTPUT_DIR, exist_ok=True)

N_TRAIN_PARTICIPANTS = int(os.environ.get("LGBM_N_TRAIN_PARTICIPANTS", "300"))
TRAIN_STRIDE_BINS = int(os.environ.get("LGBM_TRAIN_STRIDE_BINS", "6"))  # 30 min
# SAMPLE_SEED / TRAIN_SEED: added 2026-10-07 for the seed-stability check
# (code/seed_stability/). SAMPLE_SEED drives the participant sample; TRAIN_SEED
# is passed to random_state, which has NO effect here because no subsampling,
# bagging (subsample/subsample_freq) or feature fraction (colsample_bytree) is
# set -- LightGBM is deterministic given its data, so only the arm-B
# ("full resample") variant is run for it. Both default to 42, which reproduces
# the original run exactly.
SAMPLE_SEED = int(os.environ.get("SAMPLE_SEED", "42"))
TRAIN_SEED = int(os.environ.get("TRAIN_SEED", "42"))
SEED = SAMPLE_SEED  # kept for importers (extra_baselines_v2.py does not use it)
sys.path.insert(0, os.path.join(_CODE, "seed_stability"))
from seed_manifest import seed_env_set, write_seed_manifest  # noqa: E402
WRITE_SEED_MANIFEST = seed_env_set()

# TRAIN_SAMPLE_MODE (2026-11-03, FMTS training-size curve): participant sampling
# lives in code/baselines/train_sampling.py, shared by the trained baselines. The
# default (random) is the original rng.choice call, so existing runs are unchanged.
from train_sampling import TRAIN_SAMPLE_MODE, sample_train_participants  # noqa: E402,F401

MAX_STEPS_AHEAD = 48  # covers 30min(6)/1h(12)/2h(24), leaves headroom to 4h(48)
LAGS = [1, 2, 3, 6, 12, 24, 36, 72, 144]
ROLL_WINDOWS = [6, 12, 36, 144]
HORIZONS = {"30min": 6, "1h": 12, "2h": 24}


def build_features_from_series(s, t_idx, steps_ahead):
    """s: pd.Series indexed by real time bins (may have NaN gaps), 5-min freq.
    t_idx: integer position of the anchor bin (last real context bin).
    Returns a dict of causal features, or None if any required lag/rolling
    value is unavailable (NaN) -- never imputed."""
    vals = s.to_numpy()
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
    ts = s.index[t_idx]
    tod = ts.hour * 60 + ts.minute
    feats["tod_sin"] = np.sin(2 * np.pi * tod / 1440)
    feats["tod_cos"] = np.cos(2 * np.pi * tod / 1440)
    return feats


def build_training_rows(pid):
    s = load_cleaned_glucose_binned(pid)
    if s is None or len(s) < 300:
        return []
    vals = s.to_numpy()
    rows = []
    min_lag_needed = max(LAGS + ROLL_WINDOWS)
    for t_idx in range(min_lag_needed, len(vals) - MAX_STEPS_AHEAD, TRAIN_STRIDE_BINS):
        for steps_ahead in (1, 3, 6, 9, 12, 18, 24, 36, 48):  # sparse sample of horizons for training speed
            target_idx = t_idx + steps_ahead
            if target_idx >= len(vals):
                continue
            y = vals[target_idx]
            if np.isnan(y):
                continue
            feats = build_features_from_series(s, t_idx, steps_ahead)
            if feats is None:
                continue
            feats["y"] = y
            rows.append(feats)
    return rows


def main():
    print("loading window_index.csv for train population...", file=sys.stderr)
    widx = pd.read_csv(WINDOW_INDEX, dtype={"person_id": str})
    train_pids_all = sorted(widx.loc[widx["recommended_split"] == "train", "person_id"].unique())
    test_pids_all = set(widx.loc[widx["recommended_split"] == "test", "person_id"].unique())
    print(f"train pool: {len(train_pids_all)} participants, test pool: {len(test_pids_all)}", file=sys.stderr)
    assert not (set(train_pids_all) & test_pids_all), "train/test participant overlap -- leakage"

    sample_pids = sample_train_participants(train_pids_all, widx, N_TRAIN_PARTICIPANTS, SAMPLE_SEED)
    if WRITE_SEED_MANIFEST:
        print(f"SAMPLE_SEED={SAMPLE_SEED} TRAIN_SEED={TRAIN_SEED}", file=sys.stderr)
        write_seed_manifest(OUTPUT_DIR, SAMPLE_SEED, TRAIN_SEED, sample_pids)

    print(f"building training rows from {len(sample_pids)} sampled train participants...", file=sys.stderr)
    all_rows = []
    for i, pid in enumerate(sample_pids):
        all_rows.extend(build_training_rows(pid))
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(sample_pids)} participants, {len(all_rows)} rows so far", file=sys.stderr)
    train_df = pd.DataFrame(all_rows)
    print(f"total training rows: {len(train_df)}", file=sys.stderr)
    train_df.to_csv(f"{OUTPUT_DIR}/train_rows_sample.csv.gz", index=False, compression="gzip")

    import lightgbm as lgb
    feature_cols = [c for c in train_df.columns if c != "y"]
    model = lgb.LGBMRegressor(objective=OBJECTIVE,
                               n_estimators=300, learning_rate=0.05, max_depth=6,
                               num_leaves=31, random_state=TRAIN_SEED, n_jobs=4, verbosity=-1)
    print(f"LightGBM objective: {OBJECTIVE}", file=sys.stderr)
    print("fitting LightGBM...", file=sys.stderr)
    model.fit(train_df[feature_cols], train_df["y"])
    model.booster_.save_model(f"{OUTPUT_DIR}/lgbm_model.txt")
    print("fit complete", file=sys.stderr)

    # -- Score on the exact same test windows used for Chronos-2/TimesFM --
    summary_rows = []
    sig_rows = []
    for horizon_name, horizon_bins in HORIZONS.items():
        print(f"scoring {horizon_name}...", file=sys.stderr)
        path = f"{ZEROSHOT_EVAL_DIR}/zeroshot_{horizon_name}_glucose_only.csv"
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
            s = pd.Series(ctx["cgm_glucose"].to_numpy(), index=pd.DatetimeIndex(ctx["bin_start"]))
            t_idx = len(s) - 1  # last context bin, 0-indexed within this window's own context array
            for _, row in hor.iterrows():
                steps_ahead = int(round((row["bin_start"] - s.index[-1]).total_seconds() / 300))
                feats = build_features_from_series(s, t_idx, steps_ahead)
                if feats is None:
                    continue
                pred_rows.append({
                    "window_id": wid, "person_id": g["person_id"].iloc[0],
                    "bin_start": row["bin_start"], "actual": row["cgm_glucose"],
                    **feats,
                })
        pred_df = pd.DataFrame(pred_rows)
        if pred_df.empty:
            print(f"  {horizon_name}: NO SCORABLE ROWS (feature construction failed for all windows)", file=sys.stderr)
            continue
        pred_df["predictions"] = model.predict(pred_df[feature_cols])
        pred_df["abs_error"] = (pred_df["predictions"] - pred_df["actual"]).abs()
        pred_df.drop(columns=feature_cols).to_csv(f"{OUTPUT_DIR}/predictions_{horizon_name}.csv", index=False)

        scored = pred_df.dropna(subset=["actual"])
        mae = scored["abs_error"].mean()
        rmse = float(np.sqrt((scored["abs_error"] ** 2).mean()))
        n_participants = scored["person_id"].nunique()
        summary_rows.append({"horizon": horizon_name, "method": "lightgbm", "mae": round(float(mae), 4),
                              "rmse": round(rmse, 4), "n_bins": len(scored), "n_participants": n_participants})
        print(f"  {horizon_name}: n={len(scored)}, participants={n_participants}, MAE={mae:.4f}", file=sys.stderr)

        # -- significance vs persistence and vs Chronos-2, matched windows, cluster-by-participant --
        naive_path = f"{OUTPUT_DIR}/../ssmcgm_train_results"  # placeholder unused; persistence recomputed inline below
        # Persistence recomputed directly here (same definition as naive_baselines_chronos_timesfm.py):
        # last real context value, per window.
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

        c2_path = f"{CHRONOS2_DIR}/predictions_{horizon_name}_glucose_only.csv"
        c2 = pd.read_csv(c2_path, dtype={"window_id": str})
        c2["bin_start"] = pd.to_datetime(c2["bin_start"], utc=True)

        m = pred_df.merge(persist_df, on=["window_id", "bin_start"], how="inner")
        m = m.merge(c2[["window_id", "bin_start", "predictions"]].rename(columns={"predictions": "chronos2_pred"}),
                    on=["window_id", "bin_start"], how="inner")
        m = m.dropna(subset=["actual"])
        m["lgbm_err"] = (m["predictions"] - m["actual"]).abs()
        m["persist_err"] = (m["persistence"] - m["actual"]).abs()
        m["c2_err"] = (m["chronos2_pred"] - m["actual"]).abs()

        for ref_name, ref_col in [("persistence", "persist_err"), ("chronos2", "c2_err")]:
            d = m["lgbm_err"] - m[ref_col]
            per_person = d.groupby(m["person_id"]).mean()
            t_stat, p_val = stats.ttest_1samp(per_person.to_numpy(), 0.0)
            sig_rows.append({
                "horizon": horizon_name, "vs": ref_name, "n_matched_bins": len(m),
                "n_participants_clustered": len(per_person),
                "lgbm_mae_matched": round(float(m["lgbm_err"].mean()), 4),
                "ref_mae_matched": round(float(m[ref_col].mean()), 4),
                "cluster_t": round(float(t_stat), 4), "cluster_p": float(p_val),
                "lgbm_better": bool(m["lgbm_err"].mean() < m[ref_col].mean()),
            })

    pd.DataFrame(summary_rows).to_csv(f"{OUTPUT_DIR}/summary.csv", index=False)
    sig_df = pd.DataFrame(sig_rows)
    sig_df.to_csv(f"{OUTPUT_DIR}/significance.csv", index=False)
    print(sig_df.to_string(index=False), file=sys.stderr)
    print("LIGHTGBM_ZEROSHOT_BASELINE_OK", file=sys.stderr)


if __name__ == "__main__":
    main()
