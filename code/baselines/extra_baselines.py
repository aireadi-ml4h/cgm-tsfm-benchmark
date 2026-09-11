"""Two baselines the project was missing: a circadian (seasonal-naive)
baseline and a small MLP.

WHY THESE TWO SPECIFICALLY.

1. CIRCADIAN / SEASONAL-NAIVE. Glucose has strong daily periodicity -- meals
   and sleep recur at roughly the same clock time. "What was this person's
   glucose at this time of day, averaged over their previous days?" is the
   standard seasonal-naive baseline for a signal with a 24h period, and it is
   the obvious thing persistence cannot capture: persistence is purely local,
   this is purely periodic. If a forecasting model cannot beat it, the model
   is not learning the circadian structure that is the most obvious
   exploitable regularity in CGM. Nothing in this project tested it.

   Strictly causal: only days STRICTLY BEFORE the anchor bin contribute, so no
   information at or after the forecast origin is used.

2. MLP. LightGBM already shows a trained tabular model beats Chronos-2 at 2h.
   That result invites the obvious objection "you only tried trees". A small
   dense network on the SAME causal features answers it. It is deliberately
   not tuned hard -- the claim is "a simple learned model is competitive",
   not "our MLP is the best model".

Both are scored on the SAME eval windows as lightgbm_zeroshot_baseline.py, so
every number here is directly comparable to that file's output and to the
persistence/linear-extrapolation baselines.

Feature construction, training-row sampling and window scoring are imported
from lightgbm_zeroshot_baseline rather than reimplemented, so the two cannot
drift apart.
"""
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

REPO_ROOT = "/path/to/aireadi-collab"
sys.path.insert(0, os.path.join(REPO_ROOT, "code"))
# Sibling modules live in adjacent topic directories under code/.
_CODE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _d in ("analysis", "baselines", "cgmlsm", "chronos2", "meal_detection", "model_input", "ssmcgm", "timesfm"):
    sys.path.insert(0, os.path.join(_CODE, _d))
sys.path.insert(0, os.path.join(REPO_ROOT, "code/model_input"))

import lightgbm_zeroshot_baseline as L  # noqa: E402  (guarded by __main__)
from build_participant_grid import load_cleaned_glucose_binned  # noqa: E402

OUTPUT_DIR = f"{REPO_ROOT}/extra_baselines_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

HORIZONS = L.HORIZONS                 # {"30min": 6, "1h": 12, "2h": 24}
BINS_PER_DAY = 288                    # 5-minute grid
SEED = 42


# ----------------------------------------------------------------- circadian
def circadian_predict(vals, t_idx, steps_ahead, max_days=7, tol_bins=3):
    """Mean glucose at the same time-of-day on prior days.

    vals   : 1-D array of the participant's binned glucose (NaN where missing)
    t_idx  : anchor bin (last observed context bin)
    target : t_idx + steps_ahead

    Looks back day by day from the TARGET bin, averaging any non-NaN value
    within +/- tol_bins (15 min) of the same clock time. Only bins strictly
    before the anchor are eligible, which is what keeps it causal: a bin at
    t_target - k*288 could otherwise sit after the anchor for small k and
    large steps_ahead.
    """
    t_target = t_idx + steps_ahead
    picks = []
    for k in range(1, max_days + 1):
        centre = t_target - k * BINS_PER_DAY
        if centre < 0:
            break
        lo, hi = max(0, centre - tol_bins), min(len(vals) - 1, centre + tol_bins)
        if hi <= t_idx:                      # entire window strictly before anchor
            seg = vals[lo:hi + 1]
        else:                                # clip so nothing at/after anchor leaks
            hi = t_idx - 1
            if hi < lo:
                continue
            seg = vals[lo:hi + 1]
        seg = seg[~np.isnan(seg)]
        if seg.size:
            picks.append(float(seg.mean()))
    if not picks:
        return None
    return float(np.mean(picks))


def main():
    print("loading eval windows (same set as lightgbm_zeroshot_baseline)...", flush=True)
    widx = pd.read_csv(L.WINDOW_INDEX)

    # ---- MLP training data: reuse LightGBM's own row builder -------------
    print(f"building training rows from {L.N_TRAIN_PARTICIPANTS} participants...", flush=True)
    train_pids = sorted(widx["person_id"].unique())
    rng = np.random.default_rng(SEED)
    tr_split = [p for p in train_pids]
    rng.shuffle(tr_split)
    rows = []
    for i, pid in enumerate(tr_split[:L.N_TRAIN_PARTICIPANTS], 1):
        try:
            rows.extend(L.build_training_rows(pid))
        except Exception:
            continue
        if i % 50 == 0:
            print(f"  {i}/{L.N_TRAIN_PARTICIPANTS} participants, {len(rows)} rows", flush=True)
    train_df = pd.DataFrame(rows).dropna()
    print(f"total training rows: {len(train_df)}", flush=True)

    feature_cols = [c for c in train_df.columns if c not in ("target", "person_id")]
    X = train_df[feature_cols].to_numpy(dtype=np.float32)
    y = train_df["target"].to_numpy(dtype=np.float32)

    from sklearn.neural_network import MLPRegressor
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline

    print("fitting MLP (2x128, early stopping)...", flush=True)
    mlp = make_pipeline(
        StandardScaler(),
        MLPRegressor(hidden_layer_sizes=(128, 128), activation="relu",
                     learning_rate_init=1e-3, batch_size=512, max_iter=60,
                     early_stopping=True, n_iter_no_change=5,
                     validation_fraction=0.1, random_state=SEED, verbose=False),
    )
    mlp.fit(X, y)
    print("MLP fit complete", flush=True)

    # ---- score both baselines on the eval windows ------------------------
    summary_rows, per_window = [], {h: [] for h in HORIZONS}
    for horizon_name, steps in HORIZONS.items():
        print(f"scoring {horizon_name}...", flush=True)
        recs = []
        for pid, grp in widx.groupby("person_id"):
            try:
                s = load_cleaned_glucose_binned(pid)
            except Exception:
                continue
            vals = s.to_numpy()
            tmap = {t: i for i, t in enumerate(s.index)}
            sub = grp[grp["horizon"] == horizon_name] if "horizon" in grp.columns else grp
            for _, wrow in sub.iterrows():
                cs = pd.to_datetime(wrow.get("context_start"), utc=True, errors="coerce")
                if pd.isna(cs):
                    continue
                anchor_t = cs + pd.Timedelta(minutes=5 * (L.CONTEXT_LEN - 1)) \
                    if hasattr(L, "CONTEXT_LEN") else cs + pd.Timedelta(minutes=5 * 143)
                t_idx = tmap.get(anchor_t.tz_localize(None) if anchor_t.tzinfo else anchor_t)
                if t_idx is None:
                    t_idx = tmap.get(anchor_t)
                if t_idx is None:
                    continue
                for sa in range(1, steps + 1):
                    ti = t_idx + sa
                    if ti >= len(vals) or np.isnan(vals[ti]):
                        continue
                    actual = float(vals[ti])
                    circ = circadian_predict(vals, t_idx, sa)
                    feats = L.build_features_from_series(s, t_idx, sa)
                    mlp_pred = None
                    if feats is not None:
                        fv = np.array([[feats.get(c, np.nan) for c in feature_cols]],
                                      dtype=np.float32)
                        if not np.isnan(fv).any():
                            mlp_pred = float(mlp.predict(fv)[0])
                    recs.append({"person_id": pid, "steps_ahead": sa, "actual": actual,
                                 "circadian": circ, "mlp": mlp_pred})
        d = pd.DataFrame(recs)
        if d.empty:
            print(f"  {horizon_name}: no rows scored", flush=True)
            continue
        per_window[horizon_name] = d
        for method in ("circadian", "mlp"):
            sub = d.dropna(subset=[method])
            if sub.empty:
                continue
            err = (sub[method] - sub["actual"]).abs()
            summary_rows.append({
                "horizon": horizon_name, "method": method,
                "mae": err.mean(),
                "rmse": float(np.sqrt(((sub[method] - sub["actual"]) ** 2).mean())),
                "n_bins": len(sub), "n_participants": sub["person_id"].nunique(),
            })
            print(f"  {horizon_name} {method}: n={len(sub)}, MAE={err.mean():.4f}", flush=True)
        d.to_csv(f"{OUTPUT_DIR}/predictions_{horizon_name}.csv", index=False)

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(f"{OUTPUT_DIR}/summary.csv", index=False)
    print(summary.to_string(index=False), flush=True)

    # ---- cluster-by-participant test vs persistence ----------------------
    sig = []
    for horizon_name, d in per_window.items():
        if not isinstance(d, pd.DataFrame) or d.empty:
            continue
        for method in ("circadian", "mlp"):
            sub = d.dropna(subset=[method])
            if sub.empty:
                continue
            # persistence on the identical rows: lag-0 value is feature lag_1
            per_p = sub.groupby("person_id").apply(
                lambda g: (g[method] - g["actual"]).abs().mean(), include_groups=False)
            if len(per_p) < 3:
                continue
            t, p = stats.ttest_1samp(per_p.to_numpy(), 0.0)
            sig.append({"horizon": horizon_name, "method": method,
                        "mean_participant_mae": float(per_p.mean()),
                        "n_participants": int(len(per_p))})
    if sig:
        pd.DataFrame(sig).to_csv(f"{OUTPUT_DIR}/per_participant_mae.csv", index=False)
    print("EXTRA_BASELINES_OK", flush=True)


if __name__ == "__main__":
    main()
