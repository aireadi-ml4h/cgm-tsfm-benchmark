"""Circadian (seasonal-naive) and MLP baselines, on the shared eval windows.

Supersedes an earlier extra_baselines.py (not included in this repository),
which FAILED (job 42632271) and had two real defects. Both are fixed here and
both are worth recording:

  1. LEAKAGE (serious). The old version drew training participants from
     `widx["person_id"].unique()` -- every participant, including the TEST
     split. `lightgbm_zeroshot_baseline.py` correctly restricts to
     `recommended_split == "train"` and asserts the intersection with test is
     empty. That assert is reproduced here. Any number the old script had
     produced would have been contaminated.
  2. KeyError 'target'. `build_training_rows()` names the target column `y`,
     not `target`.

WHY THESE TWO BASELINES.

CIRCADIAN / SEASONAL-NAIVE. Glucose has strong daily periodicity -- meals and
sleep recur at roughly the same clock time. Persistence is purely local and
cannot express that; this baseline is purely periodic and cannot express local
level. If a forecasting model cannot beat the better of the two, it is not
learning anything a trivial method does not already have. Nothing in this
project tested it before.

MLP. LightGBM already beats Chronos-2 at 2h, which invites the objection "you
only tried trees". A dense net on the SAME causal features answers it. It is
deliberately lightly tuned: the claim is "a simple learned model is
competitive", not "this MLP is optimal".

CAUSALITY. The circadian lookback only uses bins strictly before the window's
anchor (the last real context bin), so nothing at or after the forecast origin
is used. Feature construction is `build_features_from_series` unchanged from
the LightGBM script, which is already causal by construction.

COMPARABILITY. Scoring iterates the same
`model_input/zeroshot_eval/zeroshot_{h}_glucose_only.csv` files, grouped the
same way, as `lightgbm_zeroshot_baseline.py`. So these numbers sit directly
alongside LightGBM, persistence, linear extrapolation, Chronos-2 and TimesFM.
"""
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

REPO_ROOT = os.environ["AIREADI_ROOT"]
sys.path.insert(0, os.path.join(REPO_ROOT, "code"))
# Sibling modules live in adjacent topic directories under code/.
_CODE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _d in ("analysis", "baselines", "cgmlsm", "chronos2", "meal_detection", "model_input", "ssmcgm", "timesfm"):
    sys.path.insert(0, os.path.join(_CODE, _d))
sys.path.insert(0, os.path.join(REPO_ROOT, "code/model_input"))

import lightgbm_zeroshot_baseline as L  # noqa: E402
from build_participant_grid import load_cleaned_glucose_binned  # noqa: E402

# MLP_IMPL / MLP_LOSS / MLP_OUTPUT_SUFFIX: added 2026-09-14 for the
# training-objective control. sklearn's MLPRegressor supports squared error
# ONLY -- there is no loss argument -- so testing an absolute-error objective
# needs a different implementation, not a different parameter. MLP_IMPL=torch
# swaps in a torch MLP mirroring MLPRegressor's architecture and optimiser,
# with MLP_LOSS selecting l2 or l1. Running torch/l2 as well as torch/l1 is
# what makes the comparison clean: torch-l2 vs the sklearn result checks the
# reimplementation is faithful, and torch-l1 vs torch-l2 then isolates the
# loss with the framework held fixed.
# The paper reports the torch/l1 model, so that is the default here. The
# original squared-error run is MLP_IMPL=sklearn MLP_LOSS=l2 (with e.g.
# MLP_OUTPUT_SUFFIX=_sklearn_l2 so it doesn't overwrite the default output).
MLP_IMPL = os.environ.get("MLP_IMPL", "torch")
MLP_LOSS = os.environ.get("MLP_LOSS", "l1")
# Fail loudly on a typo: otherwise any MLP_LOSS other than "l1" trains on
# squared error, and any MLP_IMPL other than "sklearn" selects torch.
if MLP_IMPL not in ("torch", "sklearn") or MLP_LOSS not in ("l1", "l2"):
    raise SystemExit(f"MLP_IMPL must be torch|sklearn and MLP_LOSS l1|l2; got {MLP_IMPL!r}, {MLP_LOSS!r}")
if MLP_IMPL == "sklearn" and MLP_LOSS != "l2":
    raise SystemExit("sklearn's MLPRegressor supports squared error only: set MLP_LOSS=l2 or MLP_IMPL=torch")
OUTPUT_DIR = f"{REPO_ROOT}/extra_baselines_results{os.environ.get('MLP_OUTPUT_SUFFIX','')}"
os.makedirs(OUTPUT_DIR, exist_ok=True)

HORIZONS = L.HORIZONS
BINS_PER_DAY = 288          # 5-minute grid
MAX_LOOKBACK_DAYS = 7
TOL_BINS = 3                # +/- 15 min around the same clock time
# SAMPLE_SEED / TRAIN_SEED: added 2026-10-07 for the seed-stability check
# (code/seed_stability/). SAMPLE_SEED drives the 300-participant sample and the
# TorchMLP's 10% early-stopping holdout; TRAIN_SEED drives torch.manual_seed
# only (weight init and mini-batch shuffle order). Both default to 42, which
# reproduces the original run exactly: previously TorchMLP seeded torch AND its
# holdout rng from one `seed`, and with both 42 the draws are identical.
# The sklearn path keeps a single random_state (init and holdout are not
# separable there) and uses SAMPLE_SEED; it is not part of the seed check.
SAMPLE_SEED = int(os.environ.get("SAMPLE_SEED", "42"))
TRAIN_SEED = int(os.environ.get("TRAIN_SEED", "42"))
SEED = SAMPLE_SEED


def circadian_predict(vals, t_idx, steps_ahead):
    """Mean glucose at the same clock time on prior days.

    vals : participant's full binned glucose (NaN where missing)
    t_idx: anchor bin index (last real context bin)

    Walks back whole days from the TARGET bin. Any candidate window is clipped
    to end before the anchor, so a bin at or after the forecast origin can
    never contribute -- which is the whole causality requirement here.
    """
    t_target = t_idx + steps_ahead
    picks = []
    for k in range(1, MAX_LOOKBACK_DAYS + 1):
        centre = t_target - k * BINS_PER_DAY
        if centre < 0:
            break
        lo = max(0, centre - TOL_BINS)
        hi = min(len(vals) - 1, centre + TOL_BINS, t_idx - 1)   # strictly pre-anchor
        if hi < lo:
            continue
        seg = vals[lo:hi + 1]
        seg = seg[~np.isnan(seg)]
        if seg.size:
            picks.append(float(seg.mean()))
    return float(np.mean(picks)) if picks else None


class TorchMLP:
    """MLPRegressor's architecture and optimiser, with a selectable loss.

    Mirrors sklearn MLPRegressor(hidden_layer_sizes=(128,128),
    activation="relu", solver="adam", learning_rate_init=1e-3,
    batch_size=512, max_iter=60, early_stopping=True, n_iter_no_change=5,
    validation_fraction=0.1) -- including the 10% early-stopping holdout and
    the patience rule, so that torch/l2 should land close to the sklearn run.
    Exposes fit/predict so it drops into the existing make_pipeline call
    unchanged.

    Early stopping is evaluated on the SELECTED loss, not always on squared
    error: stopping an L1 model on an L2 criterion would reintroduce the very
    mismatch this control exists to remove.
    """

    def __init__(self, loss="l2", seed=42, holdout_seed=42, hidden=(128, 128), lr=1e-3,
                 batch_size=512, max_epochs=60, patience=5, val_frac=0.1):
        self.loss, self.seed, self.hidden = loss, seed, hidden
        self.holdout_seed = holdout_seed
        self.lr, self.batch_size = lr, batch_size
        self.max_epochs, self.patience, self.val_frac = max_epochs, patience, val_frac

    def fit(self, X, y):
        import torch
        import torch.nn as nn

        torch.manual_seed(self.seed)
        rng = np.random.default_rng(self.holdout_seed)
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32).reshape(-1)

        idx = rng.permutation(len(X))
        n_val = max(1, int(round(self.val_frac * len(X))))
        va, tr = idx[:n_val], idx[n_val:]
        self.holdout_idx_, self.n_rows_ = va, len(X)

        dev = "cuda" if torch.cuda.is_available() else "cpu"
        layers, d = [], X.shape[1]
        for h in self.hidden:
            layers += [nn.Linear(d, h), nn.ReLU()]
            d = h
        layers.append(nn.Linear(d, 1))
        net = nn.Sequential(*layers).to(dev)

        crit = nn.L1Loss() if self.loss == "l1" else nn.MSELoss()
        opt = torch.optim.Adam(net.parameters(), lr=self.lr)

        Xtr = torch.tensor(X[tr], device=dev); ytr = torch.tensor(y[tr], device=dev)
        Xva = torch.tensor(X[va], device=dev); yva = torch.tensor(y[va], device=dev)

        best, best_state, bad = float("inf"), None, 0
        for epoch in range(self.max_epochs):
            net.train()
            perm = torch.randperm(len(Xtr), device=dev)
            for i in range(0, len(Xtr), self.batch_size):
                b = perm[i:i + self.batch_size]
                opt.zero_grad()
                crit(net(Xtr[b]).squeeze(-1), ytr[b]).backward()
                opt.step()
            net.eval()
            with torch.no_grad():
                vl = float(crit(net(Xva).squeeze(-1), yva))
            if vl < best - 1e-6:
                best, bad = vl, 0
                best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
            else:
                bad += 1
                if bad >= self.patience:
                    print(f"  early stop at epoch {epoch} (best val {self.loss} {best:.4f})",
                          file=sys.stderr)
                    break
        if best_state is not None:
            net.load_state_dict(best_state)
        self._net, self._dev = net, dev
        return self

    def predict(self, X):
        import torch
        self._net.eval()
        with torch.no_grad():
            t = torch.tensor(np.asarray(X, dtype=np.float32), device=self._dev)
            return self._net(t).squeeze(-1).cpu().numpy()


def main():
    print("loading window_index.csv...", flush=True)
    widx = pd.read_csv(L.WINDOW_INDEX, dtype={"person_id": str})
    train_pids_all = sorted(widx.loc[widx["recommended_split"] == "train", "person_id"].unique())
    test_pids_all = set(widx.loc[widx["recommended_split"] == "test", "person_id"].unique())
    # The assert the old version was missing.
    assert not (set(train_pids_all) & test_pids_all), "train/test participant overlap -- leakage"
    print(f"train pool {len(train_pids_all)}, test pool {len(test_pids_all)}", flush=True)

    # L.sample_train_participants: mode "random" (default) is the original
    # rng.choice call with the same seed, so existing runs are unchanged.
    sample_pids = L.sample_train_participants(train_pids_all, widx, L.N_TRAIN_PARTICIPANTS, SEED)

    print(f"building training rows from {len(sample_pids)} TRAIN participants...", flush=True)
    all_rows = []
    for i, pid in enumerate(sample_pids, 1):
        all_rows.extend(L.build_training_rows(pid))
        if i % 50 == 0:
            print(f"  {i}/{len(sample_pids)}, {len(all_rows)} rows", flush=True)
    train_df = pd.DataFrame(all_rows)
    print(f"total training rows: {len(train_df)}", flush=True)

    feature_cols = [c for c in train_df.columns if c != "y"]   # target is 'y'
    X = train_df[feature_cols].to_numpy(dtype=np.float32)
    y = train_df["y"].to_numpy(dtype=np.float32)

    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline

    if MLP_IMPL == "sklearn":
        from sklearn.neural_network import MLPRegressor
        print("fitting MLP (sklearn, 128x128, squared error, early stopping)...", flush=True)
        mlp = make_pipeline(
            StandardScaler(),
            MLPRegressor(hidden_layer_sizes=(128, 128), activation="relu",
                         learning_rate_init=1e-3, batch_size=512, max_iter=60,
                         early_stopping=True, n_iter_no_change=5,
                         validation_fraction=0.1, random_state=SEED, verbose=False),
        )
    else:
        print(f"fitting MLP (torch, 128x128, {MLP_LOSS} loss, early stopping)...", flush=True)
        mlp = make_pipeline(StandardScaler(),
                            TorchMLP(loss=MLP_LOSS, seed=TRAIN_SEED, holdout_seed=SAMPLE_SEED))
    mlp.fit(X, y)
    print("MLP fit complete", flush=True)
    if L.WRITE_SEED_MANIFEST:
        print(f"SAMPLE_SEED={SAMPLE_SEED} TRAIN_SEED={TRAIN_SEED}", flush=True)
        est = mlp[-1]
        L.write_seed_manifest(OUTPUT_DIR, SAMPLE_SEED, TRAIN_SEED, sample_pids,
                              holdout_idx=getattr(est, "holdout_idx_", None),
                              n_rows=getattr(est, "n_rows_", None))

    series_cache = {}

    def full_series(pid):
        if pid not in series_cache:
            try:
                series_cache[pid] = load_cleaned_glucose_binned(pid)
            except Exception:
                series_cache[pid] = None
        return series_cache[pid]

    summary_rows, sig_rows = [], []
    for horizon_name, _bins in HORIZONS.items():
        print(f"scoring {horizon_name}...", flush=True)
        path = f"{L.ZEROSHOT_EVAL_DIR}/zeroshot_{horizon_name}_glucose_only.csv"
        df = pd.read_csv(path, dtype={"window_id": str, "person_id": str})
        df["bin_start"] = pd.to_datetime(df["bin_start"], utc=True)
        df["is_context"] = df["is_context"].astype(bool)

        recs = []
        for wid, g in df.groupby("window_id", sort=False):
            g = g.sort_values("bin_start")
            ctx, hor = g[g["is_context"]], g[~g["is_context"]]
            if ctx.empty or hor.empty:
                continue
            pid = g["person_id"].iloc[0]
            s_ctx = pd.Series(ctx["cgm_glucose"].to_numpy(),
                              index=pd.DatetimeIndex(ctx["bin_start"]))
            t_ctx = len(s_ctx) - 1
            anchor_time = s_ctx.index[-1]

            # circadian needs the participant's FULL history, not the 144-bin
            # context -- a 24h lookback does not fit inside a 12h window.
            fs = full_series(pid)
            fvals, f_anchor = None, None
            if fs is not None and len(fs):
                fidx = pd.DatetimeIndex(fs.index)
                if fidx.tz is None:
                    fidx = fidx.tz_localize("UTC")
                pos = fidx.get_indexer([anchor_time])
                if pos[0] != -1:
                    fvals, f_anchor = fs.to_numpy(), int(pos[0])

            for _, row in hor.iterrows():
                sa = int(round((row["bin_start"] - anchor_time).total_seconds() / 300))
                if sa <= 0:
                    continue
                actual = row["cgm_glucose"]
                if pd.isna(actual):
                    continue
                circ = (circadian_predict(fvals, f_anchor, sa)
                        if fvals is not None else None)
                feats = L.build_features_from_series(s_ctx, t_ctx, sa)
                mlp_pred = None
                if feats is not None:
                    fv = np.array([[feats.get(c, np.nan) for c in feature_cols]],
                                  dtype=np.float32)
                    if not np.isnan(fv).any():
                        mlp_pred = float(mlp.predict(fv)[0])
                recs.append({"window_id": wid, "person_id": pid,
                             "bin_start": row["bin_start"], "actual": float(actual),
                             "circadian": circ, "mlp": mlp_pred})

        d = pd.DataFrame(recs)
        if d.empty:
            print(f"  {horizon_name}: no rows scored", flush=True)
            continue
        d.to_csv(f"{OUTPUT_DIR}/predictions_{horizon_name}.csv", index=False)

        for method in ("circadian", "mlp"):
            sub = d.dropna(subset=[method]).copy()
            if sub.empty:
                continue
            sub["abs_error"] = (sub[method] - sub["actual"]).abs()
            mae = float(sub["abs_error"].mean())
            rmse = float(np.sqrt(((sub[method] - sub["actual"]) ** 2).mean()))
            summary_rows.append({
                "horizon": horizon_name, "method": method,
                "mae": round(mae, 4), "rmse": round(rmse, 4),
                "n_bins": len(sub), "n_participants": sub["person_id"].nunique(),
            })
            print(f"  {horizon_name} {method}: n={len(sub)}, MAE={mae:.4f}", flush=True)

        # cluster-by-participant test, circadian vs mlp on identical rows
        both = d.dropna(subset=["circadian", "mlp"]).copy()
        if len(both) and both["person_id"].nunique() >= 3:
            both["e_circ"] = (both["circadian"] - both["actual"]).abs()
            both["e_mlp"] = (both["mlp"] - both["actual"]).abs()
            per_p = both.groupby("person_id")[["e_circ", "e_mlp"]].mean()
            diff = (per_p["e_circ"] - per_p["e_mlp"]).to_numpy()
            t, p = stats.ttest_1samp(diff, 0.0)
            sig_rows.append({"horizon": horizon_name, "vs": "circadian_minus_mlp",
                             "circadian_mae": float(per_p["e_circ"].mean()),
                             "mlp_mae": float(per_p["e_mlp"].mean()),
                             "cluster_t": float(t), "cluster_p": float(p),
                             "n_participants": int(len(per_p))})

    if summary_rows:
        pd.DataFrame(summary_rows).to_csv(f"{OUTPUT_DIR}/summary.csv", index=False)
        print(pd.DataFrame(summary_rows).to_string(index=False), flush=True)
    if sig_rows:
        pd.DataFrame(sig_rows).to_csv(f"{OUTPUT_DIR}/significance.csv", index=False)
        print(pd.DataFrame(sig_rows).to_string(index=False), flush=True)
    print("EXTRA_BASELINES_V2_OK", flush=True)


if __name__ == "__main__":
    main()
