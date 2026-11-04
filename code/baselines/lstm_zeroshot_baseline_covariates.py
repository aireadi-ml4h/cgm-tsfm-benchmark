"""LSTM baseline, WITH COVARIATES arm -- extends code/baselines/lstm_zeroshot_baseline.py
(glucose-only) with the same 6-channel covariate set used for Chronos-2/
TimesFM's own "with covariates" comparison (report.md Sec 1): heart_rate,
stress, respiratory_rate, oxygen_saturation, physical_activity_steps,
physical_activity_calorie_rate.

Reuses `model_input/zeroshot_eval/zeroshot_{h}_with_covariates.csv` directly
-- these files already carry the raw covariate columns on the same
window_id/is_context/bin_start schema as the glucose_only files, ffilled/
bfilled within each window's own context and falling back to 0.0 if
unresolvable (chronos2_zeroshot_eval.py's own documented convention,
reused unchanged here for apples-to-apples comparability).

Architecture, training population, test windows, and everything else is
IDENTICAL to the glucose-only version (`code/baselines/lstm_zeroshot_baseline.py`) --
only the input is extended from 1 channel (glucose) to 7 (glucose + 6
covariates), so any MAE difference is attributable to the covariates, not a
confound from a different model/population/window set.
"""
import os
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
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
LSTM_GLUCOSE_ONLY_DIR = f"{REPO_ROOT}/lstm_zeroshot_results"
OUTPUT_DIR = f"{REPO_ROOT}/lstm_zeroshot_covariates_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

CONTEXT_LEN = 144
HORIZON_BINS = {"30min": 6, "1h": 12, "2h": 24}
MAX_HORIZON = 24
COVARIATE_CHANNELS = ["heart_rate", "stress", "respiratory_rate", "oxygen_saturation",
                       "physical_activity_steps", "physical_activity_calorie_rate"]
N_CHANNELS = 1 + len(COVARIATE_CHANNELS)  # glucose + 6 covariates
N_TRAIN_PARTICIPANTS = int(os.environ.get("LSTM_N_TRAIN_PARTICIPANTS", "300"))
TRAIN_STRIDE_BINS = int(os.environ.get("LSTM_TRAIN_STRIDE_BINS", "12"))
HIDDEN_SIZE = 64
SEED = 42
BATCH_SIZE = 256
MAX_EPOCHS = 15
PATIENCE = 3
SMOKE = os.environ.get("LSTM_SMOKE", "0") == "1"

torch.manual_seed(SEED)
device = "cuda" if torch.cuda.is_available() else "cpu"


class SimpleLSTMCov(nn.Module):
    def __init__(self, n_channels=N_CHANNELS, hidden_size=HIDDEN_SIZE, n_outputs=MAX_HORIZON):
        super().__init__()
        self.lstm = nn.LSTM(input_size=n_channels, hidden_size=hidden_size, num_layers=1, batch_first=True)
        self.head = nn.Linear(hidden_size, n_outputs)

    def forward(self, x):  # x: (batch, CONTEXT_LEN, N_CHANNELS)
        out, (h, c) = self.lstm(x)
        return self.head(h[-1])


def _fill_covariate_context(g_ctx):
    """Same convention as chronos2_zeroshot_eval.py: ffill/bfill within the
    window's own context, fall back to 0.0 only if unresolvable."""
    out = {}
    for ch in COVARIATE_CHANNELS:
        s = g_ctx[ch].ffill().bfill().fillna(0.0)
        out[ch] = s.to_numpy(dtype=np.float32)
    return out


def build_train_examples_cov(pid, wide_df):
    """wide_df: this participant's rows from a with_covariates CSV restricted
    to is_context (built once per horizon file at training-data-collection
    time is wasteful; instead we build training examples directly from the
    raw per-participant binned series using load_cleaned_glucose_binned for
    glucose and the SAME participant's covariate columns pulled from the
    already-loaded window CSV -- see main() for how wide_df is assembled)."""
    if wide_df is None or len(wide_df) < CONTEXT_LEN + MAX_HORIZON + 10:
        return []
    wide_df = wide_df.sort_values("bin_start")
    glu = wide_df["cgm_glucose"].to_numpy(dtype=np.float32)
    covs = _fill_covariate_context(wide_df)
    examples = []
    for t0 in range(0, len(glu) - CONTEXT_LEN - MAX_HORIZON, TRAIN_STRIDE_BINS):
        ctx_glu = glu[t0:t0 + CONTEXT_LEN]
        tgt = glu[t0 + CONTEXT_LEN:t0 + CONTEXT_LEN + MAX_HORIZON]
        if np.isnan(ctx_glu).any() or np.isnan(tgt).any():
            continue
        ctx_cov = np.stack([covs[ch][t0:t0 + CONTEXT_LEN] for ch in COVARIATE_CHANNELS], axis=-1)
        ctx_full = np.concatenate([ctx_glu[:, None], ctx_cov], axis=-1)  # (CONTEXT_LEN, N_CHANNELS)
        examples.append((ctx_full.astype(np.float32), tgt.astype(np.float32)))
    return examples


def main():
    print(f"device={device}, SMOKE={SMOKE}", file=sys.stderr)
    widx = pd.read_csv(WINDOW_INDEX, dtype={"person_id": str})
    train_pids_all = sorted(widx.loc[widx["recommended_split"] == "train", "person_id"].unique())
    test_pids_all = set(widx.loc[widx["recommended_split"] == "test", "person_id"].unique())
    assert not (set(train_pids_all) & test_pids_all), "train/test participant overlap -- leakage"

    n_train = 5 if SMOKE else N_TRAIN_PARTICIPANTS
    rng = np.random.default_rng(SEED)
    sample_pids = rng.choice(train_pids_all, size=min(n_train, len(train_pids_all)), replace=False)

    # Build training examples from the 1h with_covariates window file's raw
    # per-window rows for the sampled train participants (these windows are
    # non-overlapping context+horizon chunks per person; concatenating a
    # person's own chunks in order and sliding across them, same approach as
    # the glucose-only script's use of load_cleaned_glucose_binned, but here
    # we need the covariate columns too, which only exist in the window CSVs,
    # not in load_cleaned_glucose_binned's raw output).
    train_src = pd.read_csv(f"{ZEROSHOT_EVAL_DIR}/zeroshot_1h_with_covariates.csv",
                             dtype={"window_id": str, "person_id": str})
    train_src = train_src[train_src["person_id"].isin(set(sample_pids))]
    train_src["bin_start"] = pd.to_datetime(train_src["bin_start"], utc=True)

    print(f"building training examples from {len(sample_pids)} participants (train split, with covariates)...",
          file=sys.stderr)
    all_examples = []
    for i, pid in enumerate(sample_pids):
        pdf = train_src[train_src["person_id"] == pid].drop_duplicates(subset=["bin_start"])
        all_examples.extend(build_train_examples_cov(pid, pdf))
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(sample_pids)}, {len(all_examples)} examples", file=sys.stderr)
    print(f"total training examples: {len(all_examples)}", file=sys.stderr)
    assert len(all_examples) > 0, "no training examples built"

    all_ctx = np.concatenate([e[0] for e in all_examples], axis=0)  # (N*CONTEXT_LEN, N_CHANNELS)
    mu = all_ctx.mean(axis=0)  # per-channel normalization, causal (fit on train only)
    sigma = all_ctx.std(axis=0)
    sigma[sigma == 0] = 1.0
    print(f"per-channel mu={mu}, sigma={sigma}", file=sys.stderr)
    glu_mu, glu_sigma = float(mu[0]), float(sigma[0])

    n_val = max(1, int(0.1 * len(all_examples)))
    perm = rng.permutation(len(all_examples))
    val_idx, train_idx = perm[:n_val], perm[n_val:]

    def to_tensors(idxs):
        ctx = np.stack([all_examples[i][0] for i in idxs])  # (n, CONTEXT_LEN, N_CHANNELS)
        tgt = np.stack([all_examples[i][1] for i in idxs])  # (n, MAX_HORIZON) -- glucose only
        ctx = (ctx - mu[None, None, :]) / sigma[None, None, :]
        return torch.from_numpy(ctx), torch.from_numpy(tgt)

    train_ctx, train_tgt = to_tensors(train_idx)
    val_ctx, val_tgt = to_tensors(val_idx)

    model = SimpleLSTMCov().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.L1Loss()

    best_val = float("inf")
    best_state = None
    epochs_no_improve = 0
    n = train_ctx.shape[0]
    max_epochs = 2 if SMOKE else MAX_EPOCHS
    for epoch in range(max_epochs):
        model.train()
        perm_e = torch.randperm(n)
        total_loss = 0.0
        for b0 in range(0, n, BATCH_SIZE):
            idx = perm_e[b0:b0 + BATCH_SIZE]
            xb, yb = train_ctx[idx].to(device), train_tgt[idx].to(device)
            opt.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, (yb - glu_mu) / glu_sigma)
            loss.backward()
            opt.step()
            total_loss += loss.item() * len(idx)
        model.eval()
        with torch.no_grad():
            vpred = model(val_ctx.to(device)).cpu()
            vloss = loss_fn(vpred, (val_tgt - glu_mu) / glu_sigma).item()
        print(f"epoch {epoch}: train_loss={total_loss/n:.4f} val_loss(norm)={vloss:.4f}", file=sys.stderr)
        if vloss < best_val:
            best_val = vloss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= PATIENCE:
                print("early stopping", file=sys.stderr)
                break
    model.load_state_dict(best_state)
    torch.save(model.state_dict(), f"{OUTPUT_DIR}/lstm_cov_model.pt")
    print("training complete", file=sys.stderr)

    summary_rows, sig_rows = [], []
    model.eval()
    for horizon_name, hbin in HORIZON_BINS.items():
        path = f"{ZEROSHOT_EVAL_DIR}/zeroshot_{horizon_name}_with_covariates.csv"
        df = pd.read_csv(path, dtype={"window_id": str, "person_id": str})
        df["bin_start"] = pd.to_datetime(df["bin_start"], utc=True)
        df["is_context"] = df["is_context"].astype(bool)

        wids, pids, ctxs, actuals_list, bin_starts_list = [], [], [], [], []
        for wid, g in df.groupby("window_id", sort=False):
            g = g.sort_values("bin_start")
            ctx = g[g["is_context"]]
            hor = g[~g["is_context"]]
            if len(ctx) != CONTEXT_LEN or hor.empty:
                continue
            ctx_glu = ctx["cgm_glucose"].to_numpy()
            if np.isnan(ctx_glu).any():
                continue
            covs = _fill_covariate_context(ctx)
            ctx_cov = np.stack([covs[ch] for ch in COVARIATE_CHANNELS], axis=-1)
            ctx_full = np.concatenate([ctx_glu[:, None], ctx_cov], axis=-1)
            hor_vals = hor["cgm_glucose"].to_numpy()
            wids.append(wid)
            pids.append(g["person_id"].iloc[0])
            ctxs.append(ctx_full.astype(np.float32))
            actuals_list.append(hor_vals)
            bin_starts_list.append(hor["bin_start"].to_numpy())

        if not ctxs:
            print(f"  {horizon_name}: NO SCORABLE WINDOWS", file=sys.stderr)
            continue
        ctx_batch = (np.stack(ctxs) - mu[None, None, :]) / sigma[None, None, :]
        with torch.no_grad():
            preds = model(torch.from_numpy(ctx_batch).to(device)).cpu().numpy() * glu_sigma + glu_mu

        rows = []
        for i in range(len(wids)):
            n_h = len(actuals_list[i])
            for h in range(n_h):
                rows.append({
                    "window_id": wids[i], "person_id": pids[i],
                    "bin_start": bin_starts_list[i][h],
                    "actual": actuals_list[i][h],
                    "predictions": float(preds[i, h]),
                })
        pred_df = pd.DataFrame(rows)
        pred_df["bin_start"] = pd.to_datetime(pred_df["bin_start"], utc=True)
        pred_df["abs_error"] = (pred_df["predictions"] - pred_df["actual"]).abs()
        pred_df.to_csv(f"{OUTPUT_DIR}/predictions_{horizon_name}.csv", index=False)

        scored = pred_df.dropna(subset=["actual"])
        mae = scored["abs_error"].mean()
        rmse = float(np.sqrt((scored["abs_error"] ** 2).mean()))
        n_participants = scored["person_id"].nunique()
        summary_rows.append({"horizon": horizon_name, "method": "lstm_covariates", "mae": round(float(mae), 4),
                              "rmse": round(rmse, 4), "n_bins": len(scored), "n_participants": n_participants})
        print(f"  {horizon_name}: n={len(scored)}, participants={n_participants}, MAE={mae:.4f}", file=sys.stderr)

        # Significance vs. this SAME model's own glucose-only result (real
        # apples-to-apples: same architecture/population/test windows, only
        # the input channels differ).
        go_path = f"{LSTM_GLUCOSE_ONLY_DIR}/predictions_{horizon_name}.csv"
        if os.path.exists(go_path):
            go = pd.read_csv(go_path, dtype={"window_id": str})
            go["bin_start"] = pd.to_datetime(go["bin_start"], utc=True)
            m = pred_df.merge(go[["window_id", "bin_start", "predictions"]].rename(
                columns={"predictions": "glucose_only_pred"}), on=["window_id", "bin_start"], how="inner")
            m = m.dropna(subset=["actual"])
            m["cov_err"] = (m["predictions"] - m["actual"]).abs()
            m["go_err"] = (m["glucose_only_pred"] - m["actual"]).abs()
            d = m["cov_err"] - m["go_err"]
            per_person = d.groupby(m["person_id"]).mean()
            t_stat, p_val = stats.ttest_1samp(per_person.to_numpy(), 0.0)
            sig_rows.append({
                "horizon": horizon_name, "vs": "lstm_glucose_only", "n_matched_bins": len(m),
                "n_participants_clustered": len(per_person),
                "cov_mae_matched": round(float(m["cov_err"].mean()), 4),
                "glucose_only_mae_matched": round(float(m["go_err"].mean()), 4),
                "cluster_t": round(float(t_stat), 4), "cluster_p": float(p_val),
                "covariates_help": bool(m["cov_err"].mean() < m["go_err"].mean()),
            })
        else:
            print(f"  WARNING: {go_path} not found, skipping vs-glucose-only significance for {horizon_name}",
                  file=sys.stderr)

    pd.DataFrame(summary_rows).to_csv(f"{OUTPUT_DIR}/summary.csv", index=False)
    sig_df = pd.DataFrame(sig_rows)
    sig_df.to_csv(f"{OUTPUT_DIR}/significance_vs_glucose_only.csv", index=False)
    print(sig_df.to_string(index=False), file=sys.stderr)
    print("LSTM_COVARIATES_BASELINE_OK", file=sys.stderr)


if __name__ == "__main__":
    main()
