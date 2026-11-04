"""Plain Mamba-2 glucose forecasting baseline -- isolates whether Mamba's
own sequence-modeling mechanism is competitive, separately from the
TFT-style static-participant-embedding/variable-selection machinery that
SSM-CGM wraps around it. SSM-CGM has already been shown to lose to a
trivial persistence baseline on unseen participants specifically because of
that wrapper's dependence on a learned participant_id embedding
(SSMCGM_PERSISTENCE_GAP_AND_PLAN.md). This baseline removes that entire
wrapper: no static embedding, no variable-selection/gating, glucose-only,
just stacked plain Mamba-2 blocks and a direct linear regression head.

Mirrors code/baselines/lstm_zeroshot_baseline.py's infrastructure exactly (feature
construction, train/test split, test-window matching against
model_input/zeroshot_eval/zeroshot_{h}_glucose_only.csv, cluster-by-
participant significance testing) -- only the model architecture differs.
Direct multi-horizon regression head, NOT autoregressive generation (avoids
the error-compounding failure mode already found in CGM-LSM's tokenized
approach).

Uses the plain `mamba_ssm.Mamba2` class directly (not SSM-CGM's own custom
Mamba2MESIdentical subclass, which adds MES-style per-head B/C/X sharing
machinery specific to its own interpretability design) -- a simple,
un-tuned stack, matching this baseline's "quick, standard comparison point"
role rather than SSM-CGM's own tuned configuration.

Requires a GPU allocation even to import (mamba_ssm's Triton kernels crash
at import time otherwise, confirmed elsewhere in this project) -- must run
via sbatch with -G 1, not on the login node.
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
from mamba_ssm import Mamba2  # noqa: E402

WINDOW_INDEX = f"{REPO_ROOT}/model_input/window_index.csv"
ZEROSHOT_EVAL_DIR = f"{REPO_ROOT}/model_input/zeroshot_eval"
CHRONOS2_DIR = f"{REPO_ROOT}/chronos2_zeroshot_results/full/imputed"
LSTM_DIR = f"{REPO_ROOT}/lstm_zeroshot_results"
OUTPUT_DIR = f"{REPO_ROOT}/mamba_zeroshot_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

CONTEXT_LEN = 144  # 12h, matching every other model in this project
HORIZON_BINS = {"30min": 6, "1h": 12, "2h": 24}
MAX_HORIZON = 24
N_TRAIN_PARTICIPANTS = int(os.environ.get("MAMBA_N_TRAIN_PARTICIPANTS", "300"))
TRAIN_STRIDE_BINS = int(os.environ.get("MAMBA_TRAIN_STRIDE_BINS", "12"))  # 1h
D_MODEL = 64      # matches LSTM baseline's hidden_size for a fair architecture-only comparison
N_LAYERS = 2       # small stack -- "quick baseline" scale, not SSM-CGM's own tuned depth=4
D_STATE = 64
D_CONV = 4
EXPAND = 2
SEED = 42
BATCH_SIZE = 256
MAX_EPOCHS = 15
PATIENCE = 3
SMOKE = os.environ.get("MAMBA_SMOKE", "0") == "1"

torch.manual_seed(SEED)
device = "cuda" if torch.cuda.is_available() else "cpu"
assert device == "cuda", "Mamba2 CUDA kernels require a GPU -- run via sbatch with -G 1"


class PlainMambaForecaster(nn.Module):
    def __init__(self, d_model=D_MODEL, n_layers=N_LAYERS, n_outputs=MAX_HORIZON):
        super().__init__()
        self.in_proj = nn.Linear(1, d_model)
        self.layers = nn.ModuleList([
            Mamba2(d_model=d_model, d_state=D_STATE, d_conv=D_CONV, expand=EXPAND)
            for _ in range(n_layers)
        ])
        self.norms = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(n_layers)])
        self.head = nn.Linear(d_model, n_outputs)

    def forward(self, x):  # x: (batch, CONTEXT_LEN)
        h = self.in_proj(x.unsqueeze(-1))  # (batch, CONTEXT_LEN, d_model)
        for layer, norm in zip(self.layers, self.norms):
            h = norm(h + layer(h))  # residual + Mamba2 block, pre-existing norm pattern
        return self.head(h[:, -1, :])  # (batch, n_outputs) -- last-step readout, direct multi-horizon


def build_train_examples(pid):
    s = load_cleaned_glucose_binned(pid)
    if s is None or len(s) < CONTEXT_LEN + MAX_HORIZON + 10:
        return []
    vals = s.to_numpy()
    examples = []
    for t0 in range(0, len(vals) - CONTEXT_LEN - MAX_HORIZON, TRAIN_STRIDE_BINS):
        ctx = vals[t0:t0 + CONTEXT_LEN]
        tgt = vals[t0 + CONTEXT_LEN:t0 + CONTEXT_LEN + MAX_HORIZON]
        if np.isnan(ctx).any() or np.isnan(tgt).any():
            continue
        examples.append((ctx.astype(np.float32), tgt.astype(np.float32)))
    return examples


def main():
    print(f"device={device}, SMOKE={SMOKE}", file=sys.stderr)
    widx = pd.read_csv(WINDOW_INDEX, dtype={"person_id": str})
    train_pids_all = sorted(widx.loc[widx["recommended_split"] == "train", "person_id"].unique())
    test_pids_all = set(widx.loc[widx["recommended_split"] == "test", "person_id"].unique())
    assert not (set(train_pids_all) & test_pids_all), "train/test participant overlap -- leakage"
    print(f"train pool: {len(train_pids_all)}, test pool: {len(test_pids_all)}", file=sys.stderr)

    n_train = 5 if SMOKE else N_TRAIN_PARTICIPANTS
    rng = np.random.default_rng(SEED)
    sample_pids = rng.choice(train_pids_all, size=min(n_train, len(train_pids_all)), replace=False)

    print(f"building training examples from {len(sample_pids)} participants...", file=sys.stderr)
    all_examples = []
    for i, pid in enumerate(sample_pids):
        all_examples.extend(build_train_examples(pid))
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(sample_pids)}, {len(all_examples)} examples", file=sys.stderr)
    print(f"total training examples: {len(all_examples)}", file=sys.stderr)
    assert len(all_examples) > 0, "no training examples built -- feature/window construction bug"

    all_ctx = np.concatenate([e[0] for e in all_examples])
    mu, sigma = float(all_ctx.mean()), float(all_ctx.std())
    print(f"normalization: mu={mu:.2f}, sigma={sigma:.2f}", file=sys.stderr)

    n_val = max(1, int(0.1 * len(all_examples)))
    perm = rng.permutation(len(all_examples))
    val_idx, train_idx = perm[:n_val], perm[n_val:]

    def to_tensors(idxs):
        ctx = np.stack([all_examples[i][0] for i in idxs])
        tgt = np.stack([all_examples[i][1] for i in idxs])
        ctx = (ctx - mu) / sigma
        return torch.from_numpy(ctx), torch.from_numpy(tgt)

    train_ctx, train_tgt = to_tensors(train_idx)
    val_ctx, val_tgt = to_tensors(val_idx)

    model = PlainMambaForecaster().to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model params: {n_params}", file=sys.stderr)
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
            loss = loss_fn(pred, (yb - mu) / sigma)
            loss.backward()
            opt.step()
            total_loss += loss.item() * len(idx)
        model.eval()
        with torch.no_grad():
            vpred_chunks = []
            for b0 in range(0, val_ctx.shape[0], BATCH_SIZE):
                vpred_chunks.append(model(val_ctx[b0:b0 + BATCH_SIZE].to(device)).cpu())
            vpred = torch.cat(vpred_chunks, dim=0)
            vloss = loss_fn(vpred, (val_tgt - mu) / sigma).item()
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
    torch.save(model.state_dict(), f"{OUTPUT_DIR}/mamba_model.pt")
    print("training complete", file=sys.stderr)

    summary_rows, sig_rows = [], []
    model.eval()
    for horizon_name, hbin in HORIZON_BINS.items():
        path = f"{ZEROSHOT_EVAL_DIR}/zeroshot_{horizon_name}_glucose_only.csv"
        df = pd.read_csv(path, dtype={"window_id": str, "person_id": str})
        df["bin_start"] = pd.to_datetime(df["bin_start"], utc=True)
        df["is_context"] = df["is_context"].astype(bool)

        wids, pids, ctxs, actuals_list, bin_starts_list = [], [], [], [], []
        persist_list = []
        for wid, g in df.groupby("window_id", sort=False):
            g = g.sort_values("bin_start")
            ctx = g[g["is_context"]]
            hor = g[~g["is_context"]]
            if len(ctx) != CONTEXT_LEN or hor.empty:
                continue
            ctx_vals = ctx["cgm_glucose"].to_numpy()
            if np.isnan(ctx_vals).any():
                continue
            hor_vals = hor["cgm_glucose"].to_numpy()
            wids.append(wid)
            pids.append(g["person_id"].iloc[0])
            ctxs.append(ctx_vals.astype(np.float32))
            actuals_list.append(hor_vals)
            bin_starts_list.append(hor["bin_start"].to_numpy())
            persist_list.append(float(ctx_vals[-1]))

        if not ctxs:
            print(f"  {horizon_name}: NO SCORABLE WINDOWS", file=sys.stderr)
            continue
        ctx_batch = (np.stack(ctxs) - mu) / sigma
        with torch.no_grad():
            pred_chunks = []
            cb = torch.from_numpy(ctx_batch)
            for b0 in range(0, cb.shape[0], BATCH_SIZE):
                pred_chunks.append(model(cb[b0:b0 + BATCH_SIZE].to(device)).cpu())
            preds = torch.cat(pred_chunks, dim=0).numpy() * sigma + mu

        rows = []
        for i in range(len(wids)):
            n_h = len(actuals_list[i])
            for h in range(n_h):
                rows.append({
                    "window_id": wids[i], "person_id": pids[i],
                    "bin_start": bin_starts_list[i][h],
                    "actual": actuals_list[i][h],
                    "predictions": float(preds[i, h]),
                    "persistence": persist_list[i],
                })
        pred_df = pd.DataFrame(rows)
        pred_df["bin_start"] = pd.to_datetime(pred_df["bin_start"], utc=True)
        pred_df["abs_error"] = (pred_df["predictions"] - pred_df["actual"]).abs()
        pred_df.to_csv(f"{OUTPUT_DIR}/predictions_{horizon_name}.csv", index=False)

        scored = pred_df.dropna(subset=["actual"])
        mae = scored["abs_error"].mean()
        rmse = float(np.sqrt((scored["abs_error"] ** 2).mean()))
        n_participants = scored["person_id"].nunique()
        summary_rows.append({"horizon": horizon_name, "method": "mamba", "mae": round(float(mae), 4),
                              "rmse": round(rmse, 4), "n_bins": len(scored), "n_participants": n_participants})
        print(f"  {horizon_name}: n={len(scored)}, participants={n_participants}, MAE={mae:.4f}", file=sys.stderr)

        c2_path = f"{CHRONOS2_DIR}/predictions_{horizon_name}_glucose_only.csv"
        c2 = pd.read_csv(c2_path, dtype={"window_id": str})
        c2["bin_start"] = pd.to_datetime(c2["bin_start"], utc=True)
        m = pred_df.merge(c2[["window_id", "bin_start", "predictions"]].rename(columns={"predictions": "chronos2_pred"}),
                           on=["window_id", "bin_start"], how="inner")

        lstm_path = f"{LSTM_DIR}/predictions_{horizon_name}.csv"
        lstm_avail = os.path.exists(lstm_path)
        if lstm_avail:
            lstm_df = pd.read_csv(lstm_path, dtype={"window_id": str})
            lstm_df["bin_start"] = pd.to_datetime(lstm_df["bin_start"], utc=True)
            m = m.merge(lstm_df[["window_id", "bin_start", "predictions"]].rename(columns={"predictions": "lstm_pred"}),
                        on=["window_id", "bin_start"], how="inner")

        m = m.dropna(subset=["actual"])
        m["mamba_err"] = (m["predictions"] - m["actual"]).abs()
        m["persist_err"] = (m["persistence"] - m["actual"]).abs()
        m["c2_err"] = (m["chronos2_pred"] - m["actual"]).abs()
        refs = [("persistence", "persist_err"), ("chronos2", "c2_err")]
        if lstm_avail:
            m["lstm_err"] = (m["lstm_pred"] - m["actual"]).abs()
            refs.append(("lstm", "lstm_err"))

        for ref_name, ref_col in refs:
            d = m["mamba_err"] - m[ref_col]
            per_person = d.groupby(m["person_id"]).mean()
            t_stat, p_val = stats.ttest_1samp(per_person.to_numpy(), 0.0)
            sig_rows.append({
                "horizon": horizon_name, "vs": ref_name, "n_matched_bins": len(m),
                "n_participants_clustered": len(per_person),
                "mamba_mae_matched": round(float(m["mamba_err"].mean()), 4),
                "ref_mae_matched": round(float(m[ref_col].mean()), 4),
                "cluster_t": round(float(t_stat), 4), "cluster_p": float(p_val),
                "mamba_better": bool(m["mamba_err"].mean() < m[ref_col].mean()),
            })

    pd.DataFrame(summary_rows).to_csv(f"{OUTPUT_DIR}/summary.csv", index=False)
    sig_df = pd.DataFrame(sig_rows)
    sig_df.to_csv(f"{OUTPUT_DIR}/significance.csv", index=False)
    print(sig_df.to_string(index=False), file=sys.stderr)
    print("MAMBA_ZEROSHOT_BASELINE_OK", file=sys.stderr)


if __name__ == "__main__":
    main()
