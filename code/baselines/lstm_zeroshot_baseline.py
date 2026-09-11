"""Simple single-layer LSTM glucose forecasting baseline -- the standard
"basic deep learning" comparison point this benchmark was missing. Matches
the role a Seq2Seq LSTM plays in the original SSM-CGM paper's own "Quick
Comparisons" (Appendix A.3, Table 3: Seq2Seq LSTM MAE=7.85 mg/dL, their
population/split).

Deliberately simple: one LSTM layer, direct multi-horizon regression head
(NOT autoregressive generation -- avoids the error-compounding failure mode
already found in CGM-LSM's tokenized/autoregressive approach). Trained from
scratch on a sample of the official AI-READI TRAIN split (verified disjoint
from test), scored zero-shot on the EXACT SAME test windows already used
for Chronos-2/TimesFM/LightGBM's glucose-only comparison
(model_input/zeroshot_eval/zeroshot_{h}_glucose_only.csv), matched via
window_id/bin_start so MAE is directly comparable.

Causal only: context is the 144 real context bins of each window (12h);
windows with ANY missing context bin are dropped (never imputed, matching
this project's convention -- cgm_glucose is never imputed anywhere else
either). No covariates (glucose-only, matching LightGBM's scope).
"""
import os
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy import stats

REPO_ROOT = "/path/to/aireadi-collab"
sys.path.insert(0, os.path.join(REPO_ROOT, "code/model_input"))
# Sibling modules live in adjacent topic directories under code/.
_CODE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _d in ("analysis", "baselines", "cgmlsm", "chronos2", "meal_detection", "model_input", "ssmcgm", "timesfm"):
    sys.path.insert(0, os.path.join(_CODE, _d))
from build_participant_grid import load_cleaned_glucose_binned  # noqa: E402

WINDOW_INDEX = f"{REPO_ROOT}/model_input/window_index.csv"
ZEROSHOT_EVAL_DIR = f"{REPO_ROOT}/model_input/zeroshot_eval"
CHRONOS2_DIR = f"{REPO_ROOT}/chronos2_zeroshot_results/full/imputed"
OUTPUT_DIR = f"{REPO_ROOT}/lstm_zeroshot_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

CONTEXT_LEN = 144  # 12h, matching every other model in this project
HORIZON_BINS = {"30min": 6, "1h": 12, "2h": 24}
MAX_HORIZON = 24
N_TRAIN_PARTICIPANTS = int(os.environ.get("LSTM_N_TRAIN_PARTICIPANTS", "300"))
TRAIN_STRIDE_BINS = int(os.environ.get("LSTM_TRAIN_STRIDE_BINS", "12"))  # 1h
HIDDEN_SIZE = 64
SEED = 42
BATCH_SIZE = 256
MAX_EPOCHS = 15
PATIENCE = 3
SMOKE = os.environ.get("LSTM_SMOKE", "0") == "1"

torch.manual_seed(SEED)
device = "cuda" if torch.cuda.is_available() else "cpu"


class SimpleLSTM(nn.Module):
    def __init__(self, hidden_size=HIDDEN_SIZE, n_outputs=MAX_HORIZON):
        super().__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=hidden_size, num_layers=1, batch_first=True)
        self.head = nn.Linear(hidden_size, n_outputs)

    def forward(self, x):  # x: (batch, CONTEXT_LEN)
        out, (h, c) = self.lstm(x.unsqueeze(-1))
        return self.head(h[-1])  # (batch, n_outputs), one value per future bin 1..MAX_HORIZON


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

    # normalize by simple global mean/std of training context values (causal: fit on train only)
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

    model = SimpleLSTM().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.L1Loss()  # MAE loss, directly matches the eval metric

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
            vpred = model(val_ctx.to(device)).cpu()
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
    torch.save(model.state_dict(), f"{OUTPUT_DIR}/lstm_model.pt")
    print("training complete", file=sys.stderr)

    # -- Score on the exact same test windows used for Chronos-2/TimesFM/LightGBM --
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
            preds = model(torch.from_numpy(ctx_batch).to(device)).cpu().numpy() * sigma + mu  # (n, MAX_HORIZON)

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
        summary_rows.append({"horizon": horizon_name, "method": "lstm", "mae": round(float(mae), 4),
                              "rmse": round(rmse, 4), "n_bins": len(scored), "n_participants": n_participants})
        print(f"  {horizon_name}: n={len(scored)}, participants={n_participants}, MAE={mae:.4f}", file=sys.stderr)

        c2_path = f"{CHRONOS2_DIR}/predictions_{horizon_name}_glucose_only.csv"
        c2 = pd.read_csv(c2_path, dtype={"window_id": str})
        c2["bin_start"] = pd.to_datetime(c2["bin_start"], utc=True)

        m = pred_df.merge(c2[["window_id", "bin_start", "predictions"]].rename(columns={"predictions": "chronos2_pred"}),
                           on=["window_id", "bin_start"], how="inner")
        m = m.dropna(subset=["actual"])
        m["lstm_err"] = (m["predictions"] - m["actual"]).abs()
        m["persist_err"] = (m["persistence"] - m["actual"]).abs()
        m["c2_err"] = (m["chronos2_pred"] - m["actual"]).abs()

        for ref_name, ref_col in [("persistence", "persist_err"), ("chronos2", "c2_err")]:
            d = m["lstm_err"] - m[ref_col]
            per_person = d.groupby(m["person_id"]).mean()
            t_stat, p_val = stats.ttest_1samp(per_person.to_numpy(), 0.0)
            sig_rows.append({
                "horizon": horizon_name, "vs": ref_name, "n_matched_bins": len(m),
                "n_participants_clustered": len(per_person),
                "lstm_mae_matched": round(float(m["lstm_err"].mean()), 4),
                "ref_mae_matched": round(float(m[ref_col].mean()), 4),
                "cluster_t": round(float(t_stat), 4), "cluster_p": float(p_val),
                "lstm_better": bool(m["lstm_err"].mean() < m[ref_col].mean()),
            })

    pd.DataFrame(summary_rows).to_csv(f"{OUTPUT_DIR}/summary.csv", index=False)
    sig_df = pd.DataFrame(sig_rows)
    sig_df.to_csv(f"{OUTPUT_DIR}/significance.csv", index=False)
    print(sig_df.to_string(index=False), file=sys.stderr)
    print("LSTM_ZEROSHOT_BASELINE_OK", file=sys.stderr)


if __name__ == "__main__":
    main()
