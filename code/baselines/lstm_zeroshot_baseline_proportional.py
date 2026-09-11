"""LSTM baseline, PROPORTIONAL (known-patient) split -- companion to
code/baselines/lstm_zeroshot_baseline.py's disjoint-split version. Same architecture
(single-layer LSTM, hidden_size=64, direct multi-horizon regression head,
context=144 bins/12h), same glucose-only scope, same causal (no-imputation)
window construction. The only thing that changes is the population: trains
and evaluates on each participant's OWN timeline (ssmcgm proportional CSVs,
where the same participants appear in both train and test, at different
times), instead of the disjoint official split.

Built to test whether a model with NO participant-identity mechanism shows
any benefit from a proportional split -- if not, that is evidence the
proportional-split advantage found for SSM-CGM's personalization is
specific to models with identity/adaptation mechanisms, not a generic
property of the split itself.

Windows are built independently within each of the train/test proportional
CSVs (sorted by each row's own `ds` sequence index, which is a real,
per-participant-monotonic time index, not a calendar date) -- never crossing
the train/test file boundary, so there is no leakage of test-period glucose
into a training window.
"""
import os
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy import stats

REPO_ROOT = "/path/to/aireadi-collab"
TRAIN_CSV = f"{REPO_ROOT}/model_input/ssmcgm/ssmcgm_train_proportional.csv"
TEST_CSV = f"{REPO_ROOT}/model_input/ssmcgm/ssmcgm_test_proportional.csv"
OUTPUT_DIR = f"{REPO_ROOT}/lstm_zeroshot_results_proportional"
os.makedirs(OUTPUT_DIR, exist_ok=True)

CONTEXT_LEN = 144
HORIZON_BINS = {"30min": 6, "1h": 12, "2h": 24}
MAX_HORIZON = 24
N_TRAIN_PARTICIPANTS = int(os.environ.get("LSTM_N_TRAIN_PARTICIPANTS", "300"))
TRAIN_STRIDE_BINS = int(os.environ.get("LSTM_TRAIN_STRIDE_BINS", "12"))
TEST_STRIDE_BINS = int(os.environ.get("LSTM_TEST_STRIDE_BINS", "6"))
HIDDEN_SIZE = 64
SEED = 42
BATCH_SIZE = 256
MAX_EPOCHS = 60
PATIENCE = 3
SMOKE = os.environ.get("LSTM_SMOKE", "0") == "1"

torch.manual_seed(SEED)
device = "cuda" if torch.cuda.is_available() else "cpu"


class SimpleLSTM(nn.Module):
    def __init__(self, hidden_size=HIDDEN_SIZE, n_outputs=MAX_HORIZON):
        super().__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=hidden_size, num_layers=1, batch_first=True)
        self.head = nn.Linear(hidden_size, n_outputs)

    def forward(self, x):
        out, (h, c) = self.lstm(x.unsqueeze(-1))
        return self.head(h[-1])


def series_by_participant(df):
    # Group by series_id, NOT participant_id -- a participant can have
    # multiple series_id segments (each its own contiguous real time run,
    # `ds` restarting at 0 per segment). Grouping by participant_id alone
    # and sorting only by `ds` would interleave rows from unrelated segments
    # that share the same `ds` value, scrambling the time order and
    # corrupting every window built from it. Real bug, found via a smoke
    # test showing LSTM MAE of ~129 vs persistence's ~10-17 -- far worse
    # than even a random/mean predictor should produce, the signature of
    # windows built from a shuffled, non-contiguous sequence.
    out = {}
    for sid, g in df.groupby("series_id"):
        g = g.sort_values("ds")
        pid = str(g["participant_id"].iloc[0])
        out.setdefault(pid, []).append(g["cgm_glucose"].to_numpy(dtype=np.float32))
    # Concatenate a participant's own segments in a stable, deterministic
    # order (by first ds/series_id) -- windows are still never built across
    # a segment boundary in a way that mixes unrelated series (see
    # build_examples: within-segment only, called per-segment below).
    return out


def all_segments(series_dict):
    """Flatten {pid: [seg1, seg2, ...]} into a list of (pid, segment) pairs,
    so callers can build windows within each segment independently."""
    return [(pid, seg) for pid, segs in series_dict.items() for seg in segs]


def build_examples(vals, stride):
    examples = []
    for t0 in range(0, len(vals) - CONTEXT_LEN - MAX_HORIZON, stride):
        ctx = vals[t0:t0 + CONTEXT_LEN]
        tgt = vals[t0 + CONTEXT_LEN:t0 + CONTEXT_LEN + MAX_HORIZON]
        if np.isnan(ctx).any() or np.isnan(tgt).any():
            continue
        examples.append((ctx.astype(np.float32), tgt.astype(np.float32)))
    return examples


def main():
    print(f"device={device}, SMOKE={SMOKE}", file=sys.stderr)
    print("loading proportional train/test CSVs...", file=sys.stderr)
    train_df = pd.read_csv(TRAIN_CSV, usecols=["series_id", "participant_id", "ds", "cgm_glucose"])
    test_df = pd.read_csv(TEST_CSV, usecols=["series_id", "participant_id", "ds", "cgm_glucose"])

    train_pids_all = set(train_df["participant_id"].astype(str))
    test_pids_all = set(test_df["participant_id"].astype(str))
    overlap = train_pids_all & test_pids_all
    print(f"train participants: {len(train_pids_all)}, test participants: {len(test_pids_all)}, "
          f"overlap: {len(overlap)} ({100*len(overlap)/len(test_pids_all):.1f}% of test) "
          f"-- CONFIRMS known-patient regime if overlap is high", file=sys.stderr)

    print(f"cgm_glucose describe: {train_df['cgm_glucose'].describe().to_dict()}", file=sys.stderr)
    print(f"n_nan: {train_df['cgm_glucose'].isna().sum()} of {len(train_df)}", file=sys.stderr)
    _one_sid = train_df['series_id'].iloc[0]
    _seg = train_df[train_df['series_id'] == _one_sid].sort_values('ds')
    print(f"sample segment {_one_sid}, first 15 (ds, cgm_glucose):\n{_seg[['ds','cgm_glucose']].head(15).to_string()}", file=sys.stderr)

    train_series = series_by_participant(train_df)
    test_series = series_by_participant(test_df)

    n_train = 5 if SMOKE else N_TRAIN_PARTICIPANTS
    rng = np.random.default_rng(SEED)
    train_pids_sample = rng.choice(sorted(train_pids_all), size=min(n_train, len(train_pids_all)), replace=False)

    print(f"building training examples from {len(train_pids_sample)} participants...", file=sys.stderr)
    all_examples = []
    for i, pid in enumerate(train_pids_sample):
        for seg in train_series.get(pid, []):
            all_examples.extend(build_examples(seg, TRAIN_STRIDE_BINS))
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(train_pids_sample)}, {len(all_examples)} examples", file=sys.stderr)
    print(f"total training examples: {len(all_examples)}", file=sys.stderr)
    if len(all_examples) == 0:
        print("NO TRAINING EXAMPLES -- aborting", file=sys.stderr)
        sys.exit(1)

    n_val = max(1, int(0.1 * len(all_examples)))
    rng.shuffle(all_examples)
    val_examples, train_examples = all_examples[:n_val], all_examples[n_val:]

    # FIX 2026-09-09: this script had NO normalization at all -- raw glucose
    # (40-400 mg/dL) went directly into the LSTM. Confirmed this broke
    # training (val MAE started at 121 and was still dropping ~1.1/epoch when
    # the 15-epoch cap hit). Matches lstm_zeroshot_baseline.py's exact
    # pattern: normalize context only (causal, fit on train only), leave
    # targets in raw units so the model learns to output raw-scale values
    # directly -- the verified, working convention from the disjoint script.
    all_ctx = np.concatenate([e[0] for e in train_examples])
    mu, sigma = float(all_ctx.mean()), float(all_ctx.std())
    print(f"normalization: mu={mu:.2f}, sigma={sigma:.2f}", file=sys.stderr)

    def to_tensors(examples):
        ctx = torch.tensor(np.stack([e[0] for e in examples]))
        ctx = (ctx - mu) / sigma
        tgt = torch.tensor(np.stack([e[1] for e in examples]))
        return ctx.to(device), tgt.to(device)

    train_ctx, train_tgt = to_tensors(train_examples)
    val_ctx, val_tgt = to_tensors(val_examples)

    model = SimpleLSTM().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.L1Loss()

    best_val, best_state, bad_epochs = float("inf"), None, 0
    n = train_ctx.shape[0]
    for epoch in range(1 if SMOKE else MAX_EPOCHS):
        model.train()
        perm = torch.randperm(n, device=device)
        for i in range(0, n, BATCH_SIZE):
            idx = perm[i:i + BATCH_SIZE]
            opt.zero_grad()
            pred = model(train_ctx[idx])
            loss = loss_fn(pred, train_tgt[idx])
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            val_pred = model(val_ctx)
            val_loss = loss_fn(val_pred, val_tgt).item()
        print(f"epoch {epoch}: val_mae={val_loss:.4f}", file=sys.stderr)
        if val_loss < best_val:
            best_val, best_state, bad_epochs = val_loss, {k: v.clone() for k, v in model.state_dict().items()}, 0
        else:
            bad_epochs += 1
            if bad_epochs >= PATIENCE:
                print(f"early stop at epoch {epoch}", file=sys.stderr)
                break
    model.load_state_dict(best_state)
    print(f"best val_mae={best_val:.4f}", file=sys.stderr)

    print("scoring on proportional test participants (held-out later timeframe)...", file=sys.stderr)
    model.eval()
    rows = []
    persist_rows = []
    for pid, segs in test_series.items():
        exs = []
        for seg in segs:
            exs.extend(build_examples(seg, TEST_STRIDE_BINS))
        if not exs:
            continue
        ctx_raw = torch.tensor(np.stack([e[0] for e in exs])).to(device)
        tgt = np.stack([e[1] for e in exs])
        anchor = ctx_raw[:, -1].cpu().numpy()  # raw units, for the persistence baseline
        ctx = (ctx_raw - mu) / sigma  # normalized, for the model -- same mu/sigma fit on train only
        with torch.no_grad():
            pred = model(ctx).cpu().numpy()
        for i in range(len(exs)):
            for h, hb in HORIZON_BINS.items():
                rows.append({"person_id": pid, "horizon": h,
                             "pred": float(pred[i, hb - 1]), "actual": float(tgt[i, hb - 1])})
                persist_rows.append({"person_id": pid, "horizon": h,
                                      "pred": float(anchor[i]), "actual": float(tgt[i, hb - 1])})

    pred_df = pd.DataFrame(rows)
    pers_df = pd.DataFrame(persist_rows)
    pred_df.to_csv(f"{OUTPUT_DIR}/predictions.csv", index=False)

    out_rows = []
    for h in HORIZON_BINS:
        m = pred_df[pred_df["horizon"] == h]
        p = pers_df[pers_df["horizon"] == h]
        lstm_mae = (m["pred"] - m["actual"]).abs()
        pers_mae = (p["pred"] - p["actual"]).abs()
        per_person_lstm = m.assign(err=lstm_mae).groupby("person_id")["err"].mean()
        per_person_pers = p.assign(err=pers_mae).groupby("person_id")["err"].mean()
        common = per_person_lstm.index.intersection(per_person_pers.index)
        d = per_person_lstm.loc[common] - per_person_pers.loc[common]
        t_stat, p_val = stats.ttest_1samp(d.to_numpy(), 0.0)
        out_rows.append({
            "horizon": h, "lstm_mae": lstm_mae.mean(), "persistence_mae": pers_mae.mean(),
            "n_bins": len(m), "n_participants": len(common),
            "cluster_p_lstm_vs_persistence": p_val, "cluster_t": t_stat,
        })
        print(f"{h}: LSTM={lstm_mae.mean():.4f} persistence={pers_mae.mean():.4f} "
              f"n_participants={len(common)} p={p_val:.4g}", file=sys.stderr)

    pd.DataFrame(out_rows).to_csv(f"{OUTPUT_DIR}/summary.csv", index=False)
    print("LSTM_PROPORTIONAL_BASELINE_OK", file=sys.stderr)


if __name__ == "__main__":
    main()
