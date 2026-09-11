"""Per-participant personalization for the LSTM zero-shot baseline
(code/baselines/lstm_zeroshot_baseline.py, MAE 7.375/11.116/15.043 at 30min/1h/2h).

Reuses the EXACT SAME causal anchor/eligibility construction as every other
personalization arm in this project (model_input/personalization/
eval_set_test.csv): per participant, one fixed anchor T; k=1/3/7-day
fine-tuning history ends at T; evaluation windows are the same eligible set
(context_start >= T) for every k, matching Chronos-2/TimesFM's convention
that "all evaluation conditions used the same eligible windows beginning at
T or later." Window_id in eval_set_test.csv is directly joinable against
model_input/zeroshot_eval/zeroshot_{h}_glucose_only.csv (same convention
verified by inspection -- both use "{person_id}_{n}" ids).

Fine-tuning: start from the pretrained zero-shot LSTM's weights (loaded from
lstm_zeroshot_results/lstm_model.pt), a small number of additional Adam
steps on that participant's own k-day history, re-using the exact same
feature construction (raw glucose context, MAE/L1 loss, same normalization
mu/sigma -- recomputed deterministically from the same seed/train-pool the
zero-shot script used, not re-estimated, so scores stay on the same scale).

Negative control (SSMCGM_PERS_SHUFFLE convention, matching
chronos2_personalization.py's assign_donors): for k=7d, fine-tune on a
different, rotated donor participant's history instead of the real one.
Real personalization should beat this; if it doesn't, the "improvement"
would be an artifact of extra gradient steps alone, not real adaptation.
"""
import copy
import os
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy import stats

REPO_ROOT = "/path/to/aireadi-collab"
sys.path.insert(0, os.path.join(REPO_ROOT, "code"))
# Sibling modules live in adjacent topic directories under code/.
_CODE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _d in ("analysis", "baselines", "cgmlsm", "chronos2", "meal_detection", "model_input", "ssmcgm", "timesfm"):
    sys.path.insert(0, os.path.join(_CODE, _d))
sys.path.insert(0, os.path.join(REPO_ROOT, "code/model_input"))
from build_participant_grid import load_cleaned_glucose_binned  # noqa: E402
from lstm_zeroshot_baseline import (  # noqa: E402
    SimpleLSTM, CONTEXT_LEN, MAX_HORIZON, HORIZON_BINS, N_TRAIN_PARTICIPANTS,
    TRAIN_STRIDE_BINS, SEED, build_train_examples,
)

EVAL_SET_PATH = f"{REPO_ROOT}/model_input/personalization/eval_set_test.csv"
ZEROSHOT_EVAL_DIR = f"{REPO_ROOT}/model_input/zeroshot_eval"
MODEL_PATH = f"{REPO_ROOT}/lstm_zeroshot_results/lstm_model.pt"
CHRONOS2_PERS_PATH_TEMPLATE = (
    f"{REPO_ROOT}/chronos2_personalization_results/test/p7d/{{fit}}/predictions_{{h}}_glucose_only.csv"
)
CHRONOS2_FIT_FOR_HORIZON = {"30min": "fit6", "1h": "fit24", "2h": "fit24", "3h": "fit24", "4h": "fit24"}
OUTPUT_DIR = f"{REPO_ROOT}/lstm_personalization_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

SMOKE = os.environ.get("LSTM_PERS_SMOKE", "0") == "1"
FT_STEPS = int(os.environ.get("LSTM_PERS_FT_STEPS", "30"))
FT_LR = float(os.environ.get("LSTM_PERS_FT_LR", "1e-4"))
SHUFFLE_DONOR = os.environ.get("LSTM_PERS_SHUFFLE", "0") == "1"
K_VALUES = [1, 3, 7] if not SHUFFLE_DONOR else [7]

device = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(SEED)


def recompute_norm_stats():
    """Exact same deterministic recomputation the zero-shot script used --
    same seed, same train pool, same build_train_examples -- so mu/sigma
    match the already-trained model's own normalization exactly."""
    widx = pd.read_csv(f"{REPO_ROOT}/model_input/window_index.csv", dtype={"person_id": str})
    train_pids_all = sorted(widx.loc[widx["recommended_split"] == "train", "person_id"].unique())
    rng = np.random.default_rng(SEED)
    sample_pids = rng.choice(train_pids_all, size=min(N_TRAIN_PARTICIPANTS, len(train_pids_all)), replace=False)
    all_ctx = []
    for pid in sample_pids:
        for ctx, _ in build_train_examples(pid):
            all_ctx.append(ctx)
    all_ctx = np.concatenate(all_ctx)
    return float(all_ctx.mean()), float(all_ctx.std())


def assign_donors(person_ids, seed):
    rng = np.random.default_rng(seed)
    shuffled = list(person_ids)
    rng.shuffle(shuffled)
    rotated = shuffled[1:] + shuffled[:1]
    donors = dict(zip(shuffled, rotated))
    assert all(k != v for k, v in donors.items()), "donor assignment has a fixed point"
    return donors


def build_finetune_examples(pid, history_start, anchor):
    s = load_cleaned_glucose_binned(pid)
    if s is None:
        return []
    s = s[(s.index >= history_start) & (s.index < anchor)]
    if len(s) < CONTEXT_LEN + MAX_HORIZON + 5:
        return []
    vals = s.to_numpy()
    stride = 1 if len(vals) < 400 else 3  # scarce k=1d history needs a finer stride
    examples = []
    for t0 in range(0, len(vals) - CONTEXT_LEN - MAX_HORIZON, stride):
        ctx = vals[t0:t0 + CONTEXT_LEN]
        tgt = vals[t0 + CONTEXT_LEN:t0 + CONTEXT_LEN + MAX_HORIZON]
        if np.isnan(ctx).any() or np.isnan(tgt).any():
            continue
        examples.append((ctx.astype(np.float32), tgt.astype(np.float32)))
    return examples


def finetune_one(base_state, examples, mu, sigma):
    model = SimpleLSTM().to(device)
    model.load_state_dict(base_state)
    opt = torch.optim.Adam(model.parameters(), lr=FT_LR)
    loss_fn = nn.L1Loss()
    ctx = torch.from_numpy((np.stack([e[0] for e in examples]) - mu) / sigma)
    tgt = torch.from_numpy((np.stack([e[1] for e in examples]) - mu) / sigma)
    n = ctx.shape[0]
    model.train()
    for step in range(FT_STEPS):
        idx = torch.randint(0, n, (min(32, n),))
        xb, yb = ctx[idx].to(device), tgt[idx].to(device)
        opt.zero_grad()
        pred = model(xb)
        loss = loss_fn(pred, yb)
        loss.backward()
        opt.step()
    model.eval()
    return model


def main():
    print(f"device={device}, SMOKE={SMOKE}, SHUFFLE_DONOR={SHUFFLE_DONOR}, K_VALUES={K_VALUES}", file=sys.stderr)
    mu, sigma = recompute_norm_stats()
    print(f"recomputed normalization: mu={mu:.4f}, sigma={sigma:.4f}", file=sys.stderr)

    base_state = torch.load(MODEL_PATH, map_location=device)

    eval_set = pd.read_csv(EVAL_SET_PATH, dtype={"person_id": str})
    for col in ["context_start", "anchor", "history_start_1d", "history_start_3d", "history_start_7d"]:
        eval_set[col] = pd.to_datetime(eval_set[col], utc=True)

    all_pids = sorted(eval_set["person_id"].unique())
    if SMOKE:
        all_pids = all_pids[:8]
        eval_set = eval_set[eval_set["person_id"].isin(all_pids)]

    donors = assign_donors(all_pids, SEED) if SHUFFLE_DONOR else None

    summary_rows, sig_rows = [], []
    for k in K_VALUES:
        hist_col = f"history_start_{k}d"
        # anchor is identical across horizons for a given person -- take one row per person
        per_person = eval_set.drop_duplicates("person_id")[["person_id", "anchor", hist_col]]

        finetuned = {}
        n_fit_ok, n_fit_skipped = 0, 0
        per_person_idx = per_person.set_index("person_id")
        for _, row in per_person.iterrows():
            pid = row["person_id"]
            src_pid = donors[pid] if donors else pid
            if donors:
                # the donor's OWN history window, not the recipient's -- matching
                # chronos2_personalization.py's convention exactly: "someone
                # else's week," not "someone else's data sampled over my
                # calendar dates," which for a different participant might not
                # even be inside their own record.
                src_row = per_person_idx.loc[src_pid]
                hist_start, anchor_val = src_row[hist_col], src_row["anchor"]
            else:
                hist_start, anchor_val = row[hist_col], row["anchor"]
            examples = build_finetune_examples(src_pid, hist_start, anchor_val)
            if len(examples) < 5:
                n_fit_skipped += 1
                continue
            finetuned[pid] = finetune_one(base_state, examples, mu, sigma)
            n_fit_ok += 1
        print(f"k={k}d: fit {n_fit_ok} participants, skipped {n_fit_skipped} (insufficient history)", file=sys.stderr)

        for horizon_name in HORIZON_BINS:
            path = f"{ZEROSHOT_EVAL_DIR}/zeroshot_{horizon_name}_glucose_only.csv"
            df = pd.read_csv(path, dtype={"window_id": str, "person_id": str})
            df["bin_start"] = pd.to_datetime(df["bin_start"], utc=True)
            df["is_context"] = df["is_context"].astype(bool)

            elig_wids = set(eval_set.loc[eval_set["horizon"] == horizon_name, "window_id"])
            df = df[df["window_id"].isin(elig_wids)]
            if df.empty:
                continue

            rows = []
            for wid, g in df.groupby("window_id", sort=False):
                pid = g["person_id"].iloc[0]
                if pid not in finetuned:
                    continue
                g = g.sort_values("bin_start")
                ctx = g[g["is_context"]]
                hor = g[~g["is_context"]]
                if len(ctx) != CONTEXT_LEN or hor.empty:
                    continue
                ctx_vals = ctx["cgm_glucose"].to_numpy()
                if np.isnan(ctx_vals).any():
                    continue
                x = torch.from_numpy(((ctx_vals.astype(np.float32) - mu) / sigma)[None, :]).to(device)
                with torch.no_grad():
                    pred = finetuned[pid](x).cpu().numpy()[0] * sigma + mu
                hor_vals = hor["cgm_glucose"].to_numpy()
                for h in range(len(hor_vals)):
                    if np.isnan(hor_vals[h]):
                        continue
                    rows.append({"window_id": wid, "person_id": pid, "bin_start": hor["bin_start"].iloc[h],
                                 "actual": hor_vals[h], "predictions": float(pred[h])})
            if not rows:
                print(f"  k={k}d {horizon_name}: NO SCORABLE WINDOWS", file=sys.stderr)
                continue
            pred_df = pd.DataFrame(rows)
            pred_df["abs_error"] = (pred_df["predictions"] - pred_df["actual"]).abs()
            tag = f"k{k}d" + ("_shuffled" if SHUFFLE_DONOR else "")
            pred_df.to_csv(f"{OUTPUT_DIR}/predictions_{tag}_{horizon_name}.csv", index=False)

            mae = pred_df["abs_error"].mean()
            rmse = float(np.sqrt((pred_df["abs_error"] ** 2).mean()))
            n_participants = pred_df["person_id"].nunique()
            summary_rows.append({"k": k, "shuffled": SHUFFLE_DONOR, "horizon": horizon_name,
                                  "mae": round(float(mae), 4), "rmse": round(rmse, 4),
                                  "n_bins": len(pred_df), "n_participants": n_participants})
            print(f"  k={k}d{'(shuffled)' if SHUFFLE_DONOR else ''} {horizon_name}: "
                  f"n={len(pred_df)}, participants={n_participants}, MAE={mae:.4f}", file=sys.stderr)

            if not SHUFFLE_DONOR:
                # vs LSTM's own zero-shot (same person/window, real predictions on disk)
                zs = pd.read_csv(f"{REPO_ROOT}/lstm_zeroshot_results/predictions_{horizon_name}.csv",
                                  dtype={"window_id": str})
                zs["bin_start"] = pd.to_datetime(zs["bin_start"], utc=True)
                m = pred_df.merge(zs[["window_id", "bin_start", "predictions"]].rename(
                    columns={"predictions": "zs_pred"}), on=["window_id", "bin_start"], how="inner")
                m["pers_err"] = (m["predictions"] - m["actual"]).abs()
                m["zs_err"] = (m["zs_pred"] - m["actual"]).abs()

                # vs Chronos-2 personalized best (p7d, test)
                try:
                    c2p_path = CHRONOS2_PERS_PATH_TEMPLATE.format(
                        h=horizon_name, fit=CHRONOS2_FIT_FOR_HORIZON[horizon_name])
                    c2p = pd.read_csv(c2p_path, dtype={"window_id": str})
                    c2p["bin_start"] = pd.to_datetime(c2p["bin_start"], utc=True)
                    m2 = pred_df.merge(c2p[["window_id", "bin_start", "predictions"]].rename(
                        columns={"predictions": "c2p_pred"}), on=["window_id", "bin_start"], how="inner")
                    m2["pers_err"] = (m2["predictions"] - m2["actual"]).abs()
                    m2["c2p_err"] = (m2["c2p_pred"] - m2["actual"]).abs()
                except FileNotFoundError:
                    m2 = None

                for ref_name, mm, a_col, b_col in (
                    [("lstm_zeroshot", m, "pers_err", "zs_err")]
                    + ([("chronos2_personalized_p7d", m2, "pers_err", "c2p_err")] if m2 is not None else [])
                ):
                    d = mm[a_col] - mm[b_col]
                    pp = d.groupby(mm["person_id"]).mean()
                    t_stat, p_val = stats.ttest_1samp(pp.to_numpy(), 0.0)
                    sig_rows.append({
                        "k": k, "horizon": horizon_name, "vs": ref_name,
                        "n_matched_bins": len(mm), "n_participants_clustered": len(pp),
                        "lstm_pers_mae_matched": round(float(mm[a_col].mean()), 4),
                        "ref_mae_matched": round(float(mm[b_col].mean()), 4),
                        "cluster_t": round(float(t_stat), 4), "cluster_p": float(p_val),
                        "lstm_pers_better": bool(mm[a_col].mean() < mm[b_col].mean()),
                    })

    pd.DataFrame(summary_rows).to_csv(
        f"{OUTPUT_DIR}/summary{'_shuffled' if SHUFFLE_DONOR else ''}.csv", index=False)
    if sig_rows:
        sig_df = pd.DataFrame(sig_rows)
        sig_df.to_csv(f"{OUTPUT_DIR}/significance.csv", index=False)
        print(sig_df.to_string(index=False), file=sys.stderr)
    print("LSTM_PERSONALIZATION_OK", file=sys.stderr)


if __name__ == "__main__":
    main()
