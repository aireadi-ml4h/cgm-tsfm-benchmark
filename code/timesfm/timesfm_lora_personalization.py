"""TimesFM 2.5 LoRA personalization: per-participant fine-tuning on a held-out
test participant's own 1/3/7 days of prior glucose history, glucose-only,
mirroring code/chronos2/chronos2_personalization.py's design (same eval set, same
causal anchor, same isolation -- one LoRA adapter per participant, fit on
only that person's own data, base model reused untouched for the next
person) but using TimesFM's real LoRA mechanism instead of Chronos-2's
pipeline.fit().

WHY THIS IS A DIFFERENT QUESTION FROM THE IN-CONTEXT ARM
===========================================================
code/timesfm/timesfm_incontext_personalization.py already tests "does showing the
model more of this person's own data help, with zero weight updates." This
tests the other half: "does actually updating the model's weights for this
person help, and by how much more or less than just showing it the data."
Same population, same eval windows, same k values -- only the adaptation
mechanism differs, by design, so the two results are directly comparable.

GLUCOSE-ONLY, SAME SCOPE AS CHRONOS-2'S PHASE 1
==================================================
TimesFM's trainable HF class (TimesFm2_5ModelForPrediction) has no covariate
input at all -- see this project's HANDOFF.md 2026-09-02 section for the
full architectural reasoning. LoRA here can only ever adapt the model's use
of the raw glucose sequence, never wearable covariates.

TRAINING WINDOWS SAMPLED FROM THE HISTORY SLICE, NOT THE HISTORY ITSELF
==========================================================================
A person's 1/3/7-day history_array() (144-2016 real+NaN bins) is not itself
one training example -- it is the POOL random (context, target) windows are
sampled from, context_len=144 (this project's standard, matching every
other model here) and fit_horizon=48 (4h), following the vendored
finetune_lora.py reference example's own random-window-sampling pattern
(TimeSeriesRandomWindowDataset). NOT horizon-matched per eval horizon (a
real simplification relative to Stephanie's own, more refined sweep for
Chronos-2, which found fit-length matters a lot -- disclosed here, not
silently adopted as equivalent).

NaN IN THE HISTORY: windows containing any NaN in context or target are
skipped during window sampling (TimesFM's plain forward() with
future_values requires a real loss target; unlike the zero-shot glucose_only
path, which tolerates raw NaN, THIS path computes an MSE loss against
future_values, so a NaN target would poison it) -- checked directly against
the reference script (which assumes clean retail data, no NaN handling at
all) rather than assumed safe to skip.
"""
import os
import sys
import time

import numpy as np
import pandas as pd
import torch

REPO_ROOT = "/path/to/aireadi-collab"
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
# Sibling modules live in adjacent topic directories under code/.
_CODE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _d in ("analysis", "baselines", "cgmlsm", "chronos2", "meal_detection", "model_input", "ssmcgm", "timesfm"):
    sys.path.insert(0, os.path.join(_CODE, _d))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "model_input"))
from build_participant_grid import load_cleaned_glucose_binned  # noqa: E402

EVAL_CONTEXT_BINS = 144  # the 144-bin context every zeroshot_eval CSV window actually has
# CONTEXT_BINS/MODEL_CONTEXT_LEN: TimesFM's trainable HF class
# (TimesFm2_5ModelForPrediction) patches the input internally with
# patch_len=32 and requires past_values' length to be an exact multiple of
# it -- confirmed directly (RuntimeError: shape '[16, -1, 32]' is invalid for
# input of size 2304, i.e. 144 is not a multiple of 32).
#
# CORRECTED 2026-09-05: an earlier version of this script worked around that
# by truncating to 128 bins (discarding 16 of the real 144), at both fit and
# eval time -- an unforced, avoidable data loss. TimesFm2_5ModelForPrediction.
# forward() takes an explicit `forecast_context_len` and, when the real input
# is SHORTER than that value, its own _preprocess() automatically front-pads
# with zeros and builds a matching padding mask (past_values_padding) that
# the base model correctly excludes from its running mean/std normalization
# (per-timestep) -- i.e. real, first-class padding support, not a hack.
# Verified empirically (code/test_timesfm_padding.py, job 42217956, GPU):
# passing the full 144 real bins with forecast_context_len=160 (160 = next
# multiple of 32 >= 144) runs cleanly, produces no NaN, and gradients reach
# 258/258 LoRA parameters within 2 training steps -- confirmed by checking
# grads BEFORE optimizer.zero_grad() each step, after an initial version of
# this same check read a false "0/258" by checking AFTER zero_grad().
# CONTEXT_BINS is now the real amount of history used (144, matching every
# other model/arm in this project); MODEL_CONTEXT_LEN is the padded length
# passed to forecast_context_len so the patch-length constraint is satisfied
# without discarding any real data.
CONTEXT_BINS = 144
MODEL_CONTEXT_LEN = 160
FIT_HORIZON = 48  # 4h, matches Chronos-2's original (pre-sweep) simpler choice -- see module docstring
BIN_MINUTES = 5
HORIZON_BINS = {"30min": 6, "1h": 12, "2h": 24, "3h": 36, "4h": 48}
CONDITION = "glucose_only"

SPLIT = os.environ.get("TIMESFM_LORA_SPLIT", "val")
EVAL_SET_PATH = f"{REPO_ROOT}/model_input/personalization/eval_set_{SPLIT}.csv"
EVAL_DIR = (f"{REPO_ROOT}/model_input/zeroshot_eval" if SPLIT == "test"
            else f"{REPO_ROOT}/model_input/zeroshot_eval_{SPLIT}")

HISTORY_DAYS = int(os.environ.get("TIMESFM_LORA_HISTORY_DAYS", "7"))
LORA_R = int(os.environ.get("TIMESFM_LORA_R", "8"))
LORA_ALPHA = int(os.environ.get("TIMESFM_LORA_ALPHA", "16"))
LEARNING_RATE = float(os.environ.get("TIMESFM_LORA_LR", "1e-4"))
NUM_STEPS = int(os.environ.get("TIMESFM_LORA_NUM_STEPS", "100"))
BATCH_SIZE = int(os.environ.get("TIMESFM_LORA_BATCH_SIZE", "16"))
SEED = int(os.environ.get("TIMESFM_LORA_SEED", "42"))

SHARD = int(os.environ.get("TIMESFM_LORA_SHARD", "0"))
NUM_SHARDS = int(os.environ.get("TIMESFM_LORA_NUM_SHARDS", "1"))
MAX_PARTICIPANTS = os.environ.get("TIMESFM_LORA_MAX_PARTICIPANTS")
MAX_PARTICIPANTS = int(MAX_PARTICIPANTS) if MAX_PARTICIPANTS else None

OUTPUT_DIR = os.environ.get(
    "TIMESFM_LORA_RESULTS_DIR",
    f"{REPO_ROOT}/timesfm_personalization_results/{SPLIT}/lora_{HISTORY_DAYS}d",
)


def history_array(person_id, history_start, anchor):
    """Copy of chronos2_personalization.py's function -- see
    timesfm_incontext_personalization.py's module docstring for why this is
    copied, not imported (avoids a cross-venv Chronos-2 dependency)."""
    s = load_cleaned_glucose_binned(person_id)
    if s is None:
        return None
    s = s[(s.index >= history_start) & (s.index < anchor)]
    if s.empty:
        return None
    full = pd.date_range(history_start, anchor - pd.Timedelta(minutes=BIN_MINUTES),
                          freq=f"{BIN_MINUTES}min", tz="UTC")
    s = s.reindex(full)
    return s.to_numpy(dtype=np.float32)


def sample_windows(values, num_samples, rng):
    """Random (context, target) windows for LoRA fine-tuning, both halves
    NaN-free. Returns lists of numpy arrays."""
    min_len = CONTEXT_BINS + FIT_HORIZON
    if len(values) < min_len:
        return [], []
    valid_starts = []
    for s in range(0, len(values) - min_len + 1):
        window = values[s:s + min_len]
        if not np.isnan(window).any():
            valid_starts.append(s)
    if not valid_starts:
        return [], []
    contexts, targets = [], []
    for _ in range(num_samples):
        s = rng.choice(valid_starts)
        contexts.append(values[s:s + CONTEXT_BINS])
        targets.append(values[s + CONTEXT_BINS:s + CONTEXT_BINS + FIT_HORIZON])
    return contexts, targets


def finetune_for_participant(base_model_id, values, device, rng):
    """One LoRA adapter fit on one participant's history. Returns a fresh
    PEFT-wrapped model; base weights untouched (loaded fresh each call, same
    isolation guarantee chronos2_personalization.py documents for its own
    .fit() call)."""
    from peft import LoraConfig, get_peft_model
    from transformers import TimesFm2_5ModelForPrediction

    contexts, targets = sample_windows(values, NUM_STEPS * BATCH_SIZE, rng)
    if not contexts:
        return None, 0.0

    model = TimesFm2_5ModelForPrediction.from_pretrained(base_model_id, torch_dtype=torch.float32).to(device)
    lora_config = LoraConfig(r=LORA_R, lora_alpha=LORA_ALPHA, target_modules="all-linear", lora_dropout=0.05, bias="none")
    model = get_peft_model(model, lora_config)
    model.train()

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)

    fit_start = time.time()
    n = len(contexts)
    for step in range(NUM_STEPS):
        idx = [((step * BATCH_SIZE) + i) % n for i in range(BATCH_SIZE)]
        ctx = torch.tensor(np.stack([contexts[i] for i in idx]), dtype=torch.float32, device=device)
        tgt = torch.tensor(np.stack([targets[i] for i in idx]), dtype=torch.float32, device=device)
        outputs = model(past_values=ctx, future_values=tgt, forecast_context_len=MODEL_CONTEXT_LEN)
        loss = outputs.loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        optimizer.zero_grad()
    fit_seconds = time.time() - fit_start
    model.eval()
    return model, fit_seconds


def predict_participant(model, device, export_by_horizon, window_ids_by_horizon, person_id):
    out = {}
    for horizon, horizon_bins in HORIZON_BINS.items():
        wanted = window_ids_by_horizon.get((person_id, horizon))
        if not wanted:
            continue
        df = export_by_horizon[horizon]
        df = df[df["window_id"].isin(wanted)]
        if df.empty:
            continue
        rows = []
        for wid, wdf in df.groupby("window_id"):
            wdf = wdf.sort_values("bin_start")
            ctx = wdf.loc[wdf["is_context"], "cgm_glucose"].to_numpy(dtype=np.float32)
            tgt = wdf.loc[~wdf["is_context"], ["bin_start", "cgm_glucose"]].reset_index(drop=True)
            if len(ctx) != EVAL_CONTEXT_BINS or np.isnan(ctx).any():
                continue
            # Full 144 real bins, padded to MODEL_CONTEXT_LEN (160) by the
            # model's own forecast_context_len mechanism -- see CONTEXT_BINS
            # comment above. No truncation: this now uses all the real
            # context, matching every other model/arm in this project.
            with torch.no_grad():
                inp = torch.tensor(ctx, dtype=torch.float32, device=device).unsqueeze(0)
                pred = model(past_values=inp, forecast_context_len=MODEL_CONTEXT_LEN).mean_predictions[0, :horizon_bins].float().cpu().numpy()
            merged = tgt.rename(columns={"cgm_glucose": "actual"}).copy()
            merged["window_id"] = wid
            merged["predictions"] = pred
            rows.append(merged)
        if rows:
            merged_df = pd.concat(rows, ignore_index=True)
            merged_df["abs_error"] = (merged_df["predictions"] - merged_df["actual"]).abs()
            out[horizon] = merged_df
    return out


def load_export(horizon, keep_window_ids):
    path = f"{EVAL_DIR}/zeroshot_{horizon}_{CONDITION}.csv"
    df = pd.read_csv(path, dtype={"window_id": str, "person_id": str})
    df = df[df["window_id"].isin(keep_window_ids)].copy()
    df["bin_start"] = pd.to_datetime(df["bin_start"], utc=True)
    df["is_context"] = df["is_context"].astype(bool)
    return df


def done_participants(path):
    if not os.path.exists(path):
        return set()
    return set(pd.read_csv(path, dtype={"person_id": str})["person_id"])


def main():
    job_start = time.time()
    e = pd.read_csv(EVAL_SET_PATH, dtype={"person_id": str, "window_id": str})
    e["anchor"] = pd.to_datetime(e["anchor"], utc=True)
    history_col = f"history_start_{HISTORY_DAYS}d"
    e[history_col] = pd.to_datetime(e[history_col], utc=True)

    window_to_person = dict(zip(e["window_id"], e["person_id"]))
    window_ids_by_horizon = e.groupby(["person_id", "horizon"])["window_id"].apply(set).to_dict()
    anchors = e.groupby("person_id").agg(anchor=("anchor", "first"), history_start=(history_col, "first"))

    all_participants = sorted(anchors.index)
    participants = [p for i, p in enumerate(all_participants) if i % NUM_SHARDS == SHARD]
    if MAX_PARTICIPANTS is not None:
        participants = participants[:MAX_PARTICIPANTS]

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    done_path = f"{OUTPUT_DIR}/participants_done_shard{SHARD}.csv"
    done = done_participants(done_path)
    todo = [p for p in participants if p not in done]

    keep_ids = set(e.loc[e["person_id"].isin(participants), "window_id"])
    export_by_horizon = {h: load_export(h, keep_ids) for h in HORIZON_BINS}
    print(f"shard {SHARD}/{NUM_SHARDS}: {len(participants)} participants ({len(todo)} to do), "
          f"split={SPLIT}, k={HISTORY_DAYS}d, lr={LEARNING_RATE}, steps={NUM_STEPS}", file=sys.stderr)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    base_model_id = "google/timesfm-2.5-200m-transformers"
    rng = np.random.default_rng(SEED)

    n_fitted, fit_seconds_total, skipped = 0, 0.0, []
    for i, pid in enumerate(todo):
        anchor, history_start = anchors.loc[pid, "anchor"], anchors.loc[pid, "history_start"]
        values = history_array(pid, history_start, anchor)
        if values is None:
            skipped.append(pid)
            continue

        model, fit_seconds = finetune_for_participant(base_model_id, values, device, rng)
        if model is None:
            skipped.append(pid)
            continue
        fit_seconds_total += fit_seconds
        n_fitted += 1

        preds = predict_participant(model, device, export_by_horizon, window_ids_by_horizon, pid)
        for horizon, merged in preds.items():
            path = f"{OUTPUT_DIR}/predictions_{horizon}_{CONDITION}.csv"
            merged.to_csv(path, mode="a", header=not os.path.exists(path), index=False)

        pd.DataFrame([{"person_id": pid, "fit_seconds": round(fit_seconds, 1)}]).to_csv(
            done_path, mode="a", header=not os.path.exists(done_path), index=False)

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        if (i + 1) % 10 == 0 or i + 1 == len(todo):
            print(f"  {i + 1}/{len(todo)} participants, {fit_seconds_total / max(n_fitted, 1):.1f}s/fit avg, "
                  f"{time.time() - job_start:.0f}s elapsed", file=sys.stderr)

    if skipped:
        print(f"WARNING: skipped {len(skipped)} participants (too little history): {skipped[:5]}", file=sys.stderr)

    rows = []
    for horizon in HORIZON_BINS:
        path = f"{OUTPUT_DIR}/predictions_{horizon}_{CONDITION}.csv"
        if not os.path.exists(path):
            continue
        full = pd.read_csv(path, dtype={"window_id": str})
        scored = full.dropna(subset=["actual"])
        rows.append({
            "horizon": horizon, "split": SPLIT, "history_days": HISTORY_DAYS,
            "lora_r": LORA_R, "learning_rate": LEARNING_RATE, "num_steps": NUM_STEPS,
            "mae_mgdl": scored["abs_error"].mean(),
            "rmse_mgdl": float(np.sqrt((scored["abs_error"] ** 2).mean())),
            "n_scored": len(scored), "n_total": len(full), "n_windows": full["window_id"].nunique(),
            "n_participants_this_shard": n_fitted,
            "mean_fit_seconds": round(fit_seconds_total / max(n_fitted, 1), 1),
            "cumulative_job_seconds": round(time.time() - job_start, 1),
        })
    if rows:
        summary = pd.DataFrame(rows)
        summary.to_csv(f"{OUTPUT_DIR}/summary_shard{SHARD}.csv", index=False)
        print(summary.to_string(index=False), file=sys.stderr)


if __name__ == "__main__":
    main()
