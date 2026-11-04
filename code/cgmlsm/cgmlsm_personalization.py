"""CGM-LSM personalization: isolated per-participant fine-tuning of the settled
CGM-LSM checkpoint on that participant's own preceding k days of glucose, then
forecasting their eval windows -- the CGM-LSM arm of RQ2, mirroring
code/chronos2/chronos2_personalization.py (Chronos-2 per-participant LoRA),
code/timesfm/timesfm_lora_personalization.py, and
code/ssmcgm/ssmcgm_personalization_finetune.py (SSM-CGM shared-weight
fine-tune). Same personalization eval set, same anchors, same k=1/3/7-day
convention, same "one adaptation per person, base model reused untouched for
the next person" isolation guarantee, so the arms are directly comparable.

BASE CHECKPOINT: the T1D-excluded GlucoFM-Bench pretrain fine-tuned on AI-READI
for 2 EPOCHS AT lr=1e-2 ($AIREADI_SCRATCH/aireadi_cgmlsm/finetune_sweep4_ep2_lr1e-2).
Always pass it explicitly via CGMLSM_PERS_BASE_CKPT; the module default is only
a fallback.

This SUPERSEDES the previously settled 1-epoch/lr=5e-6 checkpoint, and the
reason matters for interpreting this arm. Fine-tune sweeps 1-2 only ever
explored lr <= 2e-5 and concluded "gentler is always better"; sweeps 3-5
(2026-09-06) showed that was an artifact of never leaving the low-lr regime.
On val windows the new base scores 10.5/16.6/25.1 mg/dL at 30min/1h/2h against
the old base's 14.4/27.9/51.8. Every axis is bracketed: lr=2e-2 destroys the
model outright (100% PAD, zero scoreable predictions), lr=5e-3 is worse,
1 epoch is worse, 3-4 epochs are worse.

TWO CAVEATS ON THE BASE, both to be carried into any report:
  - SEED SENSITIVITY, KNOWN AND ACCEPTED BY EXPLICIT USER DECISION (2026-09-06).
    lr=1e-2 sits next to that failure boundary and its results scatter across
    seeds -- test-window 2h MAE is 24.5/26.3/30.1/30.6 for seeds 42/2024/7/123,
    and seed 42 is the LUCKIEST of the four (~3.4 mg/dL better at 2h than the
    4-seed mean). SEED 42 IS USED ANYWAY, deliberately: every other model in
    this comparison (Chronos-2, SSM-CGM, TimesFM personalization) also runs a
    single seed 42 and none has a stability check, so switching CGM-LSM alone
    to a multi-seed mean would make it LESS comparable to the others rather
    than more. The cross-model comparison is fair in that every model gets the
    same unmeasured lottery; it is not unbiased. Do not quietly "fix" this for
    CGM-LSM alone -- a project-wide multi-seed check is the open TO-DO
    (see model_input/HANDOFF.md). The base is a FIXED artifact once trained, so
    everything in this script is deterministic on top of it regardless.
  - THE 30MIN GAIN IS NOT ROBUST. Across seeds the new base gives 10.7-15.0 at
    30min against the old base's ~13.5, i.e. better on some seeds and worse on
    others. The 1h/2h improvement holds on every seed by a wide margin.

WHY THE BASE WAS CHANGED AT ALL, since it directly shapes this arm: on the old
base, fitting on a DONOR participant's 7 days reproduced 69% of the apparent
personalization gain at 2h -- most of the "personalization" effect was the
fine-tune finishing a population adaptation the base never got. The stronger
base shrinks the person-specific effect from ~7.7 to ~2 mg/dL at 2h in absolute
terms, which is the honest size of it. Note the donor share did NOT fall
proportionally (still ~60%), so "underfit base" is not the whole explanation
for the donor effect and should not be asserted as if it were.

The from-scratch AI-READI checkpoint remains unsuitable as a base for a
separate reason (model_input/HANDOFF.md "CORRECTION 2026-09-05",
code/analysis/cgmlsm_arm_matched_comparison.py): it emits `UNKNOWN` instead of a number
for 16.2% of 2h horizon bins, those bins are dropped before MAE, and it is an
ABSORBING failure -- of the 1,440 affected 2h windows, all 1,440 never recover,
and in 16,088 of 16,090 cases the ground truth was an ordinary real reading.
Personalization would change how often that collapse happens, moving MAE by
changing the DENOMINATOR rather than forecast quality.

HORIZONS: 30min/1h/2h only, by explicit decision. 3h/4h (36/48 bins) is
impossible for this checkpoint, not merely out of scope: every CGM-LSM
checkpoint has n_positions=168 = CONTEXT_BINS(144) + 24, and model.generate()
needs one learned position per context-plus-generated-token slot, so a 3h
rollout would index past the wpe table -- the exact CUDA
`indexSelectLargeIndex` assertion that already forced the 144->168 retrain
(code/cgmlsm/cgmlsm_train.py). Reaching 3h/4h would require retraining the base with
longer TRAINING chunks (not just a bigger n_positions, which would leave
positions 144-191 randomly initialized and used only at generation time).
Asserted below rather than left to crash mid-job.

NO COVARIATE ARMS. CGM-LSM's architecture takes no wearable covariates at all
(only a tokenized glucose sequence), so RQ2's phase-2 covariate question does
not exist for this model -- there is nothing to add on top of personalization.

NO PER-PERSON PARAMETERS, SO NO CROSS-PERSON LEAKAGE CHANNEL. The one delicate
part of the SSM-CGM arm -- freezing the participant_id embedding so that
adapting to one person cannot corrupt the shared "unknown" row for whoever is
fine-tuned next -- has no analogue here: this model's only inputs are cgm_token
indices (tod_token is not fed to the model, matching the authors' own
run_cgmlsm_pretrain.py), and there are no static covariates. Isolation is
therefore just "reload the checkpoint from disk per (participant, arm)", which
this script does unconditionally.

PARTICIPANTS ARE DISJOINT FROM THIS CHECKPOINT'S TRAINING POPULATION, verified
directly rather than assumed: 0 overlap between the personalization eval
participants and CGM-LSM's 1,347 train / 290 val participants. So "the base
model already saw this person" is not a confound. Re-checked at runtime below
and printed, mirroring ssmcgm_personalization_finetune.py's own sanity print.

ARMS (CGMLSM_PERS_ARMS):
  zs_glucose_only -- k=0, no fine-tuning at all. THIS IS THE BASELINE THIS
      SCRIPT MUST PRODUCE ITSELF; the settled 13.520/28.214/49.291 numbers
      cannot serve as it, because they were scored on cgmlsm_test.csv's
      windows, not the personalization eval set's windows/participants.
  p1d / p3d / p7d -- fine-tuned on that participant's own preceding k days.
  p7d_shuffled -- NEGATIVE CONTROL: fine-tuned on a DONOR participant's 7-day
      history, then scored on this participant's own windows. If p7d beats
      zero-shot only because fine-tuning on any real CGM data helps, this arm
      captures that; if the gain is adaptation to the PERSON, this arm should
      be no better than zero-shot. Donor assignment is deterministic and
      computed from the FULL participant list before sharding, so it does not
      depend on how the job is split across array tasks.

FIT CHUNKS ARE OVERLAPPING, unlike the population training input. A k=1d
history is only ~288 bins; build_cgmlsm_input.py's non-overlapping
CONTEXT_BINS chunking would yield 2 training examples, too few for a
meaningful adaptation step budget. A stride of CGMLSM_PERS_FIT_STRIDE bins
(default 12 = 1h) over the history slice yields ~13 chunks at k=1d and ~157 at
k=7d. Each chunk is still independently coverage-gated with the same
coverage.is_window_eligible rule used everywhere else in this project, and
residual NaN glucose still becomes the vocabulary's own UNKNOWN token, never a
numeric fill.
"""
import os
import sys
import time

import numpy as np
import pandas as pd
import torch

REPO_ROOT = "/path/to/aireadi-collab"
sys.path.append(os.path.join(REPO_ROOT, "code/external/cgmlsm/code/pipeline"))
# Sibling modules live in adjacent topic directories under code/.
_CODE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _d in ("analysis", "baselines", "cgmlsm", "chronos2", "meal_detection", "model_input", "ssmcgm", "timesfm"):
    sys.path.insert(0, os.path.join(_CODE, _d))
sys.path.append(os.path.join(REPO_ROOT, "code/model_input"))

from nn.cgmlsm.modeling_cgmlsm import CgmLsmLMHeadModel  # noqa: E402
from build_participant_grid import build_grid, load_participants_covariates  # noqa: E402
from cgmlsm_vocab import CGM_IDX2TKN, CGM_TKN2IDX, CGM_PAD_IDX, tokenize_glucose  # noqa: E402
from coverage import is_window_eligible  # noqa: E402
from windows import CONTEXT_BINS, HORIZONS_BINS  # noqa: E402

EVAL_HORIZONS = {k: v for k, v in HORIZONS_BINS.items() if k in ("30min", "1h", "2h")}
BIN_MINUTES = 5

BASE_CKPT = os.environ.get(
    "CGMLSM_PERS_BASE_CKPT",
    os.path.join(os.environ["AIREADI_SCRATCH"], "aireadi_cgmlsm", "finetune_sweep2_ep1_lr5e-6", "final"),
)
SPLIT = os.environ.get("CGMLSM_PERS_SPLIT", "test")
EVAL_SET_PATH = os.environ.get(
    "CGMLSM_PERS_EVAL_SET", os.path.join(REPO_ROOT, "model_input", "personalization", f"eval_set_{SPLIT}.csv")
)
OUTPUT_ROOT = os.environ.get(
    "CGMLSM_PERS_OUTPUT_ROOT", os.path.join(REPO_ROOT, "cgmlsm_personalization_results", SPLIT)
)

ARMS = os.environ.get("CGMLSM_PERS_ARMS", "zs_glucose_only,p1d,p3d,p7d,p7d_shuffled").split(",")
# A donor arm exists at EVERY history length, not just 7 days. Selecting the
# fine-tuning recipe on the own-minus-donor increment (the only criterion that
# isolates what personalization specifically buys) requires a donor arm at
# whichever k is being tuned -- so p1d_shuffled and p3d_shuffled are needed as
# well, and the k=1d/3d recipes cannot be tuned without them.
ARM_HISTORY_DAYS = {
    "zs_glucose_only": 0,
    "p1d": 1, "p3d": 3, "p7d": 7,
    "p1d_shuffled": 1, "p3d_shuffled": 3, "p7d_shuffled": 7,
}
DONOR_ARMS = {"p1d_shuffled", "p3d_shuffled", "p7d_shuffled"}

LEARNING_RATE = float(os.environ.get("CGMLSM_PERS_LR", "5e-6"))
NUM_STEPS = int(os.environ.get("CGMLSM_PERS_NUM_STEPS", "50"))
BATCH_SIZE = int(os.environ.get("CGMLSM_PERS_BATCH_SIZE", "8"))
FIT_STRIDE = int(os.environ.get("CGMLSM_PERS_FIT_STRIDE", "12"))
WEIGHT_DECAY = float(os.environ.get("CGMLSM_PERS_WEIGHT_DECAY", "0.01"))
MAX_GRAD_NORM = float(os.environ.get("CGMLSM_PERS_MAX_GRAD_NORM", "1.0"))
SEED = int(os.environ.get("CGMLSM_PERS_SEED", "42"))

SHARD = int(os.environ.get("CGMLSM_PERS_SHARD", "0"))
NUM_SHARDS = int(os.environ.get("CGMLSM_PERS_NUM_SHARDS", "1"))
MAX_PARTICIPANTS = os.environ.get("CGMLSM_PERS_MAX_PARTICIPANTS")
MAX_PARTICIPANTS = int(MAX_PARTICIPANTS) if MAX_PARTICIPANTS else None

GRID_CACHE_MAX = 4  # own grid + donor grid, with slack; grids are per-participant full histories


def token_to_mgdl(token):
    """Inverse of cgmlsm_vocab.tokenize_glucose, identical to cgmlsm_eval.py:
    numeric CGMValue tokens -> float mg/dL, everything else
    (PAD/UNKNOWN/MASK/Other_*) -> NaN, dropped at scoring rather than
    fabricated into a number."""
    return float(token) if token.isdigit() else np.nan


class GridCache:
    def __init__(self, covs):
        self.covs = covs
        self.cache = {}

    def get(self, pid):
        if pid not in self.cache:
            grid = build_grid(pid, self.covs)
            if grid is not None:
                grid = grid.sort_values("bin_start").reset_index(drop=True)
            self.cache[pid] = grid
            if len(self.cache) > GRID_CACHE_MAX:
                self.cache.pop(next(iter(self.cache)))
        return self.cache[pid]


def fit_chunks(grid, history_start, anchor):
    """Overlapping CONTEXT_BINS chunks over [history_start, anchor), each
    independently coverage-gated. Returns an (n_chunks, CONTEXT_BINS) int64
    array of CGMValue indices, or None if nothing qualifies."""
    hist = grid[(grid["bin_start"] >= history_start) & (grid["bin_start"] < anchor)]
    if hist.empty or len(hist) < CONTEXT_BINS:
        return None
    hist = hist.reset_index(drop=True)

    rows = []
    for start in range(0, len(hist) - CONTEXT_BINS + 1, FIT_STRIDE):
        chunk = hist.iloc[start:start + CONTEXT_BINS]
        eligible, _ = is_window_eligible(chunk, ["cgm_glucose"])
        if not eligible:
            continue
        tokens = tokenize_glucose(chunk["cgm_glucose"].to_numpy())
        ids = pd.Series(tokens).astype(str).map(CGM_TKN2IDX)
        if ids.isna().any():
            raise ValueError("tokenize_glucose produced a token outside the CGMValue vocab")
        rows.append(ids.to_numpy(dtype=np.int64))
    if not rows:
        return None
    return np.stack(rows)


def reset_to_base(model, base_state):
    """THE ISOLATION GUARANTEE. Restores every parameter and buffer to the base
    checkpoint's values before each (participant, arm), so no adaptation can
    carry across participants or arms. strict=True so a key the checkpoint does
    not cover would raise rather than silently leave a fine-tuned tensor in
    place, and the restore is VERIFIED against the base tensors rather than
    assumed -- a silent failure here would leak one person's adaptation into
    the next person's scores, which is worse than not personalizing at all."""
    model.load_state_dict(base_state, strict=True)
    for name in ("transformer.wte.weight", "transformer.wpe.weight"):
        if name in base_state:
            current = dict(model.named_parameters()).get(name)
            if current is not None and not torch.equal(
                    current.detach().cpu(), base_state[name].detach().cpu()):
                raise RuntimeError(
                    f"{name} did not reset to the base checkpoint -- cross-person leakage "
                    "risk, aborting rather than scoring a silently-contaminated arm.")
    return model


def finetune(model, base_state, chunk_ids, device):
    """One isolated fine-tune on chunk_ids. Manual AdamW loop rather than
    transformers.Trainer: same objective (labels = input_ids, causal LM loss
    returned by CgmLsmLMHeadModel.forward, exactly as code/cgmlsm/cgmlsm_train.py
    relies on) with none of Trainer's per-instantiation setup cost, which would
    otherwise be paid hundreds of times. The optimizer is constructed fresh per
    call, so no momentum/variance state survives either."""
    reset_to_base(model, base_state)
    model.train()

    optim = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    gen = torch.Generator().manual_seed(SEED)
    n = len(chunk_ids)
    losses = []
    fit_start = time.time()
    for _ in range(NUM_STEPS):
        idx = torch.randint(0, n, (min(BATCH_SIZE, n),), generator=gen).numpy()
        batch = torch.from_numpy(chunk_ids[idx]).to(device)
        out = model(input_ids=batch, labels=batch)
        out.loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
        optim.step()
        optim.zero_grad(set_to_none=True)
        losses.append(float(out.loss.detach()))
    model.eval()
    return model, time.time() - fit_start, float(np.mean(losses[:5])), float(np.mean(losses[-5:]))


def forecast_windows(model, grid, windows, horizon_name, horizon_bins, device):
    """All of one participant's eval windows at one horizon, in a single
    batched generate() call. Mirrors cgmlsm_eval.py's decode path exactly,
    including keeping prediction_token so a collapse into UNKNOWN stays
    visible per arm rather than only showing up as a missing row."""
    ctx_blocks, meta = [], []
    for row in windows.itertuples(index=False):
        start_idx = grid.index[grid["bin_start"] == row.context_start]
        if len(start_idx) == 0:
            continue
        s = start_idx[0]
        block = grid.iloc[s:s + CONTEXT_BINS + horizon_bins]
        if len(block) != CONTEXT_BINS + horizon_bins:
            continue  # a gap inside this specific window; cannot be scored
        block = block.reset_index(drop=True)
        ctx_tokens = tokenize_glucose(block["cgm_glucose"].to_numpy()[:CONTEXT_BINS])
        ids = pd.Series(ctx_tokens).astype(str).map(CGM_TKN2IDX)
        if ids.isna().any():
            raise ValueError("context token outside the CGMValue vocab")
        ctx_blocks.append(ids.to_numpy(dtype=np.int64))
        meta.append((row.window_id, block.iloc[CONTEXT_BINS:]))
    if not ctx_blocks:
        return None

    input_ids = torch.from_numpy(np.stack(ctx_blocks)).to(device)
    with torch.no_grad():
        generated = model.generate(
            input_ids=input_ids, max_new_tokens=horizon_bins, do_sample=False, pad_token_id=CGM_PAD_IDX,
        )
    forecast_ids = generated[:, CONTEXT_BINS:].cpu().numpy()

    rows = []
    for i, (window_id, truth) in enumerate(meta):
        for j in range(horizon_bins):
            token = CGM_IDX2TKN[forecast_ids[i, j]]
            rows.append({
                "window_id": window_id,
                "bin_start": truth["bin_start"].iloc[j],
                "step": j + 1,
                "prediction_token": token,
                "predictions": token_to_mgdl(token),
                "actual": truth["cgm_glucose"].iloc[j],
            })
    out = pd.DataFrame(rows)
    out["abs_error"] = (out["predictions"] - out["actual"]).abs()
    return out


def load_eval_set():
    e = pd.read_csv(EVAL_SET_PATH, dtype={"person_id": str, "window_id": str})
    e = e[e["horizon"].isin(EVAL_HORIZONS)].copy()
    for col in ["context_start", "t0", "anchor"] + [c for c in e.columns if c.startswith("history_start_")]:
        e[col] = pd.to_datetime(e[col], utc=True)
    return e


def done_keys(path):
    if not os.path.exists(path):
        return set()
    d = pd.read_csv(path, dtype={"person_id": str})
    return set(zip(d["person_id"], d["arm"]))


def history_window(e, pid, k):
    """(history_start, anchor) for one participant at one k, read from the eval
    set's OWN history_start_<k>d / anchor columns rather than recomputed as
    anchor - k days. Same values, but build_personalization_eval_set.py is the
    single source of truth for the k convention across all four model arms, and
    recomputing it here would silently diverge if that convention ever changes."""
    pe = e[e["person_id"] == pid]
    col = f"history_start_{k}d"
    if col not in pe.columns:
        raise ValueError(f"eval set has no {col}; k={k} is not a supported history length")
    return pe[col].iloc[0], pe["anchor"].iloc[0]


def assign_donors(all_participants):
    """Deterministic donor map for the negative control: each participant's
    donor is the next one in the sorted full list (last wraps to first).
    Computed from the FULL list before sharding, so an array task's donor
    assignment does not depend on the shard layout."""
    n = len(all_participants)
    return {p: all_participants[(i + 1) % n] for i, p in enumerate(all_participants)}


def main():
    job_start = time.time()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    for arm in ARMS:
        if arm not in ARM_HISTORY_DAYS:
            raise ValueError(f"unknown arm {arm!r}; known: {sorted(ARM_HISTORY_DAYS)}")

    # One persistent model instance, restored to base_state before every
    # (participant, arm) by reset_to_base(). Cheaper than re-reading the
    # checkpoint from disk hundreds of times, and the isolation guarantee is
    # the same restore-and-verify either way.
    model = CgmLsmLMHeadModel.from_pretrained(BASE_CKPT)
    max_h = max(EVAL_HORIZONS.values())
    if model.config.n_positions < CONTEXT_BINS + max_h:
        raise ValueError(
            f"checkpoint n_positions={model.config.n_positions} < CONTEXT_BINS({CONTEXT_BINS}) + "
            f"{max_h}: generate() would index past the wpe table and trip a CUDA "
            "indexSelectLargeIndex assertion mid-job. Retrain the base with longer chunks "
            "before extending horizons.")
    base_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    model.to(device)
    print(f"base checkpoint: {BASE_CKPT} (n_positions={model.config.n_positions}, ok for "
          f"horizons up to {max_h} bins) on {device}", file=sys.stderr)

    e = load_eval_set()
    covs = load_participants_covariates()
    grids = GridCache(covs)

    all_participants = sorted(e["person_id"].unique())
    donors = assign_donors(all_participants)

    # Same sanity print as the SSM-CGM arm: eval participants must be unseen by
    # the checkpoint's own training population, or personalization gains could
    # be memorisation rather than adaptation.
    train_pids = set(
        pd.read_csv(os.path.join(REPO_ROOT, "model_input", "cgmlsm", "cgmlsm_train.csv"),
                    usecols=["person_id"], dtype={"person_id": str})["person_id"].unique())
    print(f"personalization eval participants: {len(all_participants)}, overlap with CGM-LSM "
          f"training population: {len(train_pids & set(all_participants))}", file=sys.stderr)

    participants = [p for i, p in enumerate(all_participants) if i % NUM_SHARDS == SHARD]
    if MAX_PARTICIPANTS is not None:
        participants = participants[:MAX_PARTICIPANTS]
    print(f"shard {SHARD}/{NUM_SHARDS}: {len(participants)} participants, arms={ARMS}, "
          f"lr={LEARNING_RATE}, steps={NUM_STEPS}, batch={BATCH_SIZE}, stride={FIT_STRIDE}",
          file=sys.stderr)

    for arm in ARMS:
        os.makedirs(os.path.join(OUTPUT_ROOT, arm), exist_ok=True)

    skipped, fit_seconds_total, n_fitted = [], 0.0, 0
    for i, pid in enumerate(participants):
        pe = e[e["person_id"] == pid]
        own_grid = grids.get(pid)
        if own_grid is None:
            skipped.append((pid, "all", "no_grid"))
            continue
        anchor = pe["anchor"].iloc[0]

        for arm in ARMS:
            arm_dir = os.path.join(OUTPUT_ROOT, arm)
            done_path = os.path.join(arm_dir, f"done_shard{SHARD}.csv")
            if (pid, arm) in done_keys(done_path):
                continue

            k = ARM_HISTORY_DAYS[arm]
            fit_sec, loss_first, loss_last, n_chunks = 0.0, np.nan, np.nan, 0

            if k == 0:
                reset_to_base(model, base_state)
                model.eval()
            else:
                # A donor arm fits on someone else's history but is still
                # scored on THIS participant's windows, so the two arms differ
                # only in whose glucose the adaptation saw. The donor's own
                # anchor is used for their history slice -- slicing a donor's
                # record at this participant's anchor could fall outside the
                # donor's monitoring period entirely.
                src_pid = donors[pid] if arm in DONOR_ARMS else pid
                src_grid = grids.get(src_pid)
                if src_grid is None:
                    skipped.append((pid, arm, "no_donor_grid"))
                    continue
                history_start, src_anchor = history_window(e, src_pid, k)
                chunks = fit_chunks(src_grid, history_start, src_anchor)
                if chunks is None:
                    skipped.append((pid, arm, "no_eligible_fit_chunks"))
                    continue
                n_chunks = len(chunks)
                _, fit_sec, loss_first, loss_last = finetune(model, base_state, chunks, device)
                fit_seconds_total += fit_sec
                n_fitted += 1

            n_scored_windows = 0
            for horizon_name, horizon_bins in EVAL_HORIZONS.items():
                w = pe[pe["horizon"] == horizon_name]
                if w.empty:
                    continue
                out = forecast_windows(model, own_grid, w, horizon_name, horizon_bins, device)
                if out is None:
                    continue
                out["person_id"] = pid
                out["arm"] = arm
                out["history_days"] = k
                path = os.path.join(arm_dir, f"predictions_{horizon_name}.csv")
                out.to_csv(path, mode="a", header=not os.path.exists(path), index=False)
                n_scored_windows += out["window_id"].nunique()

            pd.DataFrame([{
                "person_id": pid, "arm": arm, "history_days": k,
                "donor_person_id": donors[pid] if arm in DONOR_ARMS else "",
                "n_fit_chunks": n_chunks, "n_eval_windows_scored": n_scored_windows,
                "fit_seconds": round(fit_sec, 1),
                "fit_loss_first5": round(loss_first, 4) if not np.isnan(loss_first) else "",
                "fit_loss_last5": round(loss_last, 4) if not np.isnan(loss_last) else "",
            }]).to_csv(done_path, mode="a", header=not os.path.exists(done_path), index=False)

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        if (i + 1) % 10 == 0 or i + 1 == len(participants):
            print(f"  {i + 1}/{len(participants)} participants, "
                  f"{fit_seconds_total / max(n_fitted, 1):.1f}s/fit avg, "
                  f"{time.time() - job_start:.0f}s elapsed", file=sys.stderr)

    if skipped:
        pd.DataFrame(skipped, columns=["person_id", "arm", "reason"]).to_csv(
            os.path.join(OUTPUT_ROOT, f"skipped_shard{SHARD}.csv"), index=False)
        print(f"WARNING: skipped {len(skipped)} (participant, arm) pairs: {skipped[:5]}", file=sys.stderr)

    # Per-arm/per-horizon summary. NOT the comparison output: arms are compared
    # only on matched keys (see cgmlsm_arm_matched_comparison.py for why diffing
    # these MAEs directly is exactly the trap this project already hit), so
    # n_nonnumeric_predictions is carried here to make any denominator drift
    # between arms visible immediately rather than silently.
    rows = []
    for arm in ARMS:
        for horizon_name in EVAL_HORIZONS:
            path = os.path.join(OUTPUT_ROOT, arm, f"predictions_{horizon_name}.csv")
            if not os.path.exists(path):
                continue
            full = pd.read_csv(path, dtype={"window_id": str, "person_id": str,
                                            "prediction_token": str})
            scored = full.dropna(subset=["actual", "predictions"])
            rows.append({
                "arm": arm, "horizon": horizon_name, "split": SPLIT,
                "history_days": ARM_HISTORY_DAYS[arm], "is_negative_control": arm in DONOR_ARMS,
                "mae_mgdl": scored["abs_error"].mean(),
                "rmse_mgdl": float(np.sqrt((scored["abs_error"] ** 2).mean())),
                "n_scored": len(scored), "n_total": len(full),
                "n_nonnumeric_predictions": int(full["predictions"].isna().sum()),
                "n_windows": full["window_id"].nunique(),
                "n_participants": full["person_id"].nunique(),
                "learning_rate": LEARNING_RATE, "num_steps": NUM_STEPS,
                "batch_size": BATCH_SIZE, "fit_stride": FIT_STRIDE,
                "mean_fit_seconds": round(fit_seconds_total / max(n_fitted, 1), 1),
                "cumulative_job_seconds": round(time.time() - job_start, 1),
            })
    if rows:
        summary = pd.DataFrame(rows)
        summary.to_csv(os.path.join(OUTPUT_ROOT, f"summary_shard{SHARD}.csv"), index=False)
        print(summary.to_string(index=False), file=sys.stderr)
    print("CGMLSM_PERSONALIZATION_OK", file=sys.stderr)


if __name__ == "__main__":
    main()
