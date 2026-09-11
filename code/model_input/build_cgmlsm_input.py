"""CGM-LSM training/validation/test input, glucose-only -- unlike SSM-CGM,
CGM-LSM's own architecture (code/external/cgmlsm/code/pipeline/nn/cgmlsm/{configuration,
modeling}_cgmlsm.py) takes no wearable covariates at all, only a tokenized
glucose sequence plus a time-of-day sequence, so there is no with_covariates
condition to build here.

CGM-LSM ships no public checkpoint (report.md Sec 15.6, HANDOFF.md) and must
be trained from scratch on AI-READI, same as SSM-CGM. Tokenization uses the
two vocabularies CGM-LSM's own run_cgmlsm_pretrain.py constructs from
scratch (cgmlsm_vocab.py, copied verbatim from there).

Train/val: continuous per-participant chunks of CONTEXT_BINS (144 bins, 12h
-- the same context length used everywhere else in this comparison,
report.md Sec 15.2) tokens each, built from build_participant_grid.build_grid()
over the official train/val split participants (participants.tsv
recommended_split, the one shared partition used by every model in this
comparison, report.md Sec 15.4). A chunk is kept only if it passes the same
GlyLLM coverage rule (coverage.is_window_eligible) used for eval windows
elsewhere in this project, checked on cgm_glucose alone. This is stricter
than SSM-CGM's own training-side policy (LOCF-fill covariates with no gap
limit, report.md Sec 15.5) needs to be: CGM-LSM has no covariates to LOCF at
all, and its vocabulary already has an honest "missing" symbol (UNKNOWN,
see cgmlsm_vocab.tokenize_glucose), so applying the same coverage bar here
just skips chunks that are mostly missing outright rather than either
fabricating a fill or training on a chunk that is nearly all UNKNOWN.
Chunks that pass still keep their genuine residual NaN bins as UNKNOWN
tokens, never a numeric fill -- consistent with this project's standing
rule that cgm_glucose is never imputed (build_participant_grid.py's own
docstring).

Test: NOT independently windowed. Reads window_index.csv's own test-split,
glucose-only (with_covariates=False) rows directly -- the exact same
(person_id, context_start) pairs export_zeroshot_windows.py already used to
build Chronos-2/TimesFM's glucose_only zero-shot eval files -- so CGM-LSM is
scored on windows identical to, not merely comparable with, the other
models' at the same horizon. This is what makes the cross-model comparison
apples-to-apples. Only 30min/1h/2h are built, matching every other model in
this comparison; 3h/4h is a Chronos-2-only exploratory extension
(code/CHRONOS2_EVAL.md) not built for any other model here either.
"""
import glob
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from build_participant_grid import build_grid, load_participants_covariates  # noqa: E402
from coverage import is_window_eligible  # noqa: E402
from windows import CONTEXT_BINS, HORIZONS_BINS  # noqa: E402
from cgmlsm_vocab import tokenize_glucose, tokenize_tod  # noqa: E402

ANALYSIS_DIR = "/path/to/aireadi-collab/analysis_dataframe"
WINDOW_INDEX_PATH = "/path/to/aireadi-collab/model_input/window_index.csv"
OUTPUT_DIR = "/path/to/aireadi-collab/model_input/cgmlsm"

# 3h/4h is a Chronos-2-only exploratory extension (CHRONOS2_EVAL.md), not
# built for CGM-LSM, matching SSM-CGM/TimesFM's own scope.
EVAL_HORIZONS = {k: v for k, v in HORIZONS_BINS.items() if k in ("30min", "1h", "2h")}
GRID_CACHE_MAX = 50  # cap memory; cache hits repeat across horizons for the same pid, same convention as export_zeroshot_windows.py


def build_train_val_chunks(pid, cov):
    """One participant's train/val contribution: non-overlapping CONTEXT_BINS
    chunks over their own bin grid, each independently coverage-checked (a
    chunk failing the bar is dropped, not fabricated or padded through)."""
    df = build_grid(pid, {pid: cov})
    if df is None or df.empty:
        return None
    df = df.sort_values("bin_start").reset_index(drop=True)

    frames = []
    n = len(df)
    for start in range(0, n - CONTEXT_BINS + 1, CONTEXT_BINS):
        chunk = df.iloc[start:start + CONTEXT_BINS]
        eligible, _ = is_window_eligible(chunk, ["cgm_glucose"])
        if not eligible:
            continue
        chunk = chunk.reset_index(drop=True)
        out = pd.DataFrame({
            "chunk_id": f"{pid}_{start // CONTEXT_BINS}",
            "person_id": pid,
            "position": range(len(chunk)),
            "bin_start": chunk["bin_start"],
            "cgm_glucose": chunk["cgm_glucose"],
            "cgm_token": tokenize_glucose(chunk["cgm_glucose"].to_numpy()),
            "tod_token": tokenize_tod(chunk["bin_start"]),
        })
        frames.append(out)
    if not frames:
        return None
    return pd.concat(frames, ignore_index=True)


def build_test_windows(covs, split="test"):
    """Eval windows for one split, read straight from window_index.csv's own
    (person_id, context_start) rows -- never re-windowed here, which is what
    keeps CGM-LSM scored on the SAME windows as Chronos-2/TimesFM.

    `split` exists so a VAL-window eval file can be built too. The original
    fine-tune sweeps (cgmlsm_finetune_sweep{,2}_results) selected their recipe
    by MAE on TEST windows -- model selection on the reporting set.

    That was an omission, not a constraint. A val split was always available in
    two forms: cgmlsm_val.csv (290 held-out participants, already used for
    eval_loss during training) and window_index.csv's own val rows (4,589/
    4,362/4,052 windows at 30min/1h/2h). Only the windowed val FILE was
    missing, and producing it needed nothing more than this `split` argument.
    The sweeps did consult val eval_loss and found it disagreed with MAE (the
    lighter recipe had worse loss but better MAE); preferring MAE over
    teacher-forced loss is reasonable, but the right response was to build
    val-window MAE rather than to select on test.
    """
    index = pd.read_csv(WINDOW_INDEX_PATH, dtype={"person_id": str})
    test_index = index[(index["recommended_split"] == split) & (index["with_covariates"] == False)]  # noqa: E712

    grid_cache = {}
    frames = []
    for horizon_name, horizon_bins in EVAL_HORIZONS.items():
        subset = test_index[test_index["horizon"] == horizon_name]
        for i, (_, w) in enumerate(subset.iterrows()):
            pid = w["person_id"]
            if pid not in grid_cache:
                grid_cache[pid] = build_grid(pid, covs)
                if len(grid_cache) > GRID_CACHE_MAX:
                    grid_cache.pop(next(iter(grid_cache)))
            df = grid_cache[pid]
            if df is None:
                continue
            context_start = pd.Timestamp(w["context_start"])
            start_idx = df.index[df["bin_start"] == context_start]
            if len(start_idx) == 0:
                continue
            start_idx = start_idx[0]
            block = df.iloc[start_idx:start_idx + CONTEXT_BINS + horizon_bins]
            if len(block) != CONTEXT_BINS + horizon_bins:
                continue
            block = block.reset_index(drop=True)
            out = pd.DataFrame({
                "window_id": f"{pid}_{horizon_name}_{i}",
                "person_id": pid,
                "horizon": horizon_name,
                "bin_start": block["bin_start"],
                "is_context": [True] * CONTEXT_BINS + [False] * horizon_bins,
                "cgm_glucose": block["cgm_glucose"],
                "cgm_token": tokenize_glucose(block["cgm_glucose"].to_numpy()),
                "tod_token": tokenize_tod(block["bin_start"]),
            })
            frames.append(out)
    if not frames:
        return pd.DataFrame(columns=["window_id", "person_id", "horizon", "bin_start", "is_context",
                                      "cgm_glucose", "cgm_token", "tod_token"])
    return pd.concat(frames, ignore_index=True)


def run_windows_only(split):
    """Builds ONLY the eval-window file for one split, skipping the ~6-minute
    train/val chunk rebuild. Used to add the val-window eval set without
    regenerating (and risking any drift in) the train/val/test files that
    every CGM-LSM result so far was produced from."""
    covs = load_participants_covariates()
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    df = build_test_windows(covs, split=split)
    name = "cgmlsm_test.csv" if split == "test" else f"cgmlsm_{split}_windows.csv"
    path = os.path.join(OUTPUT_DIR, name)
    if os.path.exists(path):
        raise SystemExit(f"{path} already exists; refusing to overwrite an input file that "
                         "existing results were produced from")
    df.to_csv(path, index=False)
    print(f"{split} windows: {df['person_id'].nunique()} participants, "
          f"{df['window_id'].nunique()} windows, {len(df)} rows -> {path}", file=sys.stderr)


def run():
    covs = load_participants_covariates()
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    paths = sorted(glob.glob(f"{ANALYSIS_DIR}/*.csv"))
    pids = [os.path.basename(p).replace(".csv", "") for p in paths]

    for split in ("train", "val"):
        frames = []
        n_pids = 0
        for i, pid in enumerate(pids):
            cov = covs.get(str(pid), {})
            if cov.get("recommended_split") != split:
                continue
            chunk_df = build_train_val_chunks(pid, cov)
            if chunk_df is not None:
                frames.append(chunk_df)
                n_pids += 1
            if (i + 1) % 300 == 0:
                print(f"  {split}: processed {i + 1}/{len(pids)}", file=sys.stderr)
        out = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(
            columns=["chunk_id", "person_id", "position", "bin_start", "cgm_glucose", "cgm_token", "tod_token"])
        out.to_csv(os.path.join(OUTPUT_DIR, f"cgmlsm_{split}.csv"), index=False)
        print(f"{split}: {n_pids} participants, {out['chunk_id'].nunique()} chunks, {len(out)} rows", file=sys.stderr)

    test_df = build_test_windows(covs)
    test_df.to_csv(os.path.join(OUTPUT_DIR, "cgmlsm_test.csv"), index=False)
    print(f"test: {test_df['person_id'].nunique()} participants, {test_df['window_id'].nunique()} windows, "
          f"{len(test_df)} rows", file=sys.stderr)


if __name__ == "__main__":
    windows_only = os.environ.get("CGMLSM_BUILD_WINDOWS_ONLY_SPLIT")
    if windows_only:
        run_windows_only(windows_only)
    else:
        run()
