"""Builds the augmented SSM-CGM training file for the paper-style
personalization experiment: original train participants' full data, PLUS
each eval-set-E test participant's own pre-anchor history (build_
ssmcgm_input.build_participant_series(..., cutoff=anchor)).

WHY THIS MATCHES THEIR PAPER, NOT A NEW METHOD
=================================================
SSM-CGM's own paper trains on each participant's own earlier data and tests
on their own later window (per-participant time-based holdout) -- every test
participant is partially seen in training. This reproduces exactly that
structure on AI-READI: the disjoint `train` population plus, for E's test
participants specifically, everything strictly before their own causal
anchor (model_input/personalization/eval_set_test.csv, the SAME anchor
Stephanie's Chronos-2 LoRA arm and the TimesFM in-context arm both use for
their own personalization). The eventual test-time evaluation (a separate
script) only scores windows at/after the anchor -- the causal boundary is
enforced by construction (build_ssmcgm_input.build_participant_series's new
`cutoff` filters on real bin_start, before ds/series_id are (re)derived),
not by a downstream check.

WHY A FRESH TRAINING RUN, NOT A WARM START FROM THE EXISTING CHECKPOINT
==========================================================================
Considered and rejected: continuing ssmcgm_train_real.py's existing
best-epoch=0 checkpoint on this augmented data. MambaTFT's participant_id/
series_id static-categorical embeddings are sized to whatever vocabulary a
TimeSeriesDataSet.from_dataset() call was built against; adding genuinely
new (test) participant IDs means a NEW, larger vocabulary, and the existing
checkpoint's embedding weights have no rows for those new IDs -- loading
would either shape-mismatch outright or require manual embedding-table
surgery (copy old rows by ID, randomly init new ones) to do correctly. That
is real, error-prone custom code for a result that would still just be an
approximation. Training fresh on the augmented set instead sizes the model
correctly from the start (no surgery needed) and is MORE faithful to their
own methodology besides: their paper trains fresh under their own split, not
warm-started from a different one.

OUTPUT
======
model_input/ssmcgm/ssmcgm_train_plus_testhistory.csv -- original
ssmcgm_train.csv rows, unchanged, plus E's test participants' pre-anchor
segments appended. ssmcgm_val.csv is NOT touched -- validation stays the
existing disjoint set, so model selection (which epoch is "best") is judged
by the same criterion as the original run, uncontaminated by this
experiment's own test population.
"""
import os
import sys

import pandas as pd

REPO_ROOT = "/path/to/aireadi-collab"
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_ssmcgm_input import build_participant_series, KEEP_COLS  # noqa: E402
from build_participant_grid import load_participants_covariates  # noqa: E402

ORIGINAL_TRAIN_CSV = f"{REPO_ROOT}/model_input/ssmcgm/ssmcgm_train.csv"
EVAL_SET_TEST_PATH = f"{REPO_ROOT}/model_input/personalization/eval_set_test.csv"
OUTPUT_PATH = f"{REPO_ROOT}/model_input/ssmcgm/ssmcgm_train_plus_testhistory.csv"


def main():
    e = pd.read_csv(EVAL_SET_TEST_PATH, dtype={"person_id": str})
    e["anchor"] = pd.to_datetime(e["anchor"], utc=True)
    anchors = e.groupby("person_id")["anchor"].first()
    print(f"E test participants (getting pre-anchor history injected): {len(anchors)}", file=sys.stderr)

    covs = load_participants_covariates()

    frames = []
    n_ok, n_empty = 0, 0
    for pid, anchor in anchors.items():
        cov = covs.get(str(pid), {})
        series = build_participant_series(pid, cov, cutoff=anchor)
        if series is None:
            n_empty += 1
            continue
        frames.append(series)
        n_ok += 1
        if n_ok % 50 == 0:
            print(f"  {n_ok}/{len(anchors)} processed", file=sys.stderr)

    print(f"built pre-anchor history for {n_ok}/{len(anchors)} test participants "
          f"({n_empty} had no rows before their own anchor)", file=sys.stderr)

    test_history_df = pd.concat(frames, ignore_index=True)
    print(f"test-history addition: {test_history_df['participant_id'].nunique()} participants, "
          f"{test_history_df['series_id'].nunique()} segments, {len(test_history_df)} rows", file=sys.stderr)

    original_train_df = pd.read_csv(ORIGINAL_TRAIN_CSV, dtype={"participant_id": str})
    # series_id collision check: original train and E's test participants are
    # disjoint populations by construction (this project's own train/test
    # split), and series_id is f"{participant_id}_{segment}" -- so a
    # collision would require the same participant_id in both, which is
    # exactly what a disjoint split rules out. Verified directly rather than
    # assumed: a real bug here would silently splice two different people's
    # rows into one "series," so it is checked, not trusted.
    overlap = set(original_train_df["participant_id"]) & set(test_history_df["participant_id"])
    if overlap:
        raise SystemExit(f"REFUSING TO WRITE: {len(overlap)} participant_id(s) appear in both "
                          f"the original train split and E's test participants: {sorted(overlap)[:5]}... "
                          f"this would mean train/test are not actually disjoint.")

    combined = pd.concat([original_train_df, test_history_df], ignore_index=True)
    combined = combined[KEEP_COLS]
    combined.to_csv(OUTPUT_PATH, index=False)
    print(f"wrote {OUTPUT_PATH}: {combined['participant_id'].nunique()} participants "
          f"({original_train_df['participant_id'].nunique()} original + {n_ok} test-history), "
          f"{combined['series_id'].nunique()} segments, {len(combined)} rows", file=sys.stderr)


if __name__ == "__main__":
    main()
