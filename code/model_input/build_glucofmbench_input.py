"""CGM-LSM pretraining input from GlucoFM-Bench
(huggingface.co/datasets/glucofmbench/GlucoFM-Bench), an 11-cohort
aggregation of open-access CGM datasets -- distinct from this project's
AI-READI-only cgmlsm_{train,val,test}.csv (code/model_input/build_cgmlsm_input.py),
which is a from-scratch fit to one population. This is pretraining, not
fine-tuning: a broad external corpus with no AI-READI participants in it,
used to build a general CGM-LSM checkpoint before any downstream use on
AI-READI (matching the authors' own run_cgmlsm_pretrain.py role: pretrain
on a large external corpus first, apply to a smaller target set after).

Raw files: model_input/glucofmbench_raw/{train,test}.parquet (downloaded
from HF directly, one row per participant-series: dataset, subject_id,
timestamp (list[float64], unix seconds), BGvalue (list[float64], mg/dL)).
GlucoFM-Bench's own train/test parquet split is participant-level (80/20,
per its README) -- this script only ever reads train.parquet; test.parquet
is deliberately left untouched here, reserved for a possible future
zero-shot check of the pretrained checkpoint, not used for
training/validation in this pass.

Sampling rate: verified directly (not assumed) that all 11 sub-cohorts are
already a uniform 5-minute grid in this file -- GlucoFM-Bench's own README
"Preprocessing" section confirms ShanghaiT1DM/ShanghaiT2DM (natively ~15min,
FreeStyle Libre) were upsampled and CGMacros (natively finer) was
downsampled to "one sample per five minutes" before publishing this
aggregation, and gaps <1h were linearly interpolated. Measuring the actual
per-cohort modal timestamp diff confirms this: 99.9%+ of consecutive-reading
gaps across all 11 cohorts are exactly 300s. This means a real, disclosed
caveat: some of ShanghaiT1DM/T2DM's "5-minute" values here are GlucoFM-Bench's
own linear interpolation of an originally-coarser device, not genuine device
readings, and this file carries no flag distinguishing the two -- unlike
this project's own AI-READI pipeline, which never interpolates cgm_glucose
for exactly this reason. Nothing here can undo GlucoFM-Bench's own upstream
choice; this is disclosed, not fixed.

Segmentation and tokenization deliberately reuse the exact same machinery as
build_cgmlsm_input.py's train/val chunks, not a reimplementation: each
series is placed on its own 5-minute-bin integer grid (reusing each row's
already-5-min-aligned real timestamps directly, not re-resampling), split
into contiguous segments at internal gaps >60 minutes (104/529 train rows
have at least one such gap -- GlucoFM-Bench's own README calls these
"sequence boundaries," but its shared parquet does not pre-split rows at
them, so this script does), then chunked into non-overlapping CONTEXT_BINS
(144-bin, 12h) blocks, each independently checked against the same GlyLLM
coverage rule (coverage.is_window_eligible) used everywhere else in this
project and dropped -- not fabricated -- if it fails. Kept chunks' residual
missing bins become the CGMValue vocabulary's own UNKNOWN token
(cgmlsm_vocab.tokenize_glucose), never a numeric fill.

Split: reproduces the CGM-LSM paper's own three-layer held-in/held-out
scheme (papers/2412.09727v3.pdf Methods), not just a single held-out slice
-- confirmed directly against the paper's text, not inferred from the repo
alone:

1. **held_out** -- 10% of subjects per cohort (not globally -- a cohort
   with few participants shouldn't be entirely absorbed into or excluded
   from held_out), selected via a seeded RNG (seed=42, matching the paper's
   own "All random splits were performed using a fixed random seed of 42"),
   entirely excluded from every other split. This is the paper's primary
   "held-out test set... excluded from the training process."
2. For the *remaining* (held-in) subjects: each subject's own chunks are
   ranked by within-subject recency (1.0 = that subject's own latest chunk),
   then pooled at the **cohort level** (not decided independently per
   subject) and the cohort's highest-recency TEMPORAL_FRACTION become
   **temporal** -- the paper's "temporal" test set, testing generalization
   across time within known patients.
3. Of what's left, a **random INTERNAL_FRACTION of the cohort's original
   total** (seeded, same rng, drawn from the pooled remainder) becomes
   **internal** -- the paper's "internal" test set.
4. The remaining ~80% becomes **train**.

The paper states "the remaining 90% of instances were randomly divided,
with 10% forming the internal test set and 80% designated for training" --
read here as 10%/80% of each held-in subject's *original* total (not of the
already-reduced 90%), since 10 (temporal) + 10 (internal) + 80 (train) is
the only reading that sums cleanly to 100%; disclosed as an interpretation
of a slightly ambiguous sentence, not a verified-to-the-decimal replication.

**Pooled at the cohort level, not decided independently per subject** --
this matters, not just a style choice. An earlier version of this script
applied TEMPORAL_FRACTION/INTERNAL_FRACTION independently to each held-in
subject's own chunk count, which rounds to zero whenever a subject has few
chunks. Cohorts made of many short per-subject series (Colas2019: 187
held-in chunks spread across ~187 subjects, so round(n_subject*0.1) was 0
for essentially every one of them) silently vanished from `temporal`/
`internal` entirely even though the cohort had plenty of chunks overall --
found by inspecting a completed pretraining run's own eval-loss anomaly
(internal loss higher than held_out's, traced to Colas2019 being present in
held_out but completely absent from internal). Pooling per cohort instead
of per subject fixes this: every cohort with >=2 held-in chunks now
contributes to every split it has room for.

Earlier versions of this script (both superseded): the first only
implemented layer 1 (as `val`), kept 100% of held-in chunks as train; the
second added layers 2/3 but per-subject (the Colas2019-dropout bug above).
This version is per-cohort-pooled layers 2/3.
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from coverage import is_window_eligible  # noqa: E402
from windows import CONTEXT_BINS  # noqa: E402
from cgmlsm_vocab import tokenize_glucose, tokenize_tod  # noqa: E402

RAW_DIR = "/path/to/aireadi-collab/model_input/glucofmbench_raw"
# GLUCOFMBENCH_OUTPUT_DIR_OVERRIDE / GLUCOFMBENCH_EXCLUDE_COHORTS: used for the
# T1D-excluded variant (AI-READI is a T2D-staged population -- healthy/
# pre-diabetes/oral-medication/insulin-dependent per participants.tsv's
# study_group, never T1D -- so mixing in the 6 T1D-labeled cohorts, which are
# ~78.5% of the full corpus by chunk count and dominated by one cohort
# (Bris-T1D Open alone ~39%), is a population-mismatch risk the user wants
# tested directly rather than assumed). Excluding cohorts happens on the raw
# per-subject dataframe, before any split/coverage logic, so held_out/
# temporal/internal/train all only ever see the retained cohorts.
OUTPUT_DIR = os.environ.get(
    "GLUCOFMBENCH_OUTPUT_DIR_OVERRIDE",
    "/path/to/aireadi-collab/model_input/glucofmbench",
)
EXCLUDE_COHORTS = [c.strip() for c in os.environ.get("GLUCOFMBENCH_EXCLUDE_COHORTS", "").split(",") if c.strip()]
BIN_SECONDS = 300  # 5 minutes, verified as this file's uniform native grid (see module docstring)
MAX_GAP_SECONDS = 3600  # 60 min -- same segment-boundary threshold as coverage.py's own gap rule
HELD_OUT_FRACTION = 0.1  # layer 1: fraction of subjects per cohort, entirely excluded
TEMPORAL_FRACTION = 0.1  # layer 2: fraction of each held-in subject's chunks, latest by time
INTERNAL_FRACTION = 0.1  # layer 3: fraction of each held-in subject's chunks, random
RNG_SEED = 42  # matches the paper's own "fixed random seed of 42" for all random splits


def series_to_chunks(dataset_name, subject_id, timestamp, bgvalue):
    """One GlucoFM-Bench row (ragged timestamp/BGvalue lists) -> list of
    coverage-eligible CONTEXT_BINS-length chunk DataFrames, following the
    exact same per-chunk policy as build_cgmlsm_input.py's train/val path."""
    ts = np.asarray(timestamp, dtype=np.float64)
    bg = np.asarray(bgvalue, dtype=np.float64)
    order = np.argsort(ts)
    ts, bg = ts[order], bg[order]

    bin_idx = np.round((ts - ts[0]) / BIN_SECONDS).astype(np.int64)
    # de-duplicate any bin collisions (two raw readings rounding to the same
    # 5-min slot) by keeping the first, consistent with this project's
    # standing "never fabricate cgm_glucose" rule -- averaging would invent
    # a value neither reading actually reported.
    _, first_pos = np.unique(bin_idx, return_index=True)
    bin_idx, bg = bin_idx[first_pos], bg[first_pos]

    gap_bins = np.diff(bin_idx)
    segment = np.concatenate([[0], np.cumsum(gap_bins * BIN_SECONDS > MAX_GAP_SECONDS)])

    chunks = []
    for seg_id in np.unique(segment):
        seg_mask = segment == seg_id
        seg_bin_idx, seg_bg = bin_idx[seg_mask], bg[seg_mask]
        # dense per-segment grid: real bins keep their value, gaps <=60min
        # (already guaranteed by the segment split above) become NaN, never
        # filled -- same as build_participant_grid.py's own cgm_glucose policy.
        dense = np.full(seg_bin_idx[-1] - seg_bin_idx[0] + 1, np.nan)
        dense[seg_bin_idx - seg_bin_idx[0]] = seg_bg
        bin_start = pd.to_datetime(ts[0] + (seg_bin_idx[0] + np.arange(len(dense))) * BIN_SECONDS, unit="s", utc=True)

        for start in range(0, len(dense) - CONTEXT_BINS + 1, CONTEXT_BINS):
            chunk_glucose = dense[start:start + CONTEXT_BINS]
            chunk_df = pd.DataFrame({
                "cgm_glucose": chunk_glucose,
                "cgm_glucose_imputed": pd.isna(chunk_glucose),
                "bin_start": bin_start[start:start + CONTEXT_BINS],
            })
            eligible, _ = is_window_eligible(chunk_df, ["cgm_glucose"])
            if not eligible:
                continue
            chunk_id = f"{dataset_name}_{subject_id}_{seg_id}_{start // CONTEXT_BINS}"
            chunks.append(pd.DataFrame({
                "chunk_id": chunk_id,
                "dataset": dataset_name,
                "subject_id": subject_id,
                "position": range(CONTEXT_BINS),
                "bin_start": chunk_df["bin_start"],
                "cgm_glucose": chunk_df["cgm_glucose"],
                "cgm_token": tokenize_glucose(chunk_df["cgm_glucose"].to_numpy()),
                "tod_token": tokenize_tod(chunk_df["bin_start"]),
            }))
    return chunks


def split_held_in_cohort_chunks(items, rng):
    """items: list of (subject_id, chunk_df, recency) for every held-in
    chunk in ONE cohort, recency in (0, 1] = this chunk's position within
    its OWN subject's chronological order (1.0 = that subject's latest
    chunk). Pooled at the COHORT level (not per-subject independently) so
    TEMPORAL_FRACTION/INTERNAL_FRACTION are applied to the cohort's total,
    not rounded to zero per subject -- the earlier per-subject version
    silently dropped entire cohorts (Colas2019: every held-in subject's own
    chunk count was too small for round(n*0.1) to ever clear 1) from
    temporal/internal even though the cohort had 187 held-in chunks overall.
    Pooling guarantees every cohort with >=2 held-in chunks contributes to
    every split it has room for, while still preferring each subject's own
    most-recent chunks for `temporal` (recency-ranked pool-wide, not just
    within one subject) and a uniform random draw for `internal`."""
    total = len(items)
    if total == 0:
        return [], [], []
    if total == 1:
        # can't meaningfully split a single chunk three ways -- goes to
        # train rather than being silently dropped or forced into a test set.
        return items, [], []

    order_by_recency_desc = sorted(range(total), key=lambda idx: -items[idx][2])
    n_temporal = min(total - 1, max(1, round(total * TEMPORAL_FRACTION)))
    temporal_idx = set(order_by_recency_desc[:n_temporal])

    remaining_idx = [idx for idx in range(total) if idx not in temporal_idx]
    n_internal = min(len(remaining_idx) - 1, max(1, round(total * INTERNAL_FRACTION))) if len(remaining_idx) >= 2 else 0
    internal_idx = set(rng.choice(remaining_idx, size=n_internal, replace=False).tolist()) if n_internal else set()

    train_idx = [idx for idx in remaining_idx if idx not in internal_idx]
    return [items[i] for i in train_idx], [items[i] for i in temporal_idx], [items[i] for i in internal_idx]


def run():
    df = pd.read_parquet(os.path.join(RAW_DIR, "train.parquet"))
    if EXCLUDE_COHORTS:
        before = df["dataset"].nunique()
        df = df[~df["dataset"].isin(EXCLUDE_COHORTS)]
        print(f"excluded cohorts {EXCLUDE_COHORTS}: {before} -> {df['dataset'].nunique()} cohorts, "
              f"{len(df)} subjects remain", file=sys.stderr)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    rng = np.random.default_rng(RNG_SEED)

    held_out_subjects = set()
    for dataset_name, sub in df.groupby("dataset"):
        subject_ids = sorted(sub["subject_id"].unique())
        n_held_out = max(1, round(len(subject_ids) * HELD_OUT_FRACTION))
        chosen = rng.choice(subject_ids, size=n_held_out, replace=False)
        held_out_subjects.update((dataset_name, sid) for sid in chosen)

    frames = {"train": [], "held_out": [], "temporal": [], "internal": []}
    n_subjects = {"train": set(), "held_out": set(), "temporal": set(), "internal": set()}
    held_in_by_cohort = {}  # dataset_name -> list of (subject_id, chunk_df, recency)

    for i, row in enumerate(df.itertuples(index=False)):
        key = (row.dataset, row.subject_id)
        chunks = series_to_chunks(row.dataset, row.subject_id, row.timestamp, row.BGvalue)
        if not chunks:
            pass
        elif key in held_out_subjects:
            frames["held_out"].extend(chunks)
            n_subjects["held_out"].add(key)
        else:
            chunks_sorted = sorted(chunks, key=lambda c: c["bin_start"].iloc[0])
            n = len(chunks_sorted)
            for rank, chunk_df in enumerate(chunks_sorted):
                recency = (rank + 1) / n  # 1/n .. 1.0, 1.0 = this subject's own most recent chunk
                held_in_by_cohort.setdefault(row.dataset, []).append((row.subject_id, chunk_df, recency))
        if (i + 1) % 100 == 0:
            print(f"  processed {i + 1}/{len(df)} series", file=sys.stderr)

    for dataset_name, items in held_in_by_cohort.items():
        train_items, temporal_items, internal_items = split_held_in_cohort_chunks(items, rng)
        for split, split_items in (("train", train_items), ("temporal", temporal_items), ("internal", internal_items)):
            for subject_id, chunk_df, _ in split_items:
                frames[split].append(chunk_df)
                n_subjects[split].add((dataset_name, subject_id))

    for split in ("train", "held_out", "temporal", "internal"):
        out = pd.concat(frames[split], ignore_index=True) if frames[split] else pd.DataFrame(
            columns=["chunk_id", "dataset", "subject_id", "position", "bin_start",
                     "cgm_glucose", "cgm_token", "tod_token"])
        out.to_csv(os.path.join(OUTPUT_DIR, f"glucofmbench_{split}.csv"), index=False)
        print(f"{split}: {len(n_subjects[split])} subjects, {out['chunk_id'].nunique()} chunks, {len(out)} rows",
              file=sys.stderr)
        if not out.empty:
            print(out.groupby("dataset")["chunk_id"].nunique().to_string(), file=sys.stderr)


if __name__ == "__main__":
    run()
