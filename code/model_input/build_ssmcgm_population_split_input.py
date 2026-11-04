"""Builds SSM-CGM train/val/test files under a population-wide temporal
split, applied identically to every participant (not the participant-disjoint
split used everywhere else in this project) -- two modes, each a real,
distinct experiment:

MODE=paper: literal replication of SSM-CGM's own published methodology.
Direct quote (arXiv:2510.04386): "As participants started and ended device
use at different times of day (8-10 days total), we are able to withhold
their last two hours of data for validation (penultimate) and testing
(last)." So per participant: last 2h -> test, penultimate 2h (the 2h
immediately before that) -> val, everything earlier -> train.

MODE=proportional: matches CGM-LSM's actual structure instead (arXiv:
2412.09727's held_out/temporal/internal split for held-in subjects: most of
each subject's own data trains, a chronological tail and an internal slice
are held out) rather than SSM-CGM's own fixed-2-hour convention. Per
participant: last 10% of real (post-dropna) rows -> test, the 10% before
that -> val, remaining ~80% -> train. Proportional, not a fixed time amount,
matching GlucoBench's and CGM-LSM's convention discussed in
zeroshot_results_reports/report.md Sec 8, not SSM-CGM's own paper's fixed
2h choice.

BOTH MODES APPLY TO THE SAME POPULATION, UNIFORMLY -- unlike the earlier,
abandoned personalization attempt (build_ssmcgm_personalization_train_input.py),
neither mode singles out a "personalization subset" while leaving other
participants' full records untouched. Every participant with valid SSM-CGM
input data plays all three roles (train/val/test) with their own data, the
same as both cited papers' own designs -- resolves the "unprincipled
asymmetry" finding from the abandoned attempt.

POPULATION: every participant currently assigned to any of train/val/test in
this project's existing disjoint scheme (i.e. everyone with a built
analysis_dataframe and a real participants.tsv covariate row) -- not
restricted to the 264 people used by the Chronos-2/TimesFM personalization
work, since neither mode here is trying to match those windows; this is a
standalone experiment on its own terms (see report.md Sec 7's conclusion
that a literal replication cannot be scored on the same windows as those
arms anyway).

OUTPUT: model_input/ssmcgm/ssmcgm_{train,val,test}_{paper,proportional}.csv
"""
import glob
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_ssmcgm_input import build_participant_series, KEEP_COLS, ANALYSIS_DIR  # noqa: E402
from build_participant_grid import load_participants_covariates, build_grid  # noqa: E402

REPO_ROOT = os.environ["AIREADI_ROOT"]
OUTPUT_DIR = f"{REPO_ROOT}/model_input/ssmcgm"

MODE = os.environ.get("SSMCGM_POP_SPLIT_MODE", "paper")
assert MODE in ("paper", "proportional"), f"SSMCGM_POP_SPLIT_MODE must be 'paper' or 'proportional', got {MODE!r}"


def participant_boundaries(person_id, cov):
    """Returns (val_start, test_start) real UTC timestamps for this
    participant, per MODE. train = bin_start < val_start,
    val = val_start <= bin_start < test_start, test = bin_start >= test_start.
    None, None if the participant has too little real data to split."""
    df = build_grid(person_id, {person_id: cov})
    if df is None or df.empty:
        return None, None
    df = df.sort_values("bin_start")
    real = df.dropna(subset=["cgm_glucose"])
    if real.empty:
        return None, None

    if MODE == "paper":
        end = real["bin_start"].max()
        test_start = end - pd.Timedelta(hours=2) + pd.Timedelta(minutes=5)
        val_start = test_start - pd.Timedelta(hours=2)
        # BUG FIX 2026-09-05: this used to require val_start itself to have
        # 2h of history before it -- verified against the paper's own text
        # (arXiv:2510.04386: "ŷ = f(y_{t-L+1:t}, ...)" with L=144 (12h) and
        # H=12 (1h)) that this is stricter than the paper actually requires.
        # The paper's forecast context reaches BACK into the participant's
        # earlier (chronologically "training") data -- only the 1h forecast
        # TARGET has to fall inside the withheld 2h window, not the context
        # too. So the real requirement is 12h of real history before
        # val_start (for the context), not just any history at all.
        if val_start - pd.Timedelta(hours=12) <= real["bin_start"].min():
            return None, None  # not enough history for a 12h context before val even starts
        return val_start, test_start
    else:  # proportional
        n = len(real)
        if n < 20:  # need enough rows for a meaningful 80/10/10 split
            return None, None
        test_idx = int(n * 0.90)
        val_idx = int(n * 0.80)
        return real["bin_start"].iloc[val_idx], real["bin_start"].iloc[test_idx]


def run():
    covs = load_participants_covariates()
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    paths = sorted(glob.glob(f"{ANALYSIS_DIR}/*.csv"))
    pids = [os.path.basename(p).replace(".csv", "") for p in paths]

    frames = {"train": [], "val": [], "test": []}
    n_skipped = 0
    for i, pid in enumerate(pids):
        cov = covs.get(str(pid), {})
        split = cov.get("recommended_split", "")
        if split not in ("train", "val", "test"):
            continue  # only participants with a real, valid covariate/split row

        val_start, test_start = participant_boundaries(pid, cov)
        if val_start is None:
            n_skipped += 1
            continue

        train_series = build_participant_series(pid, cov, time_range=(None, val_start))
        if MODE == "paper":
            # FIX 2026-09-05 (revised): val/test each get the full 12h of
            # real context reaching BACK before the split boundary (paper's
            # own convention -- see participant_boundaries' comment above)
            # PLUS the entire 2h withheld region, not just its first hour.
            # This gives TimeSeriesDataSet's own sliding-window mechanism
            # (min_encoder_length=max_encoder_length=144,
            # min/max_prediction_length=12) room to form every valid 1h
            # target position within the full 2h window -- up to 13
            # overlapping eval windows per participant (24 withheld bins -
            # 12 target bins + 1), not just one. An earlier version of this
            # fix manually restricted the target to only the withheld
            # region's first hour, discarding the other hour's worth of
            # eval signal for no real reason -- corrected here to use all
            # of what the paper actually withholds, not just enough to
            # avoid the crash. Overlapping with train_series's tail is
            # intentional (paper's own design, not leakage): the model is
            # never scored on predicting something it was trained to
            # reproduce, only on genuinely new, later targets it never saw.
            val_series = build_participant_series(
                pid, cov, time_range=(val_start - pd.Timedelta(hours=12), test_start))
            test_series = build_participant_series(
                pid, cov, time_range=(test_start - pd.Timedelta(hours=12), test_start + pd.Timedelta(hours=2)))
        else:
            val_series = build_participant_series(pid, cov, time_range=(val_start, test_start))
            test_series = build_participant_series(pid, cov, time_range=(test_start, None))

        if train_series is not None:
            frames["train"].append(train_series)
        if val_series is not None:
            frames["val"].append(val_series)
        if test_series is not None:
            frames["test"].append(test_series)

        if (i + 1) % 300 == 0:
            print(f"  processed {i + 1}/{len(pids)}", file=sys.stderr)

    print(f"MODE={MODE}: {n_skipped} participants skipped (too little real data to split)", file=sys.stderr)
    for split in ("train", "val", "test"):
        if not frames[split]:
            print(f"WARNING: {split} is empty for MODE={MODE}", file=sys.stderr)
            continue
        df = pd.concat(frames[split], ignore_index=True)[KEEP_COLS]
        out_path = f"{OUTPUT_DIR}/ssmcgm_{split}_{MODE}.csv"
        df.to_csv(out_path, index=False)
        print(f"{split}: {df['participant_id'].nunique()} participants, "
              f"{df['series_id'].nunique()} segments, {len(df)} rows -> {out_path}", file=sys.stderr)


if __name__ == "__main__":
    run()
