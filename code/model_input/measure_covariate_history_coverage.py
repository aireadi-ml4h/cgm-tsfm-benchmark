"""Phase 2 feasibility check: how much WEARABLE-COVERED history does each
covariate-eligible participant actually have before their fine-tuning anchor?

WHY THIS IS A REAL QUESTION AND NOT BOOKKEEPING
===============================================
Phase 1 (glucose-only) draws its history from load_cleaned_glucose_binned(),
i.e. the participant's full cleaned glucose record, which is exactly what the
7-day bar in build_personalization_eval_set.py is measured against (t0 = first
real glucose timestamp). So in phase 1, clearing the bar GUARANTEES 7 days of
history exist.

Phase 2 cannot inherit that guarantee. A covariate history slice has to come
from build_grid(), which LEFT-JOINS glucose onto the analysis_dataframe
wearable grid -- so the slice is bounded by the wearable record, not the
glucose record, and the two do not start or end together. A participant can
clear a bar defined on glucose t0 and still have only a fraction of
[anchor - 7d, anchor) covered by wearable rows at all.

Nothing in the phase-2 arms would announce this. The fit would just run on a
shorter array, or on an array whose covariate columns are almost entirely NaN,
and the arms would be measuring wearable coverage rather than covariates.

WHAT IS MEASURED, IN THREE TIERS
================================
For each participant in eval_set_cov_<split>.csv, over their 7-day history
slice [anchor - 7d, anchor) = 2,016 five-minute bins:

  tier 1  GRID: bins present in analysis_dataframe at all. This is the hard
          bound -- build_grid()'s left join cannot invent a row outside the
          wearable span, and a bin with no grid row has no covariates by any
          route.
  tier 2  TARGET: of the grid bins, how many carry a real cgm_glucose. This is
          what the fine-tuning loss can actually be computed on, and it is the
          quantity that differs from phase 1 (phase 1 gets glucose everywhere
          the glucose record has it, regardless of wearables).
  tier 3  CHANNELS: per-channel real fraction over the slice, using
          channel_is_real_mask()'s own definition (imputed flag False AND
          non-null for the flagged channels, non-null for the rest) -- the same
          rule window eligibility uses, so these numbers are on the same
          footing as the export's own coverage decisions.

Reads the wearable CSV and the glucose record directly rather than calling
build_grid(). Deliberate, and equivalent for this purpose: build_grid()'s
subsequent joins (predmeal_prob) and derived columns (sleep_regularity_index)
are all `how="left"` onto the wearable index or computed from it, so none of
them change which bins exist. Skipping them avoids recomputing SRI for ~200
participants for a measurement that cannot depend on it.

The imputed-flag semantics are worth restating because they cut the other way
from what "coverage" suggests: analysis_dataframe LOCF-fills gaps and records
the fact in <channel>_imputed, so a bin can have a non-null heart_rate that is
NOT real. Tier 3 counts real bins only. `full/imputed` -- the arm phase 2 is
compared against -- feeds the model the LOCF-filled values, so tier 3 is the
pessimistic view of the same data, and both are printed.

OUTPUT
======
model_input/personalization/history_coverage_cov_<split>.csv, one row per
participant, plus a summary to stderr. Nothing here is consumed by a model;
it exists so the arms' feasibility is a measured number in a file rather than
an assumption.
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_participant_grid import (  # noqa: E402
    ANALYSIS_DIR, BIN_MINUTES, channel_is_real_mask, load_cleaned_glucose_binned,
)
from windows import CONTEXT_BINS  # noqa: E402

REPO_ROOT = os.environ["AIREADI_ROOT"]
PERS_DIR = f"{REPO_ROOT}/model_input/personalization"

HISTORY_DAYS = int(os.environ.get("CHRONOS2_PERS_HISTORY_DAYS", "7"))
SPLIT = os.environ.get("CHRONOS2_PERS_SPLIT", "test")
# CHRONOS2_PERS_HISTORY_BAR_DAYS: the parallel family's window set. Worth
# measuring separately and NOT assumed to be better than bar=7 just because the
# slice is shorter: a lower bar moves every anchor EARLIER, and the five
# thin-coverage participants at bar=7 were thin precisely because their wearable
# record starts 4-6 days after their glucose record. An earlier anchor can
# therefore land before the wearable data begins for MORE participants, not
# fewer.
_BAR = int(os.environ.get("CHRONOS2_PERS_HISTORY_BAR_DAYS", "7"))
_BAR_SUFFIX = "" if _BAR == 7 else f"_bar{_BAR}d"

# The four phase-2 arms' channels (CHRONOS2_PERSONALIZATION.md). Measured
# individually so a channel that is the reason an arm is infeasible is
# identifiable, rather than only the arm being visibly infeasible.
FIVE_PHYS = ["heart_rate", "stress", "respiratory_rate",
             "physical_activity_steps", "physical_activity_calorie_rate"]
EXTRA_CHANNELS = ["oxygen_saturation", "sleep_stage"]
# The two channels window eligibility itself gates on (windows.py). A history
# slice thin in these is the specific case where phase 2 would be fitting on
# something the export would not have called an eligible window.
ELIGIBILITY_CHANNELS = ["heart_rate", "respiratory_rate"]

# Smallest history Chronos2Dataset can sample a single training window from,
# given min_past = context_length = 144 (chronos2_personalization.py). fit@24
# is the longer of the two horizon-matched fit lengths, so it is the binding
# one; fit@6 needs 150.
FIT_LENGTHS = {"fit@6": 6, "fit@24": 24}


def slice_coverage(person_id, history_start, anchor):
    expected_bins = int((anchor - history_start).total_seconds() // (BIN_MINUTES * 60))
    row = {"person_id": person_id, "expected_bins": expected_bins,
           "anchor": anchor, "history_start": history_start}

    path = os.path.join(ANALYSIS_DIR, f"{person_id}.csv")
    if not os.path.exists(path):
        row["grid_bins"] = 0
        return row
    w = pd.read_csv(path)
    if w.empty:
        row["grid_bins"] = 0
        return row
    w["bin_start"] = pd.to_datetime(w["bin_start"], utc=True)
    row["wearable_start"] = w["bin_start"].min()
    row["wearable_end"] = w["bin_start"].max()
    # how the wearable span sits relative to the slice, signed in days:
    # positive lead = wearable data starts before the slice does (good),
    # positive lag = wearable data ends after the slice does (good)
    row["wearable_lead_days"] = (history_start - row["wearable_start"]).total_seconds() / 86400.0
    row["wearable_lag_days"] = (row["wearable_end"] - anchor).total_seconds() / 86400.0

    g = load_cleaned_glucose_binned(person_id)
    w = w.set_index("bin_start")
    w = w.join((g.rename("cgm_glucose") if g is not None else
                pd.Series(dtype="float64", name="cgm_glucose")), how="left")
    if "cgm_glucose" not in w.columns:
        w["cgm_glucose"] = np.nan
    w["cgm_glucose_imputed"] = w["cgm_glucose"].isna()

    sl = w[(w.index >= history_start) & (w.index < anchor)]
    row["grid_bins"] = len(sl)
    if sl.empty:
        return row
    # a duplicated or off-grid wearable timestamp would make grid_bins
    # meaningless as a coverage number; checked rather than assumed
    row["grid_bins_unique"] = int(sl.index.nunique())
    row["target_real_bins"] = int((~sl["cgm_glucose_imputed"]).sum())
    for ch in FIVE_PHYS + EXTRA_CHANNELS:
        if ch not in sl.columns:
            row[f"real_{ch}"] = 0
            row[f"nonnull_{ch}"] = 0
            continue
        row[f"real_{ch}"] = int(channel_is_real_mask(sl, ch).sum()) \
            if ch != "sleep_stage" else int(sl[ch].notna().sum())
        row[f"nonnull_{ch}"] = int(sl[ch].notna().sum())
    row["real_both_eligibility"] = int(
        np.logical_and.reduce([channel_is_real_mask(sl, c) for c in ELIGIBILITY_CHANNELS]).sum()
    )
    return row


def main():
    path = f"{PERS_DIR}/eval_set_cov_{SPLIT}{_BAR_SUFFIX}.csv"
    if not os.path.exists(path):
        raise SystemExit(f"{path} missing -- run "
                          f"build_personalization_eval_set.py --cov {SPLIT}")
    e = pd.read_csv(path, dtype={"person_id": str})
    hist_col = f"history_start_{HISTORY_DAYS}d"
    for col in ["anchor", hist_col]:
        e[col] = pd.to_datetime(e[col], utc=True)
    anchors = e.groupby("person_id").agg(anchor=("anchor", "first"),
                                          history_start=(hist_col, "first"))

    rows = [slice_coverage(pid, r["history_start"], r["anchor"])
            for pid, r in anchors.iterrows()]
    cov = pd.DataFrame(rows).sort_values("person_id").reset_index(drop=True)
    for c in ["grid_bins", "grid_bins_unique", "target_real_bins", "real_both_eligibility"]:
        if c in cov.columns:
            cov[c] = cov[c].fillna(0).astype(int)
    cov["grid_frac"] = cov["grid_bins"] / cov["expected_bins"]
    cov["target_frac"] = cov["target_real_bins"] / cov["expected_bins"]
    out = f"{PERS_DIR}/history_coverage_cov_{SPLIT}{_BAR_SUFFIX}_k{HISTORY_DAYS}d.csv"
    cov.to_csv(out, index=False)

    n = len(cov)
    print(f"\n=== phase-2 covariate history coverage, split={SPLIT}, "
          f"k={HISTORY_DAYS}d ({cov['expected_bins'].iloc[0]} bins) ===")
    print(f"participants in eval_set_cov_{SPLIT}{_BAR_SUFFIX}.csv: {n}")
    if (cov["grid_bins_unique"] != cov["grid_bins"]).any():
        print("WARNING: duplicated wearable timestamps in some slices -- "
              "grid_bins overstates coverage for those participants")

    print("\ntier 1 GRID (bins present in analysis_dataframe):")
    for thr in [1.0, 0.999, 0.99, 0.95, 0.90, 0.75, 0.50]:
        k = int((cov["grid_frac"] >= thr).sum())
        print(f"  >= {thr:6.1%} of the slice: {k:4d} / {n}  ({k / n:6.1%})")
    print(f"  grid_frac: median {cov['grid_frac'].median():.4f}, "
          f"mean {cov['grid_frac'].mean():.4f}, min {cov['grid_frac'].min():.4f}")

    print("\ntier 2 TARGET (real cgm_glucose on the wearable grid):")
    print(f"  target_frac: median {cov['target_frac'].median():.4f}, "
          f"mean {cov['target_frac'].mean():.4f}, min {cov['target_frac'].min():.4f}")
    for name, need in FIT_LENGTHS.items():
        k = int((cov["grid_bins"] >= CONTEXT_BINS + need).sum())
        print(f"  can sample >=1 training window at {name} "
              f"(grid_bins >= {CONTEXT_BINS + need}): {k} / {n}")

    print("\ntier 3 CHANNELS (real fraction of the whole slice, median over participants):")
    for ch in FIVE_PHYS + EXTRA_CHANNELS:
        real = (cov[f"real_{ch}"] / cov["expected_bins"])
        nonnull = (cov[f"nonnull_{ch}"] / cov["expected_bins"])
        zero = int((cov[f"real_{ch}"] == 0).sum())
        print(f"  {ch:36s} real median {real.median():.3f}  "
          f"non-null (LOCF-incl.) median {nonnull.median():.3f}  "
          f"entirely-absent participants {zero}")
    both = cov["real_both_eligibility"] / cov["expected_bins"]
    print(f"  {'heart_rate AND respiratory_rate':36s} real median {both.median():.3f}")

    # Per-horizon, because the 232 above is the UNION over horizons and the
    # number that gets reported alongside a result is the one for that
    # horizon's own scored population (194 participants / 409 windows at 1h).
    print("\nper-horizon (the population each horizon's result is actually reported on):")
    frac_by_pid = cov.set_index("person_id")["grid_frac"]
    for horizon in ["30min", "1h", "2h", "3h", "4h"]:
        sub = e[e["horizon"] == horizon]
        pids = sub["person_id"].unique()
        f = frac_by_pid.reindex(pids)
        print(f"  {horizon:5s} {sub['key'].nunique():5d} windows  {len(pids):4d} participants  "
              f"full {HISTORY_DAYS}d grid: {int((f >= 1.0).sum()):4d}  >=95%: {int((f >= 0.95).sum()):4d}  "
              f"median {f.median():.4f}")

    thin = cov[cov["grid_frac"] < 0.95].sort_values("grid_frac")
    if not thin.empty:
        print(f"\n{len(thin)} participant(s) below 95% grid coverage -- these fit on a "
              "history whose covariate columns are mostly NaN:")
        print(thin[["person_id", "grid_bins", "expected_bins", "grid_frac",
                     "target_frac", "wearable_lead_days", "wearable_lag_days"]]
              .to_string(index=False))

    print(f"\nwritten: {out}")


if __name__ == "__main__":
    main()
