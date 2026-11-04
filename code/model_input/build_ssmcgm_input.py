"""SSM-CGM training (official train+val split) and eval (official test
split) inputs, built from this project's own real cleaned/merged data plus
real participants.tsv covariates -- replacing every placeholder the earlier
ssmcgm_real_data_smoketest.py used (age=0.0, clinical_site="site_0",
study_group="unknown", predmeal_flag=0.0 as a genuine feature).

Deviation from SSM-CGM's own published setup (arXiv:2510.04386,
create_tft_dataloaders in code/external/SSM-CGM/SSM_CGM.py), disclosed here and in
report.md: their code lists heartrate, oxygen_saturation, respiration_rate,
stress_level, activity_steps, calories_value, AND sleep_stage as
time_varying_known_reals/categoricals -- i.e. the model is allowed to see
their FUTURE values within the forecast horizon. A real deployment would not
know a person's future heart rate any more than their future glucose, so
this is an oracle assumption that would inflate SSM-CGM's apparent ceiling
relative to Chronos-2/TimesFM/CGM-LSM, none of which get this advantage in
this project's zero-shot exports (export_zeroshot_windows.py blanks
covariates during the horizon for the same reason). Fixed here by moving all
of them to time_varying_unknown_reals/categoricals (past-observed only).
predmeal_flag (added 2026-08-28, replacing the earlier disclosed placeholder
that always read 0.0): now a real, causally-generated meal probability from
code/meal_detection/generate_meal_probability.py, a CNN+BiLSTM detector
trained on the real CGMacros dataset (PhysioNet, DOI 10.13026/3z8q-x658),
reproducing the architecture actually implemented in SSM-CGM's own
MealDetection/mealdetection.ipynb (their labeling/windowing code itself is
not in that repo -- see cgmacros_dataset.py's module docstring for the
disclosed, necessary deviations this forced). Moved to time_varying_
unknown_real, NOT left as a known real despite the original paper's own
schema treating it as one: predmeal_flag[t] is only causal relative to its
OWN timestamp t (generate_meal_probability.py's module docstring -- it uses
glucose up to and including t, nothing later). That is a different claim
from "known at forecast time" once t falls inside the horizon: a real
deployment forecasting forward from the end of the context window does not
yet have the realized future glucose predmeal_flag's horizon-period values
would depend on, so supplying it as a known real would smuggle in exactly
the same future-glucose oracle this project already removed for heart rate/
oxygen_saturation/etc. above -- the fact that its OWN generating function
never looks past t does not make it available at the actual moment of
inference for t's still in the future relative to that moment. Only its
context-period values are real information a forecast could actually use.

LOCF (ffill/bfill) applied same as the wearable covariates; NaN can occur
even mid-series (any bin lacking a full 72-bin real causal glucose window,
e.g. near a real coverage gap), a genuinely harder case than "zero coverage
for this whole participant" (oxygen_saturation's fallback, above, is -1.0,
not 0.0 -- see that section). predmeal_flag's own 0.0 fallback (below) is
disclosed as fine here since "no detectable meal signal" is itself
a meaningful default, not a clinically implausible one the way 0% SpO2
would be.

ds/minute_of_day/tod_sin/tod_cos are index/calendar features, always
genuinely known in advance, and are left as known reals per the original
paper.
"""
import glob
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from build_participant_grid import build_grid, load_participants_covariates  # noqa: E402

REPO_ROOT = os.environ["AIREADI_ROOT"]
ANALYSIS_DIR = f"{REPO_ROOT}/analysis_dataframe"
OUTPUT_DIR = f"{REPO_ROOT}/model_input/ssmcgm"

STATIC_CATEGORICALS = ["participant_id", "clinical_site", "study_group"]
STATIC_REALS = ["age"]
TIME_VARYING_KNOWN_REALS = ["ds", "minute_of_day", "tod_sin", "tod_cos", "oxygen_saturation_applicable"]
TIME_VARYING_UNKNOWN_CATEGORICALS = ["sleep_stage"]
# never filled: dropping a row for missing any of these ("clipped to overlapping
# CGM coverage") rather than fabricating the forecasting target, see
# build_participant_series's docstring.
TIME_VARYING_UNKNOWN_REALS_CGM_ONLY = [
    "cgm_glucose", "cgm_lag_1", "cgm_lag_3", "cgm_lag_6",
    "cgm_diff_lag_1", "cgm_diff_lag_3", "cgm_diff_lag_6",
    "cgm_lagdiff_1_3", "cgm_lagdiff_3_6", "cgm_rolling_mean", "cgm_rolling_std",
]
TIME_VARYING_UNKNOWN_REALS = [
    "cgm_glucose", "cgm_lag_1", "cgm_lag_3", "cgm_lag_6",
    "cgm_diff_lag_1", "cgm_diff_lag_3", "cgm_diff_lag_6",
    "cgm_lagdiff_1_3", "cgm_lagdiff_3_6", "cgm_rolling_mean", "cgm_rolling_std",
    "heartrate", "oxygen_saturation", "respiration_rate", "stress_level",
    "activity_steps", "calories_value", "predmeal_flag", "sleep_regularity_index",
]
KEEP_COLS = (["series_id"] + STATIC_CATEGORICALS + STATIC_REALS +
             TIME_VARYING_KNOWN_REALS + TIME_VARYING_UNKNOWN_CATEGORICALS + TIME_VARYING_UNKNOWN_REALS)


def build_participant_series(person_id, cov, cutoff=None, time_range=None):
    """Training/validation policy, distinct from the eval-side GlyLLM coverage
    rule (windows.py): follows SSM-CGM's own published preprocessing (arXiv:
    2510.04386, "Missing values were imputed via Last Observation Carried
    Forward and series clipped to intervals with overlapping CGM-wearable
    coverage") rather than the stricter eval bar, since training data does
    not need trustworthy ground truth the way an evaluation target does.

    Wearable covariates are forward- then backward-filled with no gap limit
    (LOCF, matching the cited method) since they only ever inform the
    forecast, never define it. cgm_glucose and its own lag/diff/rolling
    features are never filled: fabricating the forecasting target would
    misrepresent what the model is actually learning from. Rows lacking a
    real glucose reading, or lacking a real value for one of its lag/rolling
    features because the referenced timestep itself has no reading, are
    dropped rather than imputed ("clipped to overlapping CGM coverage").
    `ds` keeps the original per-participant bin index instead of being
    renumbered after dropping rows, so a gap in time reads as a gap in `ds`;
    each participant's kept rows are then split into contiguous `series_id`
    segments at every such gap, since `TimeSeriesDataSet` needs a value at
    every position within a window, not just a monotonic index (see
    ssmcgm_input_smoketest.py and report.md Sec 15.5).

    `oxygen_saturation` needs a second-level fallback on top of LOCF: 273 of
    1,355 train participants (20.1%) have zero real oxygen_saturation
    readings anywhere in their file at all (it is only ever measured in a
    narrow overnight window, Sec 10.7/13.1), so ffill/bfill has nothing to
    propagate from for them -- confirmed directly against
    ssmcgm_timing_dryrun.py hitting this exact case (22.20% residual NaN
    after LOCF, at the row level). Remaining NaN after LOCF is filled with
    -1.0 (CORRECTED 2026-09-01: this docstring previously said 0.0, which
    contradicts the actual fill value below and HANDOFF.md's own record of
    this fix -- 0.0 would read as a real, physiologically catastrophic SpO2
    of 0%, exactly the ambiguity -1.0 was chosen to avoid), a value outside
    oxygen_saturation's real [50,100] range so it is never mistaken for a
    real reading, paired with the already-existing
    `oxygen_saturation_applicable` calendar flag (included here as a known
    real, since it is a timezone-derived fact knowable in advance, the same
    treatment chronos2_zeroshot_eval.py already gives it) so a consumer can
    tell a real 0-region apart from a filled one."""
    df = build_grid(person_id, {person_id: cov})
    if df is None or df.empty:
        return None

    df = df.sort_values("bin_start").reset_index(drop=True)
    n = len(df)
    df["participant_id"] = str(person_id)
    df["clinical_site"] = cov.get("clinical_site", "")
    df["study_group"] = cov.get("study_group", "")
    df["age"] = cov.get("age", np.nan)
    # real, causally-generated (predmeal_prob joined in by build_grid); LOCF
    # then 0.0 fallback, see module docstring.
    df["predmeal_flag"] = df["predmeal_prob"].ffill().bfill().fillna(0.0)

    df["ds"] = np.arange(n)  # original bin index, preserved even after rows are dropped below
    df["minute_of_day"] = (df["bin_start"].dt.hour * 60 + df["bin_start"].dt.minute).astype(float)
    tod = df["minute_of_day"] / 1440.0
    df["tod_sin"] = np.sin(2 * np.pi * tod)
    df["tod_cos"] = np.cos(2 * np.pi * tod)

    df["activity_steps"] = df["physical_activity_steps"].fillna(0.0)
    df["calories_value"] = df["physical_activity_calorie_rate"].fillna(0.0)
    df["oxygen_saturation_applicable"] = df["oxygen_saturation_applicable"].fillna(False).astype(float)
    # -1, not 0: 0% SpO2 is life-incompatible, so it reads as a real clinical
    # catastrophe rather than an obviously-fake placeholder to anyone (or
    # any downstream code) that doesn't check oxygen_saturation_applicable.
    # -1 matches this project's own existing sentinel convention for other
    # streams (stress/respiratory_rate's -1/-2 device codes, report.md
    # Sec 2) -- unambiguous in either direction, can never be mistaken for
    # a real reading. Checked directly whether a still-constant column (a
    # participant with zero real oxygen_saturation anywhere gets one fixed
    # value for their entire series either way) causes a divide-by-zero in
    # TimeSeriesDataSet's normalization: no NaN/Inf appeared in the encoded
    # tensor across 30 real batches including such a participant, so this
    # is a plausibility fix, not a numerical-safety one.
    df["oxygen_saturation"] = df["oxygen_saturation"].ffill().bfill().fillna(-1.0)
    # heartrate/respiration_rate/stress_level: SAME zero-coverage gap as
    # oxygen_saturation, just smaller (1/3/2 train participants respectively
    # have no real reading anywhere in their file at all, per build_ssmcgm_
    # input_dropna_oxysat.py's own docstring finding on this shared issue)
    # -- ffill/bfill alone leaves these NaN for that handful of participants,
    # which TimeSeriesDataSet's _check_tensors rejects outright regardless of
    # how few rows are affected (caught directly: 11 real rows out of 3.5M,
    # confirmed via ssmcgm_train_real.py's smoke test before this was fixed,
    # not caught by the smaller-scale smoketests earlier in this project).
    # Same -1.0 sentinel as oxygen_saturation: outside each channel's real
    # range (heart rate/respiration rate are never negative; stress's own
    # scale is 0-100, where 0 is itself a real reading, so -1 avoids
    # colliding with a genuine value the way 0 would for stress specifically).
    df["heartrate"] = df["heart_rate"].ffill().bfill().fillna(-1.0)
    df["respiration_rate"] = df["respiratory_rate"].ffill().bfill().fillna(-1.0)
    df["stress_level"] = df["stress"].ffill().bfill().fillna(-1.0)
    df["sleep_stage"] = df["sleep_stage"].fillna("unknown")
    # sleep_regularity_index (added 2026-08-28): NOT part of SSM-CGM's own
    # published schema -- a project addition (build_participant_grid.
    # compute_sleep_regularity_index, real Phillips et al. 2017 SRI),
    # included here for the same "ablate every variable across every model"
    # consistency this project is applying to Chronos-2/TimesFM too.
    # -200.0 fallback, not 0.0: the real Phillips scale is -100..+100, and
    # 0.0 is itself a meaningful reading on it ("as regular as chance"), the
    # same reasoning as oxygen_saturation's -1.0 above.
    df["sleep_regularity_index"] = df["sleep_regularity_index"].ffill().bfill().fillna(-200.0)

    g = df["cgm_glucose"]
    df["cgm_lag_1"] = g.shift(1)
    df["cgm_lag_3"] = g.shift(3)
    df["cgm_lag_6"] = g.shift(6)
    df["cgm_diff_lag_1"] = g - df["cgm_lag_1"]
    df["cgm_diff_lag_3"] = g - df["cgm_lag_3"]
    df["cgm_diff_lag_6"] = g - df["cgm_lag_6"]
    df["cgm_lagdiff_1_3"] = df["cgm_lag_1"] - df["cgm_lag_3"]
    df["cgm_lagdiff_3_6"] = df["cgm_lag_3"] - df["cgm_lag_6"]
    df["cgm_rolling_mean"] = g.rolling(6, min_periods=1).mean()
    df["cgm_rolling_std"] = g.rolling(6, min_periods=1).std()

    df = df.dropna(subset=TIME_VARYING_UNKNOWN_REALS_CGM_ONLY)
    if df.empty:
        return None

    # cutoff (added 2026-09-03, for the SSM-CGM paper-style personalization
    # experiment): keeps only rows strictly before a real UTC timestamp --
    # each held-out test participant's own causal anchor (model_input/
    # personalization/eval_set_test.csv), the same anchor Stephanie's LoRA
    # arm and the TimesFM in-context arm both use. Applied here, BEFORE the
    # gap/series_id derivation below, so a truncated participant's kept rows
    # get correctly re-segmented (the tail past the cutoff simply doesn't
    # exist, not a hole inside a segment) rather than filtering an
    # already-built ds/series_id column, which would leave both referring
    # to a series that no longer matches its own row count. `ds` itself is
    # still the ORIGINAL per-participant bin index at this point (see
    # comment below), so this filter and the ordinary no-cutoff path share
    # every line of logic up to here -- no separate code path to keep in
    # sync.
    if cutoff is not None:
        df = df[df["bin_start"] < cutoff]
        if df.empty:
            return None

    # time_range (added 2026-09-05, for the two SSM-CGM population-wide
    # temporal-split experiments): keeps rows with start <= bin_start < end,
    # either bound optionally None (unbounded). Same placement rationale as
    # `cutoff` above -- applied before ds/series_id re-derivation so a
    # participant's kept rows are correctly re-segmented, not filtered after
    # the fact. Unlike `cutoff` (a single absolute anchor shared across
    # participants), time_range's bounds are computed PER PARTICIPANT by the
    # caller (build_ssmcgm_population_split_input.py), from that person's
    # own real recording span -- necessary since AI-READI participants start
    # and end device wear on different calendar dates.
    if time_range is not None:
        start, end = time_range
        if start is not None:
            df = df[df["bin_start"] >= start]
        if end is not None:
            df = df[df["bin_start"] < end]
        if df.empty:
            return None

    # TimeSeriesDataSet's allow_missing_timesteps only tells it gaps in `ds`
    # exist; it does not skip windows that span one, and still requires a
    # value at every position such a window would cover -- tried and
    # confirmed via ssmcgm_input_smoketest.py (37.54% "NaN" cgm_glucose even
    # after the dropna above, all of it from windows spanning a gap). The
    # actual fix: split each participant's kept rows into separate
    # contiguous segments at every gap, so no window is ever asked to span
    # one. `series_id` (not `participant_id`) is the TimeSeriesDataSet group,
    # so a person with 3 segments is 3 independent series; `participant_id`
    # stays their real ID and is still passed as a static categorical, so
    # the model's per-person embedding is shared correctly across their
    # segments instead of being reset by the split.
    gap = df["ds"].diff().fillna(1)
    segment = (gap > 1).cumsum()
    df["series_id"] = df["participant_id"] + "_" + segment.astype(str)
    df["ds"] = df.groupby("series_id").cumcount()

    return df[KEEP_COLS]


def run():
    covs = load_participants_covariates()
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    paths = sorted(glob.glob(f"{ANALYSIS_DIR}/*.csv"))
    pids = [os.path.basename(p).replace(".csv", "") for p in paths]

    frames = {"train": [], "val": [], "test": []}
    n_by_split = {"train": 0, "val": 0, "test": 0, "": 0}
    for i, pid in enumerate(pids):
        cov = covs.get(str(pid), {})
        split = cov.get("recommended_split", "")
        n_by_split[split] = n_by_split.get(split, 0) + 1
        if split not in frames:
            continue
        series = build_participant_series(pid, cov)
        if series is None:
            continue
        frames[split].append(series)
        if (i + 1) % 300 == 0:
            print(f"  processed {i + 1}/{len(pids)}", file=sys.stderr)

    # separate files per official split (never a merged train+val file --
    # see report.md Sec 15.4 / the leakage discussion this was fixed after):
    # SSM-CGM trains on train, validates on val, is evaluated on test,
    # the same test population as every other model.
    for split in ("train", "val", "test"):
        df = pd.concat(frames[split], ignore_index=True)
        df.to_csv(os.path.join(OUTPUT_DIR, f"ssmcgm_{split}.csv"), index=False)
        print(f"{split}: {df['participant_id'].nunique()} participants, {df['series_id'].nunique()} segments, {len(df)} rows",
              file=sys.stderr)

    print(f"participants by recommended_split: {n_by_split}", file=sys.stderr)


if __name__ == "__main__":
    run()
