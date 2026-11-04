"""Shared per-participant grid every model-input builder reads from: joins
this project's own analysis_dataframe/<person_id>.csv (7 cleaned wearable
streams, 5-min bins) with cleaned glucose (final_cleaned_data/glucose,
resampled onto the same grid) and real per-participant covariates from
participants.tsv (age, clinical_site, study_group, recommended_split) --
replacing the placeholder age=0.0/clinical_site="site_0"/study_group="unknown"
the earlier ssmcgm_real_data_smoketest.py used.

Coverage/completeness rule (per model-input builder, applied to whichever
channels that model actually consumes): adopted from GlyLLM (arXiv:2606.12699,
the paper most directly comparable -- same dataset, same forecasting task),
not the generic Cichosz et al. (2025) 70%-CGM-coverage figure already cited in
report.md for the population-level dataframe: a window is eligible only if,
for every channel it uses, (a) the longest continuous gap within the window
is <=60 minutes, and (b) the overall fraction missing is <=5%. Stricter than
Cichosz's number and specific to this exact task, which matters more here
than at the population level: this is the layer that actually decides what a
model trains/evaluates on, not just what final_cleaned_data retains.

cgm_glucose is never filled/interpolated anywhere in this module -- a short-
gap linear interpolation was tried and deliberately reverted: computed once
on this shared per-participant grid, before any window boundary exists, it
risked filling a gap in what later becomes one window's context using a real
value that later becomes that same window's own horizon/target (or, for
SSM-CGM, an equally uncontrolled boundary from its own internal windowing).
A leak-safe version would need to run per-window, strictly confined to one
side of the context/horizon split, which is a per-model-input-builder concern,
not something this shared grid should do on everyone's behalf. Left as raw
NaN; eligible windows already bound any surviving gap to <=60min/<=5%
missing via the coverage rule above.
"""
import csv
import itertools
import os

import numpy as np
import pandas as pd

REPO_ROOT = os.environ["AIREADI_ROOT"]
# ANALYSIS_DIR_OVERRIDE: lets a caller point at a different analysis_dataframe
# variant (e.g. the Chronos-2/TimesFM-only non-LOCF'd wearables build) without
# changing the default every other consumer (SSM-CGM's build_ssmcgm_input.py,
# build_window_index.py) reads.
ANALYSIS_DIR = os.environ.get("ANALYSIS_DIR_OVERRIDE", f"{REPO_ROOT}/analysis_dataframe")
GLUCOSE_DIR = f"{REPO_ROOT}/final_cleaned_data/glucose"
PARTICIPANTS_TSV = f"{REPO_ROOT}/aireadi_dataset/dataset/participants.tsv"

BIN_MINUTES = 5
# predmeal_prob: precomputed by code/meal_detection/generate_meal_probability.py
# (a real, causally-applied CNN+BiLSTM meal detector trained on CGMacros,
# replacing the earlier disclosed placeholder predmeal_flag=0.0 in
# build_ssmcgm_input.py) -- one CSV per participant, same convention as
# analysis_dataframe's own per-participant files. Read here, not computed
# inline: the detector needs PyTorch/a GPU, which most build_grid() callers
# (plain-pandas, CPU-only venvs) don't carry, see that script's docstring.
PREDMEAL_PROB_DIR = os.environ.get(
    "PREDMEAL_PROB_DIR_OVERRIDE", f"{REPO_ROOT}/model_input/meal_detection/predmeal_prob"
)

# Channels this project's own analysis_dataframe already flags real-vs-imputed for.
IMPUTED_FLAG_CHANNELS = ["heart_rate", "stress", "respiratory_rate", "oxygen_saturation"]
# Channels with no separate imputed flag (steps summed+zero-filled, calorie rate
# always populated by construction, sleep only forward-filled within its own
# session span) -- coverage here is approximated by NaN presence, disclosed as
# weaker than the flagged channels since zero-fill can mask true non-wear.
NAN_ONLY_CHANNELS = ["physical_activity_steps", "physical_activity_calorie_rate", "sleep_asleep"]
# Trailing window (Phillips et al. 2017, Sci Rep 7:3216 -- the paper that
# defines the Sleep Regularity Index) for compute_sleep_regularity_index()
# below; 7 days is that paper's own convention and comfortably fits AI-READI's
# 8-10 day recordings while still leaving >=1 day of history to score against.
SLEEP_REGULARITY_WINDOW_DAYS = 7


def load_participants_covariates():
    rows = {}
    with open(PARTICIPANTS_TSV, newline="") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            rows[row["person_id"]] = {
                "age": float(row["age"]) if row.get("age") not in (None, "") else np.nan,
                "clinical_site": row.get("clinical_site", ""),
                "study_group": row.get("study_group", ""),
                "recommended_split": row.get("recommended_split", ""),
            }
    return rows


def load_cleaned_glucose_binned(person_id):
    path = os.path.join(GLUCOSE_DIR, f"{person_id}.csv")
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    if df.empty:
        return None
    df["final_timestamp"] = pd.to_datetime(df["final_timestamp"], errors="coerce", utc=True)
    df = df.dropna(subset=["final_timestamp", "final_value"])
    if df.empty:
        return None
    s = df.set_index("final_timestamp")["final_value"].sort_index()
    binned = s.resample(f"{BIN_MINUTES}min").mean()
    # a bin is "real" (cgm_glucose_imputed=False) only if at least one kept
    # glucose reading actually falls in it -- no fill applied here or in
    # build_grid(), gaps stay NaN (see module docstring for why).
    return binned


def load_predmeal_prob(person_id):
    """None if generate_meal_probability.py hasn't been run for this person
    yet (or at all) -- callers must treat this the same as any other
    optional real channel (fillna downstream, never fabricate here)."""
    path = os.path.join(PREDMEAL_PROB_DIR, f"{person_id}.csv")
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    df["bin_start"] = pd.to_datetime(df["bin_start"], utc=True)
    return df.set_index("bin_start")["predmeal_prob"]


def compute_sleep_regularity_index(df):
    """Sleep Regularity Index (Phillips et al. 2017, Sci Rep 7:3216): for
    day d, compares the binary sleep/wake state at every matching clock-time
    bin across each CONSECUTIVE pair of days (day j vs day j+1, per their
    actual formula -- SRI = -100 + 200/(M(N-1)) * sum_j sum_i
    1[s(i,j)=s(i,j+1)], a sum over N-1 adjacent-day pairs, NOT every
    pairwise combination of days) in a trailing window of up to
    SLEEP_REGULARITY_WINDOW_DAYS days STRICTLY BEFORE d (never including d
    itself) -- causal by construction, since day d's own not-yet-elapsed
    sleep/wake behavior is never used to score day d.

    CORRECTED 2026-08-28: the first version of this function used
    itertools.combinations (every pair of days in the window, e.g. 21 pairs
    for a 7-day window) instead of consecutive pairs (6 pairs) -- a
    different statistic (overall day-to-day similarity across the whole
    window, not night-to-night transition consistency), caught on direct
    review, not a subtle numerical difference.

    Score is on Phillips' original -100..+100 scale (+100 = identical
    sleep/wake timing on every consecutive day in the window; 0 is the
    expected value for independent/unrelated days). One value per calendar
    day, broadcast to every bin of that day.

    NaN for a participant's first day (or any day with fewer than 2 prior
    days available to compare) and for `sleep_asleep`-derived days with no
    bins in common between a given pair (handled per-pair, not per-day).

    Uses build_grid()'s own `bin_start` (UTC) directly for both day
    boundaries and intraday clock-time position -- the same convention this
    project's own tod_sin/tod_cos/minute_of_day features already use
    (build_ssmcgm_input.py), not a per-participant local-timezone conversion.

    sleep_asleep NaN (not currently in a recorded sleep session -- the
    common case outside of sleep) is treated as "awake" (0), inheriting the
    same disclosed limitation NAN_ONLY_CHANNELS already carries elsewhere in
    this module (zero-fill can mask true non-wear, not just genuine
    wakefulness) rather than introducing a new one."""
    is_asleep = df["sleep_asleep"].fillna(0.0).to_numpy()
    calendar_day = df["bin_start"].dt.date
    intraday_bin = ((df["bin_start"].dt.hour * 60 + df["bin_start"].dt.minute) // BIN_MINUTES).astype(int).to_numpy()

    day_vectors = {}
    for day, idx in df.groupby(calendar_day).groups.items():
        pos = df.index.get_indexer(idx)
        day_vectors[day] = pd.Series(is_asleep[pos], index=intraday_bin[pos])

    days_sorted = sorted(day_vectors)
    sri_by_day = {}
    for i, day in enumerate(days_sorted):
        window_days = days_sorted[max(0, i - SLEEP_REGULARITY_WINDOW_DAYS):i]  # strictly before day i
        match_rates = []
        for a, b in zip(window_days, window_days[1:]):  # consecutive pairs only, see docstring
            va, vb = day_vectors[a], day_vectors[b]
            common = va.index.intersection(vb.index)
            if len(common) == 0:
                continue
            match_rates.append(float((va.loc[common].to_numpy() == vb.loc[common].to_numpy()).mean()))
        sri_by_day[day] = (200.0 * np.mean(match_rates) - 100.0) if match_rates else np.nan

    return calendar_day.map(sri_by_day)


def build_grid(person_id, covariates_by_pid):
    wearable_path = os.path.join(ANALYSIS_DIR, f"{person_id}.csv")
    if not os.path.exists(wearable_path):
        return None
    wearable = pd.read_csv(wearable_path)
    if wearable.empty:
        return None
    wearable["bin_start"] = pd.to_datetime(wearable["bin_start"], utc=True)

    glucose_binned = load_cleaned_glucose_binned(person_id)
    if glucose_binned is None:
        return None

    # left join on the wearable grid (not outer): a "with covariates" window
    # needs wearable data present regardless, and this avoids injecting rows
    # outside analysis_dataframe's own span where every _imputed/_applicable
    # flag would otherwise be NaN rather than a real True/False.
    df = wearable.set_index("bin_start").join(glucose_binned.rename("cgm_glucose"), how="left")
    df.index.name = "bin_start"
    df = df.sort_index().reset_index()
    df["person_id"] = person_id
    # coverage.py's eligibility check keys off this flag, not off whether
    # cgm_glucose itself is NaN -- cgm_glucose is never filled (see module
    # docstring), so the two are equivalent here, but this keeps the same
    # shape as the other channels' real/imputed flags.
    df["cgm_glucose_imputed"] = df["cgm_glucose"].isna()
    df["sleep_regularity_index"] = compute_sleep_regularity_index(df)

    predmeal_prob = load_predmeal_prob(person_id)
    if predmeal_prob is not None:
        df = df.set_index("bin_start").join(predmeal_prob, how="left").reset_index()
    else:
        df["predmeal_prob"] = np.nan

    cov = covariates_by_pid.get(str(person_id), {})
    df["age"] = cov.get("age", np.nan)
    df["clinical_site"] = cov.get("clinical_site", "")
    df["study_group"] = cov.get("study_group", "")
    df["recommended_split"] = cov.get("recommended_split", "")

    return df


def channel_is_real_mask(df, channel):
    """True where this channel's bin is a real (non-imputed, non-missing)
    observation -- used by the coverage rule in coverage.py."""
    if channel == "cgm_glucose":
        return ~df["cgm_glucose_imputed"]
    if channel in IMPUTED_FLAG_CHANNELS:
        # outer join against glucose's own timestamp range (build_grid) can
        # introduce rows outside analysis_dataframe's original span, where
        # the _imputed flag is NaN rather than True/False -- `==False` (not
        # `~`) treats that correctly as "not known real" without raising.
        return (df[f"{channel}_imputed"] == False) & df[channel].notna()  # noqa: E712
    if channel in NAN_ONLY_CHANNELS:
        return df[channel].notna()
    raise ValueError(f"unknown channel: {channel}")
