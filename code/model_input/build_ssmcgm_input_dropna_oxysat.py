"""Variant of build_ssmcgm_input.py that DROPS every participant with zero
real coverage on any of the four LOCF'd wearable channels (heartrate,
oxygen_saturation, respiration_rate, stress_level), instead of a collaborator's
sentinel-fill approach now live in build_ssmcgm_input.py (fillna(-1.0) +
oxygen_saturation_applicable as a real/filled flag, keeping all 1,355 train
participants). Both approaches address the same underlying finding --
273/1,355 train participants (20.1%) have zero real oxygen_saturation
readings anywhere in their file, so ffill/bfill has nothing to propagate
from, which crashes TimeSeriesDataSet's no-NaN check -- but this project
wanted a side-by-side "drop" alternative rather than overwriting the
sentinel-fill version already shared. Output goes to a SEPARATE directory
(model_input/ssmcgm_dropna_oxysat/) so it does not touch or conflict with
the shared model_input/ssmcgm/ files.

Only oxygen_saturation was reported as having a large zero-coverage
population (273 participants). heartrate/respiration_rate/stress_level have
a much smaller version of the same problem (1/3/2 zero-coverage train
participants respectively, confirmed directly against the shared file) --
left unfixed, those would crash TimeSeriesDataSet the same way, just later.
Dropped by the same rule here for internal consistency (a "drop-if-missing"
policy that only applies to one of four channels sharing the exact same
failure mode would be arbitrary), not because the user asked about those
three specifically.
"""
import glob
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from build_participant_grid import build_grid, load_participants_covariates  # noqa: E402

ANALYSIS_DIR = "/path/to/aireadi-collab/analysis_dataframe"
OUTPUT_DIR = "/path/to/aireadi-collab/model_input/ssmcgm_dropna_oxysat"

STATIC_CATEGORICALS = ["participant_id", "clinical_site", "study_group"]
STATIC_REALS = ["age"]
TIME_VARYING_KNOWN_REALS = ["ds", "minute_of_day", "tod_sin", "tod_cos", "predmeal_flag", "oxygen_saturation_applicable"]
TIME_VARYING_UNKNOWN_CATEGORICALS = ["sleep_stage"]
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
    "activity_steps", "calories_value",
]
KEEP_COLS = (["series_id"] + STATIC_CATEGORICALS + STATIC_REALS +
             TIME_VARYING_KNOWN_REALS + TIME_VARYING_UNKNOWN_CATEGORICALS + TIME_VARYING_UNKNOWN_REALS)

# raw (pre-fill) source columns in the build_grid() output, keyed by the
# renamed feature name that TIME_VARYING_UNKNOWN_REALS actually uses
LOCF_CHANNELS = {
    "heartrate": "heart_rate",
    "oxygen_saturation": "oxygen_saturation",
    "respiration_rate": "respiratory_rate",
    "stress_level": "stress",
}


def build_participant_series(person_id, cov):
    """Same policy as build_ssmcgm_input.py's build_participant_series
    (LOCF for wearables, never-fill for cgm_glucose/its derived features,
    per-gap series_id segmentation) EXCEPT: a participant is dropped
    entirely (returns None) if any of the four LOCF'd wearable channels has
    zero real (non-NaN) readings anywhere in their file -- rather than
    falling back to a sentinel fill."""
    df = build_grid(person_id, {person_id: cov})
    if df is None or df.empty:
        return None

    for _, raw_col in LOCF_CHANNELS.items():
        if df[raw_col].notna().sum() == 0:
            return None

    df = df.sort_values("bin_start").reset_index(drop=True)
    n = len(df)
    df["participant_id"] = str(person_id)
    df["clinical_site"] = cov.get("clinical_site", "")
    df["study_group"] = cov.get("study_group", "")
    df["age"] = cov.get("age", np.nan)
    df["predmeal_flag"] = 0.0

    df["ds"] = np.arange(n)
    df["minute_of_day"] = (df["bin_start"].dt.hour * 60 + df["bin_start"].dt.minute).astype(float)
    tod = df["minute_of_day"] / 1440.0
    df["tod_sin"] = np.sin(2 * np.pi * tod)
    df["tod_cos"] = np.cos(2 * np.pi * tod)

    df["activity_steps"] = df["physical_activity_steps"].fillna(0.0)
    df["calories_value"] = df["physical_activity_calorie_rate"].fillna(0.0)
    df["oxygen_saturation_applicable"] = df["oxygen_saturation_applicable"].fillna(False).astype(float)
    # every remaining participant has real coverage on all four channels
    # (checked above), so ffill/bfill always fully resolves -- no sentinel
    # fallback needed here, unlike build_ssmcgm_input.py's version.
    df["heartrate"] = df["heart_rate"].ffill().bfill()
    df["oxygen_saturation"] = df["oxygen_saturation"].ffill().bfill()
    df["respiration_rate"] = df["respiratory_rate"].ffill().bfill()
    df["stress_level"] = df["stress"].ffill().bfill()
    df["sleep_stage"] = df["sleep_stage"].fillna("unknown")

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
    n_dropped_zero_coverage = {"train": 0, "val": 0, "test": 0}
    for i, pid in enumerate(pids):
        cov = covs.get(str(pid), {})
        split = cov.get("recommended_split", "")
        n_by_split[split] = n_by_split.get(split, 0) + 1
        if split not in frames:
            continue
        series = build_participant_series(pid, cov)
        if series is None:
            n_dropped_zero_coverage[split] += 1
            continue
        frames[split].append(series)
        if (i + 1) % 300 == 0:
            print(f"  processed {i + 1}/{len(pids)}", file=sys.stderr)

    for split in ("train", "val", "test"):
        df = pd.concat(frames[split], ignore_index=True)
        df.to_csv(os.path.join(OUTPUT_DIR, f"ssmcgm_{split}.csv"), index=False)
        print(f"{split}: {df['participant_id'].nunique()} participants, {df['series_id'].nunique()} segments, "
              f"{len(df)} rows, {n_dropped_zero_coverage[split]} participants dropped "
              f"(zero real coverage on >=1 of heartrate/oxygen_saturation/respiration_rate/stress_level "
              f"or no glucose-eligible rows at all)", file=sys.stderr)

    print(f"participants by recommended_split: {n_by_split}", file=sys.stderr)


if __name__ == "__main__":
    run()
