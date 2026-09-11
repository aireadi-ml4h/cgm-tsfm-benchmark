"""Non-overlapping (context, horizon) window extraction over a participant's
grid (build_participant_grid.build_grid), walking forward chronologically --
same non-overlapping convention as this project's own coverage_clipping_*.py
scripts and GlucoFM-Bench's leakage-avoidance rationale (arXiv:2606.06881).

Two coverage-eligible window sets are computed per (participant, horizon):
glucose-only (coverage checked on cgm_glucose alone) and glucose+covariates
(checked on cgm_glucose plus every wearable channel a model consumes) --
supporting this project's own research question of whether wearable
covariates help, per Project_Brainstorm.pdf, without conflating the two
conditions' eligible-window sets.
"""
import pandas as pd

from coverage import is_window_eligible

CONTEXT_BINS = 144  # 12h, GlucoFM-Bench's default few-shot/full-shot context
# 3h/4h added 2026-08-27 for a Chronos-2-only extended-horizon check (not
# yet built/run for SSM-CGM/TimesFM/CGM-LSM) -- see code/CHRONOS2_EVAL.md.
# 30min/1h/2h remain the horizons cited in report.md Sec 15.2's literature
# convention; 3h/4h are exploratory, not claimed to match any cited paper.
HORIZONS_BINS = {"30min": 6, "1h": 12, "2h": 24, "3h": 36, "4h": 48}
# Eligibility gate for the "with covariates" condition: heart_rate +
# respiratory_rate only, NOT GlyLLM's full 3-channel set (which also
# requires stress). Measured on a 50-participant sample at 1h horizon:
# requiring all 6 non-sleep wearable channels jointly yielded only 28
# eligible windows/16 participants; GlyLLM's exact 3-channel set (adding
# stress) yielded 155/37; dropping stress specifically yielded 299/47.
# stress is not just another missingness-heavy channel (28.71% sentinel-
# excluded population-wide, report.md Sec 11.5, the highest of any stream)
# -- it is a proprietary, HRV-derived algorithmic composite index (Jun et
# al. 2026, report.md Sec 7.1 caveat), not a direct physiological
# measurement like heart_rate or respiratory_rate, so gating eligibility on
# its presence conflates "this window has real physiological signal" with
# "Garmin's proprietary scoring algorithm happened to run." stress,
# oxygen_saturation, steps, and calorie_rate are still included as extra
# feature columns in the exported window whenever present -- just not
# required non-missing for the window to qualify.
COVARIATE_ELIGIBILITY_CHANNELS = ["heart_rate", "respiratory_rate"]
COVARIATE_FEATURE_CHANNELS = [
    "heart_rate", "stress", "respiratory_rate", "oxygen_saturation",
    "physical_activity_steps", "physical_activity_calorie_rate",
]
# oxygen_saturation_applicable disambiguates "not applicable, outside the
# device's own overnight collection window" from "missing/non-wear during
# the window" (report.md Sec 10.7) -- without it, a NaN oxygen_saturation
# during the day is indistinguishable from a genuine dropout at night once
# exported. Kept as its own list, not folded into COVARIATE_FEATURE_CHANNELS,
# because unlike the physiological readings above, this is a calendar fact
# (is this bin inside the nightly window?) rather than something a real
# forecast deployment wouldn't know in advance -- it is NOT blanked during
# the horizon in build_window_rows(), only the true feature channels are.
COVARIATE_APPLICABILITY_CHANNELS = ["oxygen_saturation_applicable"]
# sleep_asleep excluded even from the feature/eligibility distinction above:
# it is only ever non-NaN during a recorded sleep session's own span
# (report.md Sec 13), so NaN outside of sleep is expected/correct, not a
# coverage failure or a meaningful "missing feature" signal.


def extract_windows(df, horizon_bins, context_bins=CONTEXT_BINS, with_covariates=False):
    """Returns a list of dicts: {context_df, horizon_df, context_start, target_start}
    for every non-overlapping (context_bins + horizon_bins) block that passes
    the coverage rule on both its context and its horizon portion."""
    eligibility_channels = ["cgm_glucose"] + (COVARIATE_ELIGIBILITY_CHANNELS if with_covariates else [])
    block_len = context_bins + horizon_bins
    n = len(df)
    windows = []
    for start in range(0, n - block_len + 1, block_len):
        block = df.iloc[start:start + block_len]
        context = block.iloc[:context_bins]
        horizon = block.iloc[context_bins:]
        context_ok, context_stats = is_window_eligible(context, eligibility_channels)
        horizon_ok, horizon_stats = is_window_eligible(horizon, ["cgm_glucose"])  # ground truth must be real
        if context_ok and horizon_ok:
            windows.append({
                "context_df": context.reset_index(drop=True),
                "horizon_df": horizon.reset_index(drop=True),
                "context_start": context["bin_start"].iloc[0],
                "target_start": horizon["bin_start"].iloc[0],
            })
    return windows
