"""GlyLLM-style window coverage rule (see build_participant_grid.py docstring
for why this threshold was chosen over the generic Cichosz 70% figure):
a window is eligible only if, for every channel it uses, the longest
continuous run of missing bins is <=MAX_GAP_BINS and the overall missing
fraction is <=MAX_MISSING_FRACTION.
"""
import numpy as np
import pandas as pd

from build_participant_grid import channel_is_real_mask

BIN_MINUTES = 5
MAX_GAP_MINUTES = 60
MAX_GAP_BINS = MAX_GAP_MINUTES // BIN_MINUTES  # 12
MAX_MISSING_FRACTION = 0.05


def longest_false_run(is_real):
    """Longest run of consecutive False (missing) values in a boolean array."""
    if len(is_real) == 0:
        return 0
    is_missing = ~is_real.to_numpy()
    max_run = 0
    run = 0
    for v in is_missing:
        run = run + 1 if v else 0
        max_run = max(max_run, run)
    return max_run


def window_channel_stats(df_window, channel):
    """oxygen_saturation is only structurally applicable during its own
    overnight window (report.md Sec 10.7) -- checked only over its own
    applicable bins, vacuously satisfied if the window has none, exactly
    matching the population-level precedent already used in
    coverage_clipping_ssmcgm.py/coverage_clipping_timesfm_cgmlsm.py. Applying
    the raw 24h denominator here reproduced the exact same structural bug
    those scripts already fixed once -- caught via smoke-testing before this
    was run population-wide (zero eligible covariate windows on 3/3 test
    participants was the tell)."""
    if channel == "oxygen_saturation":
        applicable = df_window["oxygen_saturation_applicable"] == True  # noqa: E712 (NaN-safe, see build_participant_grid outer-join note)
        if not applicable.any():
            return {"channel": channel, "missing_fraction": 0.0, "longest_gap_minutes": 0}
        df_window = df_window.loc[applicable]
    is_real = channel_is_real_mask(df_window, channel)
    missing_fraction = 1.0 - is_real.mean() if len(is_real) else 1.0
    longest_gap_bins = longest_false_run(is_real)
    return {
        "channel": channel,
        "missing_fraction": float(missing_fraction),
        "longest_gap_minutes": int(longest_gap_bins * BIN_MINUTES),
    }


def is_window_eligible(df_window, channels):
    """Returns (eligible: bool, per_channel_stats: list[dict])."""
    stats = [window_channel_stats(df_window, c) for c in channels]
    eligible = all(
        s["missing_fraction"] <= MAX_MISSING_FRACTION and s["longest_gap_minutes"] <= MAX_GAP_MINUTES
        for s in stats
    )
    return eligible, stats
