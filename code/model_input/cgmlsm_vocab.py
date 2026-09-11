"""CGM-LSM's two token vocabularies, copied verbatim from the authors' own
construction in code/external/cgmlsm/1-CGMLSM/run_cgmlsm_pretrain.py (also duplicated
once already in code/cgmlsm_toy_pretrain_smoketest.py) -- kept in one place
here so build_cgmlsm_input.py and any later training/eval script share the
exact same token<->index mapping instead of re-deriving it.

CGMValue: mg/dL reading -> token. "PAD"/"UNKNOWN"/"MASK" plus "Other_0".."
Other_6" (their WellDoc dataset's own error-code sentinels, unused here but
kept so the vocab shape matches theirs) plus one token per integer mg/dL bin
from 10 to 400.

HM5MinStep: time-of-day -> token, one per 5-minute step in a day ("00:00",
"00:05", ..., "23:55"), 288 tokens total.
"""
import numpy as np
import pandas as pd

CGM_IDX2TKN = ["PAD", "UNKNOWN", "MASK"] + [f"Other_{i}" for i in range(0, 7)] + [str(i) for i in range(10, 401)]
CGM_TKN2IDX = {tkn: idx for idx, tkn in enumerate(CGM_IDX2TKN)}
CGM_PAD_IDX = CGM_TKN2IDX["PAD"]

_interval = pd.Timedelta(minutes=5)
TOD_IDX2TKN = [pd.Timestamp("2022-01-01") + _interval * i for i in range(24 * 12)]
TOD_IDX2TKN = [f"{t.hour:02d}:{t.minute:02d}" for t in TOD_IDX2TKN]
TOD_TKN2IDX = {tkn: idx for idx, tkn in enumerate(TOD_IDX2TKN)}


def tokenize_glucose(values):
    """mg/dL -> CGMValue token string. NaN (never imputed anywhere upstream,
    see build_participant_grid.py's own docstring) becomes the vocab's own
    "UNKNOWN" symbol rather than a fabricated number; in-range values are
    clamped into [10, 400] and rounded to the nearest integer bin, matching
    cgmlsm_toy_pretrain_smoketest.py's tokenize_mgdl."""
    out = []
    for v in values:
        if pd.isna(v):
            out.append("UNKNOWN")
        else:
            out.append(str(int(np.clip(round(float(v)), 10, 400))))
    return out


def tokenize_tod(bin_start_series):
    """bin_start (already snapped to 5-minute bins upstream) -> HM5MinStep
    token string. Uses the timestamp's stored hour/minute as-is, the same
    convention build_ssmcgm_input.py's minute_of_day/tod_sin/tod_cos already
    use on this project's UTC-stored bin_start -- not converted to any
    participant-local timezone, for consistency with that precedent."""
    return bin_start_series.dt.hour.map("{:02d}".format) + ":" + bin_start_series.dt.minute.map("{:02d}".format)
