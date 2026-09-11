"""Isolates the effect of covariates from the effect of which windows get
selected, by restricting the glucose_only vs with_covariates comparison to
the exact same underlying (person_id, context_start) windows in both
conditions, then runs paired tests on the per-window loss differential.

Why restriction is needed: with_covariates' eligibility requirement
(coverage on cgm_glucose AND heart_rate AND respiratory_rate, windows.py's
COVARIATE_ELIGIBILITY_CHANNELS) is strictly stronger than glucose_only's
(cgm_glucose alone), checked at IDENTICAL context_start positions per
participant -- extract_windows() walks the same context_bins+horizon_bins-
spaced positions regardless of with_covariates, only the eligibility bar
differs. Confirmed directly against model_input/window_index.csv before
writing anything downstream of this: with_covariates' eligible-window set is
a strict subset of glucose_only's at every horizon (intersection size ==
with_covariates count, exactly, for all 5 horizons). So the raw MAE/RMSE
comparison in chronos2_zeroshot_results/summary.csv mixes two effects: (a)
what covariates do to the forecast, and (b) with_covariates' smaller,
systematically-different sample (only windows with good HR/RR coverage too)
being an easier or harder subset than the full glucose_only population.
This script isolates (a) by scoring glucose_only ONLY on the same windows
with_covariates used (which, by the subset relationship above, means simply
restricting glucose_only's predictions to with_covariates' own window set).

The actual matched-restriction + paired-test logic lives in
chronos2_matched_comparison.py's compare_arms() -- shared with the LOO
ablation comparisons (chronos2_loo_matched_comparison.py), which hit the
exact same glucose_only-is-a-different-population confound. This script is
just that one call, looped over horizons, kept around as its own entry
point since covariate_comparison.csv is the name already referenced in
report.md/CHRONOS2_EVAL.md.

Paired test -- two variants reported, not one:
  - "dm_hln": Diebold-Mariano (1995) statistic with the
    Harvey-Leybourne-Newbold (1997) small-sample correction, t(T-1)
    reference distribution instead of Normal. Fixed Bartlett-truncation at
    horizon_bins-1 lags, per the DM/HLN convention for h-step-ahead
    forecasts -- checked before implementing: statsmodels 0.14.6 (installed
    for this script) has no built-in DM test, and its general HAC tool
    (statsmodels.stats.sandwich_covariance.cov_hac) defaults to the
    data-driven Newey-West automatic bandwidth floor(4(T/100)^(2/9)), not
    this fixed horizon-tied bandwidth, so there was no default to inherit;
    hand-implemented from the published formulas instead.
    Caveat, disclosed rather than silently applied: DM/HLN's own
    justification for the h-1 lag truncation is correlation between
    OVERLAPPING adjacent-origin forecasts in a single rolling time series.
    Our windows are explicitly non-overlapping (windows.py's extract_windows
    walks forward by context_bins+horizon_bins each step) and are drawn
    from many different participants, not one series -- so this correction's
    literal motivation doesn't transfer cleanly here. Reported because it
    was specifically asked for, not because it's judged the best-justified
    choice below.
  - "cluster_by_participant": paired t-test on participant-level MEAN loss
    differentials (one observation per participant, not per window).
    Judged the more defensible correction for THIS design: independent
    participants, correlated within-participant windows (shared physiology),
    which is a cross-sectional repeated-measures structure, not a
    single-series overlapping-forecast one.
Both use per-window mean squared error (over that window's scored, i.e.
non-NaN-ground-truth, horizon bins) as the base loss, differenced as
(glucose_only - with_covariates); positive means covariates reduced loss.
"""
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from chronos2_matched_comparison import compare_arms  # noqa: E402

REPO_ROOT = "/path/to/aireadi-collab"
EVAL_DIR = os.environ.get("CHRONOS2_EVAL_DIR_OVERRIDE", f"{REPO_ROOT}/model_input/zeroshot_eval")
RESULTS_DIR = os.environ.get("CHRONOS2_RESULTS_DIR_OVERRIDE", f"{REPO_ROOT}/chronos2_zeroshot_results/full/imputed")
HORIZON_BINS = {"30min": 6, "1h": 12, "2h": 24, "3h": 36, "4h": 48}


def run_horizon(horizon_name, horizon_bins):
    arm_a = {"label": "glucose_only", "results_dir": RESULTS_DIR, "condition": "glucose_only", "eval_dir": EVAL_DIR}
    arm_b = {"label": "with_covariates", "results_dir": RESULTS_DIR, "condition": "with_covariates", "eval_dir": EVAL_DIR}
    row = compare_arms(horizon_name, horizon_bins, arm_a, arm_b)
    print(f"{horizon_name}: matched n={row['n_matched_windows']}, mean(loss_diff)={row['mean_loss_diff_mse']:.4f} "
          f"(+ve => covariates lower MSE), naive_p={row['naive_paired_p']:.4f}, dm_hln_p={row['dm_hln_p']:.4f}, "
          f"cluster_p={row['cluster_by_participant_p']:.4f} (n_participants={row['n_participants_clustered']})",
          file=sys.stderr)
    return row


def main():
    rows = [run_horizon(h, b) for h, b in HORIZON_BINS.items()]
    out = pd.DataFrame(rows)
    out.to_csv(f"{RESULTS_DIR}/covariate_comparison.csv", index=False)
    print(out.to_string(index=False), file=sys.stderr)


if __name__ == "__main__":
    main()
