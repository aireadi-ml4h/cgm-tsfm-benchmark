"""Matched-window comparison for the LOO ablation arms (chronos2_loo_ablation.sbatch)
against both reference points, using the same restriction logic as
chronos2_covariate_comparison.py (chronos2_matched_comparison.compare_arms) --
raw summary.csv MAE/RMSE diffs are not used here, per the standing rule that
any two Chronos-2 arms with different eligible-window sets must be restricted
to their common windows before comparing (see chronos2_matched_comparison.py's
docstring).

Two reference comparisons per ablated channel, both matched:
  - vs glucose_only: is dropping this channel's covariate better than not
    using covariates at all.
  - vs the full with_covariates baseline: the direct marginal-channel
    effect (both arms already share the same with_covariates-eligible
    window set -- restriction here is a no-op in practice, but goes through
    the same call path rather than being special-cased as "already matched,
    skip the check").
"""
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from chronos2_matched_comparison import compare_arms  # noqa: E402

REPO_ROOT = "/path/to/aireadi-collab"
BASELINE_RESULTS_DIR = f"{REPO_ROOT}/chronos2_zeroshot_results/full/imputed"
LOO_RESULTS_DIR = f"{REPO_ROOT}/chronos2_zeroshot_results/loo"
HORIZON_BINS = {"30min": 6, "1h": 12, "2h": 24, "3h": 36, "4h": 48}
LOO_CHANNELS = [
    "heart_rate", "stress", "respiratory_rate", "oxygen_saturation",
    "physical_activity_steps", "physical_activity_calorie_rate",
]


def run():
    rows = []
    for channel in LOO_CHANNELS:
        loo_results_dir = f"{LOO_RESULTS_DIR}/drop_{channel}"
        for horizon_name, horizon_bins in HORIZON_BINS.items():
            arm_loo = {"label": f"drop_{channel}", "results_dir": loo_results_dir, "condition": "with_covariates"}

            arm_glucose_only = {"label": "glucose_only", "results_dir": BASELINE_RESULTS_DIR, "condition": "glucose_only"}
            row_vs_go = compare_arms(horizon_name, horizon_bins, arm_glucose_only, arm_loo)
            row_vs_go["ablated_channel"] = channel
            row_vs_go["reference"] = "glucose_only"
            rows.append(row_vs_go)

            arm_full_covariates = {"label": "with_covariates", "results_dir": BASELINE_RESULTS_DIR, "condition": "with_covariates"}
            row_vs_full = compare_arms(horizon_name, horizon_bins, arm_full_covariates, arm_loo)
            row_vs_full["ablated_channel"] = channel
            row_vs_full["reference"] = "with_covariates"
            rows.append(row_vs_full)

    out = pd.DataFrame(rows)
    out_path = f"{LOO_RESULTS_DIR}/loo_ablation_matched_comparison.csv"
    out.to_csv(out_path, index=False)
    print(f"wrote {len(out)} rows to {out_path}", file=sys.stderr)
    print(out.to_string(index=False), file=sys.stderr)


if __name__ == "__main__":
    run()
