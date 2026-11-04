"""TimesFM's own glucose_only vs with_covariates baseline comparison,
mirroring code/analysis/chronos2_covariate_comparison.py exactly (same compare_arms()
call, same matched-window restriction, same three tests) -- the equivalent
comparison never existed for TimesFM before this, only for Chronos-2.
"""
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from chronos2_matched_comparison import compare_arms  # noqa: E402

REPO_ROOT = os.environ["AIREADI_ROOT"]
EVAL_DIR = os.environ.get("TIMESFM_EVAL_DIR_OVERRIDE", f"{REPO_ROOT}/model_input/zeroshot_eval")
RESULTS_DIR = os.environ.get("TIMESFM_RESULTS_DIR_OVERRIDE", f"{REPO_ROOT}/timesfm_zeroshot_results/full/imputed")
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
