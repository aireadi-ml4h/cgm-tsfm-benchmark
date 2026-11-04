"""Runs the SAME matched-window statistical comparison (chronos2_matched_
comparison.compare_arms -- naive paired t, Diebold-Mariano/HLN, cluster-by-
participant, all on restricted common windows) against BOTH Chronos-2's and
TimesFM's result directories, for every ablation/addition arm either model
has, not just the original 6-channel sweep chronos2_loo_matched_comparison.py
covered.

Why this exists, not two separate scripts: everything reported so far in
this conversation for TimesFM's LOO sweep, and for BOTH models' sleep_stage/
predmeal_prob/oxygen-2x2 arms, was a raw summary.csv MAE delta -- no
significance test, and (for TimesFM specifically) not even matched to a
common window set the way chronos2_matched_comparison.py's own docstring
says every arm-vs-arm comparison must be. Since compare_arms() takes
results_dir/eval_dir as plain parameters (nothing Chronos-2-specific baked
in -- both models write the identical predictions_<horizon>_<condition>.csv
schema, window_id/bin_start/predictions/actual/abs_error, and both read
window keys from the same shared model_input/zeroshot_eval/*.csv files),
reusing it directly for TimesFM guarantees the two models are held to
IDENTICAL statistical machinery -- not a second implementation that could
subtly diverge from the first. This is the single script covering both
models; MODEL selects which one via a CLI arg / env var.

Does NOT replace chronos2_loo_matched_comparison.py (a collaborator's original,
6-channel-only, Chronos-2-only script) -- that output stays as-is. This
writes its own, more complete output per model:
  <model>_zeroshot_results/loo/full_matched_comparison.csv   (8 LOO channels
    x 5 horizons x 2 references = 80 rows)
  <model>_zeroshot_results/full/addon_matched_comparison.csv (sleep_stage,
    predmeal_prob x 5 horizons x 2 references = 20 rows)
"""
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from chronos2_matched_comparison import compare_arms  # noqa: E402

REPO_ROOT = os.environ["AIREADI_ROOT"]
HORIZON_BINS = {"30min": 6, "1h": 12, "2h": 24, "3h": 36, "4h": 48}

# The full LOO channel list as actually run for each model (see
# slurm/*_loo_ablation*.sbatch) -- the original 6 physiological channels,
# plus the two that complete the oxygen_saturation 2x2 (flag vs reading).
LOO_CHANNELS = [
    "heart_rate", "stress", "respiratory_rate", "oxygen_saturation",
    "physical_activity_steps", "physical_activity_calorie_rate",
    "oxygen_saturation_applicable", "oxygen_saturation_reading_only",
]
# opt-in extra channels, each its own arm under full/with_<name>, compared
# against the with_covariates baseline (does adding this on top of the full
# covariate set help) and against glucose_only (for completeness/parity with
# the LOO arms' two-reference pattern).
# "ssmcgm_overlap" (added 2026-09-01) is the combined arm, not a single
# channel: all five working add-on flags on at once (slurm/chronos2_zeroshot_
# eval_ssmcgm_overlap.sbatch). It goes through the identical two-reference
# machinery as the singleton arms, which is the point -- the before/after
# question ("does matching SSM-CGM's feature set help or hurt Chronos-2")
# gets the same matched-window restriction and the same significance tests
# as every marginal-channel question already reported, rather than a
# separate raw-MAE diff that would not be comparable to them.
ADDON_ARMS = ["sleep_stage", "predmeal_prob", "tod", "sleep_regularity", "static", "ssmcgm_overlap"]


def run_for_model(model):
    baseline_dir = f"{REPO_ROOT}/{model}_zeroshot_results/full/imputed"
    loo_dir = f"{REPO_ROOT}/{model}_zeroshot_results/loo"
    full_dir = f"{REPO_ROOT}/{model}_zeroshot_results/full"

    loo_rows = []
    for channel in LOO_CHANNELS:
        channel_dir = f"{loo_dir}/drop_{channel}"
        if not os.path.isdir(channel_dir):
            print(f"SKIP {model} drop_{channel}: no results dir found", file=sys.stderr)
            continue
        for horizon_name, horizon_bins in HORIZON_BINS.items():
            arm_loo = {"label": f"drop_{channel}", "results_dir": channel_dir, "condition": "with_covariates"}

            arm_glucose_only = {"label": "glucose_only", "results_dir": baseline_dir, "condition": "glucose_only"}
            row = compare_arms(horizon_name, horizon_bins, arm_glucose_only, arm_loo)
            row["ablated_channel"] = channel
            row["reference"] = "glucose_only"
            loo_rows.append(row)

            arm_full = {"label": "with_covariates", "results_dir": baseline_dir, "condition": "with_covariates"}
            row = compare_arms(horizon_name, horizon_bins, arm_full, arm_loo)
            row["ablated_channel"] = channel
            row["reference"] = "with_covariates"
            loo_rows.append(row)

    addon_rows = []
    for addon in ADDON_ARMS:
        addon_dir = f"{full_dir}/with_{addon}"
        if not os.path.isdir(addon_dir):
            print(f"SKIP {model} with_{addon}: no results dir found", file=sys.stderr)
            continue
        for horizon_name, horizon_bins in HORIZON_BINS.items():
            arm_addon = {"label": f"with_{addon}", "results_dir": addon_dir, "condition": "with_covariates"}

            arm_glucose_only = {"label": "glucose_only", "results_dir": baseline_dir, "condition": "glucose_only"}
            row = compare_arms(horizon_name, horizon_bins, arm_glucose_only, arm_addon)
            row["added_channel"] = addon
            row["reference"] = "glucose_only"
            addon_rows.append(row)

            arm_full = {"label": "with_covariates", "results_dir": baseline_dir, "condition": "with_covariates"}
            row = compare_arms(horizon_name, horizon_bins, arm_full, arm_addon)
            row["added_channel"] = addon
            row["reference"] = "with_covariates"
            addon_rows.append(row)

    if loo_rows:
        out_path = f"{loo_dir}/full_matched_comparison.csv"
        pd.DataFrame(loo_rows).to_csv(out_path, index=False)
        print(f"{model}: wrote {len(loo_rows)} rows to {out_path}", file=sys.stderr)
    if addon_rows:
        out_path = f"{full_dir}/addon_matched_comparison.csv"
        pd.DataFrame(addon_rows).to_csv(out_path, index=False)
        print(f"{model}: wrote {len(addon_rows)} rows to {out_path}", file=sys.stderr)


def main():
    models = os.environ.get("MATCHED_COMPARISON_MODELS", "chronos2,timesfm").split(",")
    for model in models:
        model = model.strip()
        if model:
            run_for_model(model)


if __name__ == "__main__":
    main()
