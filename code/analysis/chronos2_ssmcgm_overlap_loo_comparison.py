"""Matched-window comparison for the within-overlap LOO sweep
(slurm/chronos2_loo_ablation_ssmcgm_overlap.sbatch).

Separate from zeroshot_matched_comparison_all.py rather than folded into it:
that script's LOO section is hardcoded to compare loo/drop_<channel> against
full/imputed, which is the right reference for the original 6-channel
physiological sweep but the wrong one here. These arms are leave-one-out from
the SSM-CGM-matched arm (full/with_ssmcgm_overlap), so the primary reference
has to be that arm, not the baseline.

Goes through chronos2_matched_comparison.compare_arms() like every other
arm-vs-arm comparison in this project -- same common-window restriction, same
naive paired t / Diebold-Mariano-HLN / cluster-by-participant tests -- rather
than diffing summary.csv rows, per that module's standing rule. Restriction
is a no-op in practice here (every arm shares the with_covariates-eligible
window set, since ABLATE_CHANNELS never touches window selection) but goes
through the same call path instead of being special-cased.

Three references per arm, each answering a different question:
  - vs with_ssmcgm_overlap: the marginal effect of that channel inside the
    matched-feature set. This is the LOO question proper.
  - vs full/imputed with_covariates: does dropping this channel get back to
    the un-matched baseline, or only partway? The decisive one for
    drop_calendar -- if it lands at baseline, the calendar block accounts for
    the whole of the overlap arm's degradation.
  - vs full/imputed glucose_only: kept for parity with the two-reference
    pattern the existing loo/ and addon comparisons already report, so all
    three trees can be read side by side.
"""
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from chronos2_matched_comparison import compare_arms  # noqa: E402

REPO_ROOT = "/path/to/aireadi-collab"
BASELINE_RESULTS_DIR = f"{REPO_ROOT}/chronos2_zeroshot_results/full/imputed"
OVERLAP_RESULTS_DIR = f"{REPO_ROOT}/chronos2_zeroshot_results/full/with_ssmcgm_overlap"
LOO_DIR = f"{REPO_ROOT}/chronos2_zeroshot_results/loo_ssmcgm_overlap"
HORIZON_BINS = {"30min": 6, "1h": 12, "2h": 24, "3h": 36, "4h": 48}

# Calendar-block arms first, matching the sbatch's own ordering and for the
# reason given there: drop_calendar is the decisive test and the rest are
# confirmatory. drop_tod removes tod_sin AND tod_cos (see the sbatch's note),
# so these three fully partition the calendar block's four states.
ARMS = [
    "drop_calendar",
    "drop_tod",
    "drop_minute_of_day",
    "drop_sleep_stage",
    "drop_predmeal_prob",
    "drop_sleep_regularity_index",
]

REFERENCES = [
    ("ssmcgm_overlap", OVERLAP_RESULTS_DIR, "with_covariates"),
    ("with_covariates", BASELINE_RESULTS_DIR, "with_covariates"),
    ("glucose_only", BASELINE_RESULTS_DIR, "glucose_only"),
]


def run():
    rows = []
    for arm in ARMS:
        arm_dir = f"{LOO_DIR}/{arm}"
        if not os.path.isdir(arm_dir):
            print(f"SKIP {arm}: no results dir at {arm_dir}", file=sys.stderr)
            continue
        for horizon_name, horizon_bins in HORIZON_BINS.items():
            arm_b = {"label": arm, "results_dir": arm_dir, "condition": "with_covariates"}
            for ref_label, ref_dir, ref_condition in REFERENCES:
                arm_a = {"label": ref_label, "results_dir": ref_dir, "condition": ref_condition}
                row = compare_arms(horizon_name, horizon_bins, arm_a, arm_b)
                row["ablated_arm"] = arm
                row["reference"] = ref_label
                rows.append(row)

    if not rows:
        print("no arms found -- nothing written", file=sys.stderr)
        return

    out = pd.DataFrame(rows)
    out_path = f"{LOO_DIR}/overlap_loo_matched_comparison.csv"
    out.to_csv(out_path, index=False)
    print(f"wrote {len(out)} rows to {out_path}", file=sys.stderr)


if __name__ == "__main__":
    run()
