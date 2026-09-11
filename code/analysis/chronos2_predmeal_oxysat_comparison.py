"""Matched-window comparison for the predmeal_prob x oxygen_saturation 2x2
(slurm/chronos2_predmeal_prob_drop_oxysat.sbatch).

Completes a clean 2x2 out of arms that mostly already existed:

                      | SpO2 kept                  | SpO2 dropped
  --------------------+----------------------------+---------------------------
  no predmeal_prob    | full/imputed               | loo/drop_oxygen_saturation
  with predmeal_prob  | full/with_predmeal_prob    | full/with_predmeal_prob_
                      |                            |   drop_oxysat  (NEW)

Only the fourth cell needed running. The point is to test whether
predmeal_prob's one real benefit (-0.173 mg/dL at 3h on top of baseline,
p=0.001) is larger, equal, or smaller once oxygen_saturation -- the only
channel that clearly degrades the forecast -- is out of the way.

Goes through chronos2_matched_comparison.compare_arms() like every other
arm-vs-arm comparison in this project, per that module's standing rule that
raw summary.csv diffs are never used. All four cells share the
with_covariates-eligible window set (ABLATE_CHANNELS and the INCLUDE_* flags
never touch window selection), so restriction is a no-op here but goes
through the same call path rather than being special-cased.

Pairs reported, each answering one question:
  - drop_oxysat            -> drop_oxysat + predmeal : the marginal value of
    the meal signal on the better channel set. THE test this arm exists for;
    compare its magnitude to the -0.173 @3h that the same channel bought on
    top of baseline.
  - with_predmeal_prob     -> drop_oxysat + predmeal : the marginal value of
    removing SpO2 when the meal signal is already present. The other margin
    of the same 2x2.
  - full/imputed baseline  -> drop_oxysat + predmeal : the total effect of
    both changes together, i.e. is this the best Chronos-2 config found.
  - full/imputed glucose_only -> drop_oxysat + predmeal : kept for parity
    with the two-reference pattern every other comparison tree reports, so
    all of them can be read side by side.

An interaction term is NOT computed here. With only one replicate per cell
and no per-cell variance estimate, a difference-in-differences on these four
numbers would have no honest standard error attached; the three pairwise
tests above, read together, are what the design actually supports.
"""
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from chronos2_matched_comparison import compare_arms  # noqa: E402

REPO_ROOT = "/path/to/aireadi-collab"
BASELINE_DIR = f"{REPO_ROOT}/chronos2_zeroshot_results/full/imputed"
PREDMEAL_DIR = f"{REPO_ROOT}/chronos2_zeroshot_results/full/with_predmeal_prob"
DROP_OXYSAT_DIR = f"{REPO_ROOT}/chronos2_zeroshot_results/loo/drop_oxygen_saturation"
NEW_ARM_DIR = f"{REPO_ROOT}/chronos2_zeroshot_results/full/with_predmeal_prob_drop_oxysat"
# The three non-calendar add-ons together, with and without oxygen_saturation.
# DROP_CALENDAR_DIR is the SpO2-kept counterpart and required no new run: that
# arm is all five INCLUDE_* flags with ABLATE_CHANNELS=tod_sin,minute_of_day,
# which leaves exactly the 6 physiological channels +
# oxygen_saturation_applicable + sleep_stage + predmeal_prob +
# sleep_regularity_index.
DROP_CALENDAR_DIR = f"{REPO_ROOT}/chronos2_zeroshot_results/loo_ssmcgm_overlap/drop_calendar"
SLEEP_MEAL_DROP_OXYSAT_DIR = f"{REPO_ROOT}/chronos2_zeroshot_results/full/with_sleep_meal_drop_oxysat"
SLEEP_STAGE_SPLIT_DIR = f"{REPO_ROOT}/chronos2_zeroshot_results/full/with_sleep_stage_predmeal_drop_oxysat"
SLEEP_REG_SPLIT_DIR = f"{REPO_ROOT}/chronos2_zeroshot_results/full/with_sleep_reg_predmeal_drop_oxysat"
HORIZON_BINS = {"30min": 6, "1h": 12, "2h": 24, "3h": 36, "4h": 48}

# (arm under test, its label, list of (reference label, dir, condition)).
# References ordered most- to least-central to the question each arm answers.
ARMS = [
    # predmeal_prob alone on the drop_oxygen_saturation channel set.
    (NEW_ARM_DIR, "predmeal_drop_oxysat", [
        ("drop_oxygen_saturation", DROP_OXYSAT_DIR, "with_covariates"),
        ("with_predmeal_prob", PREDMEAL_DIR, "with_covariates"),
        ("with_covariates", BASELINE_DIR, "with_covariates"),
        ("glucose_only", BASELINE_DIR, "glucose_only"),
    ]),
    # All three non-calendar add-ons on the drop_oxygen_saturation channel set
    # -- the missing cell of the grid, added 2026-09-01. Its three references
    # are each one controlled step away:
    #   drop_calendar          -> + remove oxygen_saturation (SpO2's effect
    #                             with all three add-ons present)
    #   predmeal_drop_oxysat   -> + add the two sleep channels (do they stay
    #                             inert once SpO2 is gone?)
    #   drop_oxygen_saturation -> + add all three add-ons at once
    (SLEEP_MEAL_DROP_OXYSAT_DIR, "sleep_meal_drop_oxysat", [
        ("drop_calendar", DROP_CALENDAR_DIR, "with_covariates"),
        ("predmeal_drop_oxysat", NEW_ARM_DIR, "with_covariates"),
        ("drop_oxygen_saturation", DROP_OXYSAT_DIR, "with_covariates"),
        ("with_covariates", BASELINE_DIR, "with_covariates"),
        ("glucose_only", BASELINE_DIR, "glucose_only"),
    ]),
    # The individual sleep-channel split (added 2026-09-01): sleep_meal_drop_
    # oxysat added both sleep channels at once and so cannot attribute the
    # pair's harm. Each arm here adds exactly ONE of them to the best config.
    # References:
    #   predmeal_drop_oxysat   -> this channel's own marginal effect (THE test)
    #   sleep_meal_drop_oxysat -> does removing the OTHER channel recover?
    # Prediction on record (see the sbatch): sleep_regularity_index should be
    # largely inert, because it is one value per calendar day broadcast to
    # every bin, hence exactly constant for any 12h window inside one UTC day
    # (~half of them) and therefore zeroed by instance norm.
    (SLEEP_STAGE_SPLIT_DIR, "sleep_stage_predmeal_drop_oxysat", [
        ("predmeal_drop_oxysat", NEW_ARM_DIR, "with_covariates"),
        ("sleep_meal_drop_oxysat", SLEEP_MEAL_DROP_OXYSAT_DIR, "with_covariates"),
        ("with_covariates", BASELINE_DIR, "with_covariates"),
    ]),
    (SLEEP_REG_SPLIT_DIR, "sleep_reg_predmeal_drop_oxysat", [
        ("predmeal_drop_oxysat", NEW_ARM_DIR, "with_covariates"),
        ("sleep_meal_drop_oxysat", SLEEP_MEAL_DROP_OXYSAT_DIR, "with_covariates"),
        ("with_covariates", BASELINE_DIR, "with_covariates"),
    ]),
]


def run():
    rows = []
    for arm_dir, arm_label, references in ARMS:
        if not os.path.isdir(arm_dir):
            print(f"SKIP {arm_label}: no dir at {arm_dir}", file=sys.stderr)
            continue
        arm_b = {"label": arm_label, "results_dir": arm_dir, "condition": "with_covariates"}
        for horizon_name, horizon_bins in HORIZON_BINS.items():
            for ref_label, ref_dir, ref_condition in references:
                if not os.path.isdir(ref_dir):
                    print(f"SKIP reference {ref_label}: no dir at {ref_dir}", file=sys.stderr)
                    continue
                arm_a = {"label": ref_label, "results_dir": ref_dir, "condition": ref_condition}
                row = compare_arms(horizon_name, horizon_bins, arm_a, arm_b)
                row["arm"] = arm_label
                row["reference"] = ref_label
                rows.append(row)

    if not rows:
        print("no arms found -- nothing written", file=sys.stderr)
        return

    out = pd.DataFrame(rows)
    out_path = f"{REPO_ROOT}/chronos2_zeroshot_results/full/predmeal_oxysat_matched_comparison.csv"
    out.to_csv(out_path, index=False)
    print(f"wrote {len(out)} rows to {out_path}", file=sys.stderr)


if __name__ == "__main__":
    run()
