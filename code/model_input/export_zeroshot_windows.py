"""Builds zero-shot eval inputs for Chronos-2 and TimesFM 2.5 from
window_index.csv, restricted by default to the official AI-READI test split
(participants.tsv's recommended_split) -- the single held-out population
both models AND SSM-CGM/CGM-LSM are evaluated on for the cross-model
comparison (report.md decision, see model_input/README.md). The split is
overridable via ZEROSHOT_EVAL_SPLIT for the personalization work's val-split
hyperparameter sweep; see that constant's comment.

Every window's covariate columns are blanked (NaN) during the horizon/target
period, not just the context period -- a real forecasting deployment would
not know future heart rate/steps/etc. any more than SSM-CGM should (see the
oracle-covariate fix applied there); this keeps all four models on the same
footing rather than only fixing the issue for one of them.

Output: one long-format CSV per (horizon, with_covariates) combination,
written for both Chronos-2 and TimesFM to consume (Chronos-2 reads it
directly as a pandas.DataFrame per its own predict_df contract; TimesFM's
list-of-np.ndarray format is built from the same rows at model-run time,
since it needs no on-disk format of its own).
"""
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from build_participant_grid import build_grid, load_participants_covariates  # noqa: E402
from windows import (  # noqa: E402
    COVARIATE_APPLICABILITY_CHANNELS, COVARIATE_FEATURE_CHANNELS, CONTEXT_BINS, HORIZONS_BINS,
)

WINDOW_INDEX_PATH = "/path/to/aireadi-collab/model_input/window_index.csv"
# ZEROSHOT_EVAL_OUTPUT_DIR_OVERRIDE: used for the non-LOCF wearables variant
# (paired with build_participant_grid's ANALYSIS_DIR_OVERRIDE) so it lands in
# its own directory rather than overwriting the LOCF'd zeroshot_eval/ used by
# the existing Chronos-2 results. WINDOW_INDEX_PATH is NOT overridden --
# window selection (which (person_id, context_start) pairs are eligible) must
# stay identical to what SSM-CGM/CGM-LSM were compared against; only the
# covariate values populated into each window change.
OUTPUT_DIR = os.environ.get(
    "ZEROSHOT_EVAL_OUTPUT_DIR_OVERRIDE",
    "/path/to/aireadi-collab/model_input/zeroshot_eval",
)
# ZEROSHOT_EVAL_SPLIT: which recommended_split to export. Defaults to "test",
# which is what every existing zeroshot_eval/ file and every Chronos-2/
# TimesFM/CGM-LSM result in this project is built from -- so the default
# reproduces the previous hard-coded behavior exactly.
#
# Added 2026-09-02 for the personalization fine-tuning work
# (code/chronos2/chronos2_personalization.py), which needs a VAL-split copy of these
# same windows to pick fine-tuning hyperparameters on. Sweeping lr/steps/LoRA
# rank on the test split would make the headline personalization number a
# swept one rather than a held-out one -- the mistake the CGM-LSM fine-tuning
# sweeps in HANDOFF.md made and that this avoids. Export val to its own
# directory (ZEROSHOT_EVAL_OUTPUT_DIR_OVERRIDE) and point the sweep at it via
# compare_arms()'s existing per-arm `eval_dir`.
#
# This changes nothing about window SELECTION: window_index.csv already
# contains every split's eligible windows, computed by the same
# build_window_index.py pass under the same coverage rule. This only chooses
# which of those rows get materialized into an export file.
SPLIT = os.environ.get("ZEROSHOT_EVAL_SPLIT", "test")


def build_window_rows(df, window_id, context_start, horizon_bins, with_covariates):
    context_start = pd.Timestamp(context_start)
    start_idx = df.index[df["bin_start"] == context_start]
    if len(start_idx) == 0:
        return None
    start_idx = start_idx[0]
    block = df.iloc[start_idx:start_idx + CONTEXT_BINS + horizon_bins].copy()
    if len(block) != CONTEXT_BINS + horizon_bins:
        return None
    block["window_id"] = window_id
    block["is_context"] = [True] * CONTEXT_BINS + [False] * horizon_bins
    cols = ["window_id", "person_id", "bin_start", "is_context", "cgm_glucose"]
    if with_covariates:
        cols += COVARIATE_FEATURE_CHANNELS + COVARIATE_APPLICABILITY_CHANNELS
        # blank covariates during the horizon: no model gets future wearable
        # values it wouldn't have at real forecast time (see module docstring).
        # COVARIATE_APPLICABILITY_CHANNELS (oxygen_saturation_applicable) is
        # deliberately NOT blanked here -- it's a calendar fact (inside the
        # nightly collection window or not), not a physiological reading, so
        # a real deployment would know it in advance same as time-of-day.
        block.loc[~block["is_context"], COVARIATE_FEATURE_CHANNELS] = float("nan")
    # sleep_stage is always exported (both conditions) as an extra column,
    # not gated behind the with_covariates boolean or its own file-naming
    # permutation: neither Chronos-2's nor TimesFM's original covariate set
    # included it (report.md/HANDOFF.md decision), so it's added here as an
    # opt-in extra a consuming eval script can choose to use, blanked during
    # the horizon for the same no-future-leakage reason as the numeric
    # covariates.
    cols += ["sleep_stage"]
    block.loc[~block["is_context"], "sleep_stage"] = None
    # predmeal_prob (added 2026-08-28): also always exported as an opt-in
    # extra, same treatment as sleep_stage. Blanked during the horizon like
    # the numeric physiological covariates, NOT like oxygen_saturation_
    # applicable: predmeal_prob[t] only depends on real glucose up to t
    # (generate_meal_probability.py), so a horizon-period value would
    # require glucose a real forecast doesn't have yet -- see
    # build_ssmcgm_input.py's module docstring for the same reasoning.
    cols += ["predmeal_prob"]
    block.loc[~block["is_context"], "predmeal_prob"] = float("nan")
    # sleep_regularity_index (added 2026-08-28): also always exported as an
    # opt-in extra. Blanked during the horizon for consistency with every
    # other derived signal here, even though its own definition (build_
    # participant_grid.compute_sleep_regularity_index) never actually uses
    # same-day data -- a horizon-period value would, in principle, be
    # knowable in advance without leakage, but treating it as an exception
    # would mean judging each new covariate's leak-safety case-by-case
    # instead of applying one simple rule project-wide (only genuine
    # calendar facts like oxygen_saturation_applicable/tod_sin/tod_cos get
    # real future values; every other derived/physiological signal is
    # blanked). Consistency of the rule matters more here than the small
    # amount of real information being discarded.
    cols += ["sleep_regularity_index"]
    block.loc[~block["is_context"], "sleep_regularity_index"] = float("nan")

    # Static covariates (added 2026-08-28, completing SSM-CGM's own static
    # covariate list -- arXiv:2510.04386 Sec 2: "age, diabetes status,
    # clinical site, and participant ID" -- plus glucose_scale/glucose_center,
    # its 2nd-most-important static feature by their own VSN ranking,
    # Table 5). age/clinical_site/study_group are real participant-level
    # values already present in build_grid()'s own df (build_participant_
    # grid.py, from participants.tsv), genuinely constant for the whole
    # window and known trivially in advance -- never blanked, same
    # treatment as oxygen_saturation_applicable. participant_id is not
    # re-exported here: it's already the window's own person_id column.
    #
    # glucose_scale/glucose_center are NOT computed the way SSM-CGM's own
    # paper implies (a lifetime per-participant statistic spanning train
    # AND test, via pytorch_forecasting's GroupNormalizer) -- that would
    # leak population statistics computed partly from the test period into
    # the test input itself. Computed here PER WINDOW from only that
    # window's own context glucose (144 real bins), the same causal
    # boundary every other covariate in this project respects.
    context_glucose = block.loc[block["is_context"], "cgm_glucose"]
    block["glucose_center"] = context_glucose.mean()
    block["glucose_scale"] = context_glucose.std()
    cols += ["age", "clinical_site", "study_group", "glucose_center", "glucose_scale"]
    return block[cols]


def run():
    index = pd.read_csv(WINDOW_INDEX_PATH, dtype={"person_id": str})
    test_index = index[index["recommended_split"] == SPLIT].reset_index(drop=True)
    if test_index.empty:
        raise SystemExit(f"no windows for recommended_split={SPLIT!r} in {WINDOW_INDEX_PATH}")
    print(f"exporting split={SPLIT}: {len(test_index)} index rows, "
          f"{test_index['person_id'].nunique()} participants -> {OUTPUT_DIR}", file=sys.stderr)
    covs = load_participants_covariates()
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    grid_cache = {}
    for horizon_name, horizon_bins in HORIZONS_BINS.items():
        for with_covariates in [False, True]:
            subset = test_index[(test_index["horizon"] == horizon_name) &
                                 (test_index["with_covariates"] == with_covariates)]
            rows = []
            for i, (_, w) in enumerate(subset.iterrows()):
                pid = w["person_id"]
                if pid not in grid_cache:
                    grid_cache[pid] = build_grid(pid, covs)
                    if len(grid_cache) > 50:  # cap memory: keep it small, cache hits are per-pid-repeated across horizons/conditions
                        grid_cache.pop(next(iter(grid_cache)))
                df = grid_cache[pid]
                if df is None:
                    continue
                block = build_window_rows(df, f"{pid}_{i}", w["context_start"], horizon_bins, with_covariates)
                if block is not None:
                    rows.append(block)
            if not rows:
                print(f"WARNING: 0 windows for horizon={horizon_name} with_covariates={with_covariates}", file=sys.stderr)
                continue
            out = pd.concat(rows, ignore_index=True)
            fname = f"zeroshot_{horizon_name}_{'with_covariates' if with_covariates else 'glucose_only'}.csv"
            out.to_csv(os.path.join(OUTPUT_DIR, fname), index=False)
            print(f"{fname}: {out['window_id'].nunique()} windows, {len(out)} rows", file=sys.stderr)


if __name__ == "__main__":
    run()
