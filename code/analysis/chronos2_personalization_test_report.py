"""Test-split reporting for Chronos-2 personalization, phases 1 and 2.

The counterpart to chronos2_personalization_val_rank.py, which ranks val
configs. This one does no selection at all -- the config is already frozen by
the time this runs -- it answers the pre-specified questions and refuses to
let a null be misread.

WHAT IT ANSWERS, AND WHY THESE PAIRS
====================================
Each row is one (question, horizon). Nothing here is exploratory; every pair
is named in CHRONOS2_PERSONALIZATION.md before any of it ran.

PHASE 1, on the shared eval set E:
  p{1,3,7}d vs zero-shot glucose_only   does personalization help at all
                                        (the PRIMARY comparison: same
                                        channels, same windows, only .fit()
                                        differs)
  p7d vs p1d, p7d vs p3d                dose-response -- is more history
                                        better? The whole reason all arms are
                                        pinned to one window set.
  p7d vs p7d_shuffled                   THE NEGATIVE CONTROL. Without it a
                                        gain is equally well explained by the
                                        model adapting to AI-READI/Dexcom-G6/
                                        5-minute data in general as by
                                        adapting to the PERSON. This row is
                                        not optional and should be read before
                                        any of the rows above.
  p7d vs best zero-shot config          is personalization the new
                                        recommendation, not just an
                                        improvement on its own baseline

PHASE 2, on E_cov:
  each arm vs its own zero-shot twin    does personalization + these channels
                                        beat zero-shot + the same channels
  arm2 - arm1                           isolates predmeal_prob
  arm3 - arm2                           isolates oxygen_saturation
  arm4 - arm2                           isolates sleep_stage
  arm1 vs phase-1 p7d                   the actual phase-2 question: once you
                                        have personalized, do wearables add
                                        anything on top?

WHAT IS FOR PUBLICATION AND WHAT IS NOT (set 2026-09-04 by the user)
====================================================================
Two separate outputs, printed in this order:

  1. MAE / RMSE BY HORIZON, one row per arm  -> test_arm_scores.csv
     THE PAPER OUTPUT. The format TSFM papers (Chronos, TimesFM, Moirai)
     actually use. Every arm in a table is scored on the identical window set.
     No p-values anywhere in it.

  2. Pairwise effect + confidence interval tables  -> test_report.csv
     INTERNAL WORKING NOTES. The p-values are computed and retained for the
     author's own reference, but they label nothing.

NO CATEGORICAL VERDICT COLUMN. Two attempts at one were both wrong, and the
full record is in interval()'s docstring because it is the kind of mistake that
comes back. Summary: labelling rows "helped"/"hurt" on cluster p < 0.05 made a
significance test decide the conclusions; replacing that with "|delta| vs MDE"
was the SAME THING ten times stricter, because MDE = 2.802 * SE makes
"|delta| > MDE" identical to p < 0.005. That gate suppressed 17 of the 36 rows
with p < 0.05, including this project's headline result (p7d vs zero-shot at
1h and 2h).

MDE IS A DESIGN STATISTIC. It answers "how big must an effect be for me to
detect it reliably?", asked BEFORE spending the GPU time -- which is how it was
first used here, correctly, for the phase-2 go/no-go. Comparing an observed
estimate to it afterwards is circular. It stays in the CSV and in the sbatch
banners for the prospective use only.

WHAT EACH ROW CARRIES INSTEAD: the estimate, its 95% CI, whether that CI
excludes zero, and `bounds_effects_above` -- the tightest bound the data
establishes (max |90% CI bound|, the TOST construction). A NULL IS REPORTED AS
A BOUND, NOT AS A FAILURE: "rules out effects larger than 0.35 mg/dL", never
"covariates do not help". The intervals on this data are tight; they simply
include zero, and the MDE framing was hiding that. Equivalence tests and win
rates: code/chronos2_personalization_equivalence.py.

MDEs are computed the same way as chronos2_personalization_power_calibration.py
(that module's functions are imported, not re-implemented) and carry the same
caveat: they are LOWER bounds, since fine-tuning adds per-participant variance
a zero-shot pair does not have.

HORIZON BLOCKS
==============
Printed as two separate tables, per the standing rule:
  30min/1h/2h   CROSS-MODEL COMPARABLE -- the only horizons CGM-LSM and
                SSM-CGM report, so the only range where model-vs-model is
                answerable.
  3h/4h         no baseline model reaches these, so they cannot enter the
                cross-model table -- but they are a genuine finding about
                foundation-model behaviour at longer horizons and are reported
                on their own merits. NOT de-emphasised, NOT dropped.

DIRECTORY LAYOUT THIS EXPECTS
=============================
Fitting is horizon-matched, so one "arm" is TWO fits and a per-horizon mapping,
not one directory:

    chronos2_personalization_results/test/<arm>/fit6/    -> scores 30min
    chronos2_personalization_results/test/<arm>/fit24/   -> scores 1h,2h,3h,4h

3h/4h come off the fit@24 model and are therefore NOT horizon-matched for
their own horizon; that is a disclosed limitation, printed in the 3h/4h block's
header rather than left implicit. Whichever sbatch runs the arms must write
into this layout -- `arm_dirs()` is the single place it is defined.

Arms that do not exist yet are skipped with a note, so this is runnable (and
worth running) before any arm has finished.
"""
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from chronos2_matched_comparison import (  # noqa: E402
    DEFAULT_EVAL_DIR, compare_arms, load_window_keys)
from chronos2_personalization_power_calibration import mde, per_participant_diffs  # noqa: E402

REPO_ROOT = "/path/to/aireadi-collab"
# CHRONOS2_PERS_TEST_ROOT: lets a smoke run point the personalization arms at a
# stand-in tree, so this script's whole path (compare_arms -> MDE -> verdict ->
# both horizon blocks) can be exercised BEFORE any real arm exists. Without it
# the only pre-arm check possible is "it prints nothing and exits", which tests
# almost none of the logic. See slurm/chronos2_personalization_report_smoke.sbatch.
PERS = os.environ.get("CHRONOS2_PERS_TEST_ROOT",
                      f"{REPO_ROOT}/chronos2_personalization_results/test")
ZS = f"{REPO_ROOT}/chronos2_zeroshot_results"
ZSF = f"{REPO_ROOT}/chronos2_zeroshot_results_ffillfix"
PERS_DIR = f"{REPO_ROOT}/model_input/personalization"

HORIZON_BINS = {"30min": 6, "1h": 12, "2h": 24, "3h": 36, "4h": 48}
PRIMARY = ["30min", "1h", "2h"]
EXTENDED = ["3h", "4h"]
# which fit length supplies each horizon; see "DIRECTORY LAYOUT" above
FIT_FOR_HORIZON = {"30min": "fit6", "1h": "fit24", "2h": "fit24",
                   "3h": "fit24", "4h": "fit24"}

# An effect smaller than this is not clinically interesting even if it were
# significant, so a non-significant result below it is a REAL null rather than
# an underpowered one. 0.1 mg/dL is ~1.3% of the 30min MAE and well inside the
# 0.031 mg/dL shift the whole ffill fix produced across 13 arms -- i.e. the
# scale at which this project has already decided differences do not matter.
NEGLIGIBLE_DMAE = 0.10

# key set name -> participants it expects each arm to fit. Filled by
# load_key_sets(); used by arm_completeness() to tell a partially-fitted arm
# apart from a finished one.
PARTICIPANT_COUNTS = {}


def arm_dirs(arm, condition, bar=7):
    """A personalization arm: per-horizon results dir + condition. The ONE
    place the horizon-matched directory layout is defined.

    `bar` selects the family. bar=7 is the primary family (all arms on one
    shared window set, so 1d/3d/7d are mutually comparable); a lower bar is the
    parallel family, which trades that comparability for ~3x the windows and
    nearly all of the power back. The two live in separate result roots because
    predictions are appended, so a shared root would silently interleave two
    window sets in one file."""
    root = PERS if bar == 7 else f"{PERS}_bar{bar}d"
    return {h: dict(label=arm, condition=condition,
                    results_dir=f"{root}/{arm}/{FIT_FOR_HORIZON[h]}")
            for h in HORIZON_BINS}


def zs_arm(label, condition, results_dir):
    """A zero-shot reference: one directory for every horizon."""
    return {h: dict(label=label, condition=condition, results_dir=results_dir)
            for h in HORIZON_BINS}


GLU = "glucose_only"
COV = "with_covariates"

ARMS = {
    # phase 1
    "p1d": arm_dirs("p1d", GLU),
    "p3d": arm_dirs("p3d", GLU),
    "p7d": arm_dirs("p7d", GLU),
    "p7d_shuffled": arm_dirs("p7d_shuffled", GLU),
    # phase 2 (see slurm/chronos2_personalization_cov_smoke.sbatch for the
    # channel set behind each name)
    "cov_arm1": arm_dirs("cov_arm1_5phys", COV),
    "cov_arm2": arm_dirs("cov_arm2_5phys_meal", COV),
    "cov_arm3": arm_dirs("cov_arm3_5phys_meal_oxysat", COV),
    "cov_arm4": arm_dirs("cov_arm4_5phys_meal_sleep", COV),
    # parallel family, bar=3 / k=3d. Same four channel sets, ~3x the windows
    # (1,192 vs 409 at 1h), MDE roughly halved. NOT comparable to the bar=7
    # arms above -- different windows and different participants.
    "cov3_arm1": arm_dirs("cov_arm1_5phys", COV, bar=3),
    "cov3_arm2": arm_dirs("cov_arm2_5phys_meal", COV, bar=3),
    "cov3_arm3": arm_dirs("cov_arm3_5phys_meal_oxysat", COV, bar=3),
    "cov3_arm4": arm_dirs("cov_arm4_5phys_meal_sleep", COV, bar=3),
    # the glucose-only reference INSIDE the parallel family. Must be the
    # personalized arm at the same k and the same bar; the bar=7 p3d arm is on a
    # different window set and a zero-shot arm answers a different question.
    "p3d_bar3": arm_dirs("p3d", GLU, bar=3),
    # zero-shot references
    "zs_glucose_only": zs_arm("zs_glucose_only", GLU, f"{ZS}/full/imputed"),
    # CORRECTED 2026-09-03 (CHRONOS2_EVAL.md): the best zero-shot config
    # within the cross-model range is drop-SpO2 ALONE, not drop-SpO2 +
    # predmeal_prob. Do not swap this back.
    "zs_best_config": zs_arm("zs_best_config", COV, f"{ZS}/loo/drop_oxygen_saturation"),
    # each phase-2 arm's exact zero-shot twin -- the same pairs the mode=none
    # plumbing check used, which is why "arm N vs its own twin" isolates
    # fine-tuning and nothing else
    "zs_cov_arm1": zs_arm("zs_cov_arm1", COV, f"{ZS}/loo/drop_oxygen_saturation"),
    "zs_cov_arm2": zs_arm("zs_cov_arm2", COV, f"{ZSF}/full/with_predmeal_prob_drop_oxysat"),
    "zs_cov_arm3": zs_arm("zs_cov_arm3", COV, f"{ZSF}/full/with_predmeal_prob"),
    "zs_cov_arm4": zs_arm("zs_cov_arm4", COV, f"{ZSF}/full/with_sleep_stage_predmeal_drop_oxysat"),
}

# (question, arm_a, arm_b, key_set). compare_arms' differential is a - b, so
# arm_b is always the one being tested and a NEGATIVE delta_mae means arm_b
# won. key_set picks which eval set the row is restricted to.
COMPARISONS = [
    # --- phase 1 -------------------------------------------------------
    ("personalization 1d vs zero-shot", "zs_glucose_only", "p1d", "E"),
    ("personalization 3d vs zero-shot", "zs_glucose_only", "p3d", "E"),
    ("personalization 7d vs zero-shot", "zs_glucose_only", "p7d", "E"),
    ("dose-response: 7d vs 1d", "p1d", "p7d", "E"),
    ("dose-response: 7d vs 3d", "p3d", "p7d", "E"),
    ("NEGATIVE CONTROL: 7d own vs 7d donor", "p7d_shuffled", "p7d", "E"),
    ("7d vs best zero-shot config", "zs_best_config", "p7d", "E"),
    # --- phase 2 -------------------------------------------------------
    ("cov arm1 vs its zero-shot twin", "zs_cov_arm1", "cov_arm1", "E_cov"),
    ("cov arm2 vs its zero-shot twin", "zs_cov_arm2", "cov_arm2", "E_cov"),
    ("cov arm3 vs its zero-shot twin", "zs_cov_arm3", "cov_arm3", "E_cov"),
    ("cov arm4 vs its zero-shot twin", "zs_cov_arm4", "cov_arm4", "E_cov"),
    ("isolates predmeal_prob (arm2-arm1)", "cov_arm1", "cov_arm2", "E_cov"),
    ("isolates oxygen_saturation (arm3-arm2)", "cov_arm2", "cov_arm3", "E_cov"),
    ("isolates sleep_stage (arm4-arm2)", "cov_arm2", "cov_arm4", "E_cov"),
    # the actual phase-2 question. Restricted to E_cov so both arms score the
    # identical windows; the phase-1 arm's anchor is the same one the covariate
    # arms use (build_personalization_eval_set.py reuses it), so this is not
    # confounded by anchor position.
    ("do wearables add to a personalized model (arm1 vs p7d)", "p7d", "cov_arm1", "E_cov"),
    # --- phase 2, PARALLEL family (bar=3 / k=3d) -----------------------
    # Same questions, ~3x the windows, MDE roughly halved. Reported as its own
    # block: these rows are NOT comparable to the bar=7 rows above (different
    # windows, different participants), and the 1/3/7-day dose-response lives
    # only in the primary family.
    ("bar3: cov arm1 vs its zero-shot twin", "zs_cov_arm1", "cov3_arm1", "E_cov_bar3d"),
    ("bar3: cov arm2 vs its zero-shot twin", "zs_cov_arm2", "cov3_arm2", "E_cov_bar3d"),
    ("bar3: cov arm3 vs its zero-shot twin", "zs_cov_arm3", "cov3_arm3", "E_cov_bar3d"),
    ("bar3: cov arm4 vs its zero-shot twin", "zs_cov_arm4", "cov3_arm4", "E_cov_bar3d"),
    ("bar3: isolates predmeal_prob (arm2-arm1)", "cov3_arm1", "cov3_arm2", "E_cov_bar3d"),
    ("bar3: isolates oxygen_saturation (arm3-arm2)", "cov3_arm2", "cov3_arm3", "E_cov_bar3d"),
    ("bar3: isolates sleep_stage (arm4-arm2)", "cov3_arm2", "cov3_arm4", "E_cov_bar3d"),
    ("bar3: do wearables add to a personalized model (arm1 vs p3d)",
     "p3d_bar3", "cov3_arm1", "E_cov_bar3d"),
]


def load_key_sets():
    """The window sets each comparison is restricted to. Getting this wrong is
    the one error that would silently invalidate a row: a bar=3 arm scored
    against the bar=7 key set collapses onto the intersection and throws away
    the power the lower bar existed for."""
    wanted = {
        "E": "eval_set_test.csv",
        "E_cov": "eval_set_cov_test.csv",
        "E_bar3d": "eval_set_test_bar3d.csv",
        "E_cov_bar3d": "eval_set_cov_test_bar3d.csv",
    }
    sets = {}
    for name, fname in wanted.items():
        path = f"{PERS_DIR}/{fname}"
        if not os.path.exists(path):
            # a family whose eval set has not been built is simply absent; its
            # comparisons then skip, rather than the whole report failing
            print(f"note: {fname} missing, comparisons on {name} will be skipped")
            continue
        df = pd.read_csv(path, dtype={"person_id": str})
        sets[name] = set(df["key"])
        PARTICIPANT_COUNTS[name] = df["person_id"].nunique()
    if "E" not in sets:
        raise SystemExit(f"{PERS_DIR}/eval_set_test.csv missing -- "
                          "run build_personalization_eval_set.py")
    return sets


def arm_completeness(arm, key_set_name, key_sets):
    """(fitted, expected) participants for one arm, or (None, None) for a
    zero-shot arm (which has no fitting loop and is always complete).

    WHY THIS GATE EXISTS. The report is meant to be run partway through, so it
    will see arms that are still fitting. Those arms' prediction files are
    valid but PARTIAL, and a partial arm produces a row that looks exactly like
    a finished one -- just with fewer participants. The first real run of this
    script emitted a covariate row built from 5 participants and labelled it
    "null, UNDERPOWERED", which is true of the statistics and false about the
    experiment. Incomplete and underpowered are different claims and must not
    share a label."""
    done = f"{arm['results_dir']}/participants_done_shard0.csv"
    if not os.path.exists(done):
        return None, None
    fitted = pd.read_csv(done, dtype={"person_id": str})["person_id"].nunique()
    # Expected = participants the arm's own eval set asks it to fit. Taken from
    # the eval set's person_id column, not parsed out of the key strings (a key
    # is person_id + "_" + str(context_start), and splitting that on "_" is
    # fragile because the timestamp contains no underscore only by luck).
    expected = PARTICIPANT_COUNTS[key_set_name]
    return fitted, expected


# The PUBLICATION-FACING tables: (title, key set, arms in row order).
#
# This is the format TSFM papers actually use -- one row per model/arm, one
# column per horizon, MAE and RMSE -- and it is what should appear in a
# writeup. The pairwise delta/MDE tables below it are the author's working
# notes, not the paper.
#
# Zero-shot references are included in each table so the personalized arms
# have their baseline on the same rows rather than in a separate document.
ARM_TABLES = [
    # zs_best_config is deliberately NOT here, even though "is personalization
    # better than the best zero-shot config" is a phase-1 question. It is a
    # with_covariates arm, so it only scores the covariate subset of E, and
    # since every row in a table shares one window set, including it dragged
    # all of phase 1 from 1,149/1,024/925 windows down to 461/409/379 -- a 60%
    # loss for one row. It appears in the phase-2 table below as zs_cov_arm1,
    # which is the same results directory, on the eval set that actually suits
    # it.
    ("PHASE 1 -- history length, eval set E", "E",
     ["zs_glucose_only", "p1d", "p3d", "p7d", "p7d_shuffled"]),
    ("PHASE 2 -- covariate arms, eval set E_cov (bar=7d, k=7d)", "E_cov",
     ["zs_glucose_only", "p7d",
      "zs_cov_arm1", "cov_arm1", "cov_arm2", "cov_arm3", "cov_arm4"]),
    ("PHASE 2 PARALLEL -- bar=3d / k=3d, eval set E_cov_bar3d", "E_cov_bar3d",
     ["p3d_bar3", "cov3_arm1", "cov3_arm2", "cov3_arm3", "cov3_arm4"]),
]


def arm_scores(arm, horizon, keys):
    """MAE and RMSE for ONE arm restricted to a fixed window set.

    Uses the same `abs_error` column compare_arms scores off, so a number here
    and the matched MAE in the pairwise table below cannot drift apart."""
    wk = load_window_keys(horizon, arm["condition"],
                          arm.get("eval_dir", DEFAULT_EVAL_DIR))
    p = pd.read_csv(
        f"{arm['results_dir']}/predictions_{horizon}_{arm['condition']}.csv",
        dtype={"window_id": str})
    p = p.assign(key=p["window_id"].map(wk["key"].to_dict()))
    p = p[p["key"].isin(keys)].dropna(subset=["actual"])
    if p.empty:
        return None
    return {"mae": float(p["abs_error"].mean()),
            "rmse": float(np.sqrt((p["abs_error"] ** 2).mean())),
            "keys": set(p["key"])}


def arm_table(title, key_set, arm_names, key_sets):
    """One publication table. Every arm is scored on the SAME windows -- the
    intersection of the eval set with every included arm's scored windows.

    INCOMPLETE ARMS ARE EXCLUDED, not intersected in. An arm still fitting
    covers a fraction of the participants, so intersecting it would silently
    shrink the window set for every other arm in the table and make the
    published numbers depend on which arm happened to be mid-run. Excluded
    arms are named underneath so the table is never quietly short a row."""
    if key_set not in key_sets:
        return None, [], f"key set {key_set} not built"

    usable, pending = [], []
    for name in arm_names:
        per_h = ARMS[name]
        if any(not os.path.exists(
                f"{per_h[h]['results_dir']}/predictions_{h}_{per_h[h]['condition']}.csv")
               for h in HORIZON_BINS):
            pending.append(f"{name} (not run)")
            continue
        short = [arm_completeness(per_h[h], key_set, key_sets) for h in HORIZON_BINS]
        worst = [(f, e) for f, e in short if f is not None and f < 0.98 * e]
        if worst:
            f, e = min(worst, key=lambda t: t[0])
            pending.append(f"{name} ({f}/{e} participants fitted)")
            continue
        usable.append(name)

    if not usable:
        return None, pending, "no complete arm in this family yet"

    # common windows per horizon, across the eval set and every usable arm
    scored = {}
    for h in HORIZON_BINS:
        common = set(key_sets[key_set])
        per_arm = {}
        for name in usable:
            s = arm_scores(ARMS[name][h], h, key_sets[key_set])
            if s is None:
                per_arm = {}
                break
            per_arm[name] = s
            common &= s["keys"]
        if not per_arm or not common:
            continue
        # rescore on the intersection only if it is actually narrower, so the
        # common case costs one pass rather than two
        for name in usable:
            s = (per_arm[name] if per_arm[name]["keys"] == common
                 else arm_scores(ARMS[name][h], h, common))
            scored[(name, h)] = s
        scored[("__n__", h)] = len(common)

    if not scored:
        return None, pending, "no horizon has a common window set yet"

    rows = []
    for name in usable:
        row = {"arm": name}
        for h in HORIZON_BINS:
            s = scored.get((name, h))
            row[f"{h}_mae"] = s["mae"] if s else float("nan")
            row[f"{h}_rmse"] = s["rmse"] if s else float("nan")
        rows.append(row)
    n_windows = {h: scored.get(("__n__", h)) for h in HORIZON_BINS}
    return pd.DataFrame(rows), pending, n_windows


def interval(d_person):
    """Effect estimate with intervals. NO categorical verdict -- see below.

    THIS REPLACED TWO BAD GATES IN A ROW, 2026-09-05. Recorded so neither
    comes back.

    Gate 1 returned "helped"/"hurt" on cluster p < 0.05, which made a
    significance test decide the project's conclusions.

    Gate 2 replaced that with "|delta| vs MDE", which I described in this
    docstring as "never a function of any p-value". THAT WAS FALSE. For a
    one-sample t-test:
        significance at alpha=0.05  <=>  |delta| > 1.96 * SE
        MDE                          =   2.802 * SE
    so "|delta| > MDE" is exactly |t| > 2.802, i.e. **p < 0.005** -- the same
    kind of threshold, ten times stricter. It suppressed 17 of the 36 rows
    with p < 0.05, INCLUDING the project's headline result (p7d vs zero-shot
    at 1h, d=-0.436 p=0.036, and at 2h, d=-0.804 p=0.012, both labelled
    "below MDE -- not resolvable").

    It was also incoherent on its own terms. MDE answers a DESIGN question
    asked before running -- "how big must an effect be for me to detect it
    reliably?" Comparing an observed estimate to it afterwards asks "was my
    effect as big as the effect I would have needed to be confident of
    detecting an effect that big?", which is circular. MDE's legitimate use is
    the go/no-go decision before spending the GPU time, which is where it was
    first used correctly, and it stays in the sbatch banners for that.

    What this returns instead is the estimate and its 95% CI, plus the tightest
    bound the data establishes (max |90% CI bound|, the TOST construction) so a
    null can be stated as "rules out effects larger than X" rather than as a
    failure to detect. No label, no threshold. The reader sees the interval.
    Full version with equivalence tests and win rates:
    code/chronos2_personalization_equivalence.py.

    d_person must be the per-participant MAE differential series (a - b)."""
    d = -d_person.to_numpy()  # to this project's convention: negative => b better
    n = len(d)
    mean = float(d.mean())
    se = float(d.std(ddof=1) / np.sqrt(n))
    t95 = stats.t.ppf(0.975, df=n - 1)
    t90 = stats.t.ppf(0.95, df=n - 1)
    return {
        "delta_mae_person": mean,
        "ci95_lo": mean - t95 * se,
        "ci95_hi": mean + t95 * se,
        "excludes_zero": bool((mean - t95 * se) > 0 or (mean + t95 * se) < 0),
        "bounds_effects_above": max(abs(mean - t90 * se), abs(mean + t90 * se)),
    }


def print_arm_tables(key_sets):
    """The publication-facing output, printed FIRST because it is the thing
    that goes in a writeup. MAE/RMSE per arm per horizon, no p-values."""
    pd.set_option("display.width", 240)
    frames = []
    for title, key_set, arm_names in ARM_TABLES:
        print("\n" + "=" * 118)
        print(f"MAE / RMSE BY HORIZON -- {title}")
        print("=" * 118)
        df, pending, meta = arm_table(title, key_set, arm_names, key_sets)
        if df is None:
            print(f"  (skipped: {meta})")
            if pending:
                print("  pending: " + ", ".join(pending))
            continue
        print(df.to_string(index=False, float_format=lambda v: f"{v:.2f}",
                            na_rep="--"))
        print("  windows scored (identical across every row): "
              + ", ".join(f"{h}={meta[h]}" for h in HORIZON_BINS if meta[h]))
        if pending:
            print("  NOT IN THIS TABLE (excluded so they cannot shrink the "
                  "shared window set): " + ", ".join(pending))
        frames.append(df.assign(table=title, eval_set=key_set))
    return frames


def main():
    key_sets = load_key_sets()
    arm_frames = print_arm_tables(key_sets)
    if arm_frames:
        arm_out = f"{REPO_ROOT}/chronos2_personalization_results/test_arm_scores.csv"
        pd.concat(arm_frames, ignore_index=True).to_csv(arm_out, index=False)

    rows, skipped = [], []
    for question, a_name, b_name, key_set in COMPARISONS:
        if key_set not in key_sets:
            skipped.append((question, "all", f"key set {key_set} not built"))
            continue
        for horizon, hbins in HORIZON_BINS.items():
            arm_a = ARMS[a_name][horizon]
            arm_b = ARMS[b_name][horizon]
            missing = [x["results_dir"] for x in (arm_a, arm_b)
                       if not os.path.exists(
                           f"{x['results_dir']}/predictions_{horizon}_{x['condition']}.csv")]
            if missing:
                skipped.append((question, horizon, missing[0]))
                continue
            stat = compare_arms(horizon, hbins, arm_a, arm_b,
                                 restrict_keys=key_sets[key_set])
            d_person, n_paired = per_participant_diffs(arm_a, arm_b, horizon,
                                                        key_sets[key_set])
            mde_mae, sd_mae, n_p = mde(d_person["ab"])
            # THREE DIFFERENT QUANTITIES LIVE HERE. Keeping them straight is
            # not pedantry -- an earlier version of this function fed
            # `delta_mae` (pooled, per-ROW) and compare_arms' cluster p (on the
            # MSE differential) and an MAE-scale MDE (per-PARTICIPANT) into one
            # verdict, and produced |delta| 2.83 > MDE 2.41 at p=0.886, which
            # is arithmetically impossible for a single t-test. Caught by
            # slurm/chronos2_personalization_report_smoke.sbatch. Participants
            # have unequal window counts in the real eval set too (~3.9 windows
            # each at 1h), so the pooled and per-participant means genuinely
            # differ and this was not only a small-n artifact.
            #
            #   delta_mae            pooled over rows -- what this project
            #                        QUOTES, so it is reported
            #   cluster_p_mse        compare_arms' own test, on MSE -- the
            #                        project's PRIMARY test, so it is reported
            #   delta_mae_person /   per-participant MAE, same series the MDE
            #   cluster_p_mae        is derived from -- the only self-consistent
            #                        triple, so the VERDICT uses these
            #
            # per_participant_diffs returns a - b, so a positive mean means
            # arm_b is better; negated here to match delta_mae's convention
            # (negative = arm_b better) so both columns read the same way.
            delta_mae = (stat[f"{arm_b['label']}_mae_matched"]
                         - stat[f"{arm_a['label']}_mae_matched"])
            iv = interval(d_person["ab"])
            delta_mae_person = iv["delta_mae_person"]
            _, cluster_p_mae = stats.ttest_1samp(d_person["ab"].to_numpy(), 0.0)

            # An arm still fitting yields a valid-looking row on a fraction of
            # the population. Verdicts are suppressed for those; the row is kept
            # (the numbers are real, just premature) but labelled so it cannot
            # be read as a result. 2% slack absorbs participants legitimately
            # skipped for too-short history, which are recorded in
            # skipped_shard<N>.csv and are not incompleteness.
            fitted = [arm_completeness(a, key_set, key_sets) for a in (arm_a, arm_b)]
            shortfall = [(f, e) for f, e in fitted if f is not None and f < 0.98 * e]
            # The completeness gate STAYS. Unlike the two inference gates it
            # replaced, it is not a claim about effect size -- it says the arm
            # has not finished fitting, so the row is premature rather than
            # small. Those are different statements.
            if shortfall:
                f, e = min(shortfall, key=lambda t: t[0])
                row_note = f"INCOMPLETE ({f}/{e} participants fitted)"
            else:
                row_note = ""

            rows.append({
                "question": question, "horizon": horizon,
                "arm_a": arm_a["label"], "arm_b": arm_b["label"],
                "eval_set": key_set,
                "a_mae": stat[f"{arm_a['label']}_mae_matched"],
                "b_mae": stat[f"{arm_b['label']}_mae_matched"],
                "delta_mae": delta_mae,
                "delta_mae_person": delta_mae_person,
                "ci95_lo": iv["ci95_lo"], "ci95_hi": iv["ci95_hi"],
                "excludes_zero": iv["excludes_zero"],
                "bounds_effects_above": iv["bounds_effects_above"],
                "n_windows": n_paired, "n_participants": n_p,
                "naive_p": stat["naive_paired_p"],
                "dm_hln_p": stat["dm_hln_p"],
                "cluster_p_mse": stat["cluster_by_participant_p"],
                "cluster_p_mae": cluster_p_mae,
                # RETAINED, but for prospective design only -- it must not be
                # compared against delta_mae_person to classify a row. See
                # interval()'s docstring.
                "mde_dMAE_80pct": mde_mae,
                "note": row_note,
            })

    if not rows:
        print("no comparisons runnable yet -- no arm has predictions on disk.")
        print(f"({len(skipped)} (question, horizon) pairs skipped; first missing: "
              f"{skipped[0][2] if skipped else 'n/a'})")
        return

    df = pd.DataFrame(rows)
    out = f"{REPO_ROOT}/chronos2_personalization_results/test_report.csv"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    df.to_csv(out, index=False)

    # cluster_p_mae is shown (not cluster_p_mse) because it is on the same
    # per-participant MAE series as delta_mae_person and the CI, so the three
    # are mutually consistent. cluster_p_mse and the MDE stay in the CSV.
    show = ["question", "horizon", "a_mae", "b_mae",
            "delta_mae_person", "ci95_lo", "ci95_hi", "excludes_zero",
            "bounds_effects_above", "n_windows", "n_participants",
            "cluster_p_mae", "note"]
    pd.set_option("display.width", 240)
    pd.set_option("display.max_colwidth", 44)

    print("\n\n" + "#" * 120)
    print("# EVERYTHING BELOW IS INTERNAL WORKING NOTES, NOT PAPER OUTPUT.")
    print("#")
    print("# The pairwise tables carry cluster_p_* columns, kept for the "
          "author's own reference. NOTHING")
    print("# labels a row: there is no verdict column, because both attempts "
          "at one turned out to be")
    print("# significance thresholds in disguise (see interval()'s docstring). "
          "Read the CI.")
    print("# What belongs in a writeup is the MAE/RMSE tables above, plus the "
          "MDE whenever a null is stated")
    print("# (\"not resolvable at n=194, MDE 0.39-1.08 mg/dL\", never "
          "\"wearables do not help\").")
    print("#" * 120)

    print("=" * 120)
    print("CROSS-MODEL COMPARABLE HORIZONS (30min/1h/2h)")
    print("  The only horizons CGM-LSM and SSM-CGM report, so the only range in "
          "which model-vs-model is answerable.")
    print("  delta_mae < 0 means arm_b (the arm under test) is better.")
    print("=" * 120)
    print(df[df["horizon"].isin(PRIMARY)][show]
          .to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    print("\n" + "=" * 120)
    print("BEYOND THE CROSS-MODEL RANGE (3h/4h)")
    print("  No baseline model reaches these horizons, so they cannot enter the "
          "cross-model table -- but they are")
    print("  a genuine finding about foundation-model behaviour at long horizons "
          "and are reported on their own")
    print("  merits. Do NOT read their exclusion from the comparison as "
          "de-emphasis.")
    print("  DISCLOSED: these are scored from the fit@24 model, so unlike "
          "30min/1h/2h they are not horizon-matched.")
    print("=" * 120)
    print(df[df["horizon"].isin(EXTENDED)][show]
          .to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    bounded = df[(~df["excludes_zero"]) & (df["note"] == "")]
    if not bounded.empty:
        print(f"\n{len(bounded)} of {len(df)} rows have a 95% CI containing zero.")
        print("  State these as a BOUND, not as a failure: 'rules out effects "
              "larger than <bounds_effects_above> mg/dL',")
        print("  never as 'X does not help'. The intervals here are tight; they "
              "just include zero.")
        print(f"  Widest bound among them: "
              f"{bounded['bounds_effects_above'].max():.3f} mg/dL; "
              f"median {bounded['bounds_effects_above'].median():.3f}.")

    if skipped:
        print(f"\n{len(skipped)} (question, horizon) pairs skipped -- arm not run yet:")
        # "all" is used when a whole comparison is skipped because its key set
        # was never built, so it is not a horizon name and must not be looked up
        # in HORIZON_BINS (that raised ValueError).
        order = list(HORIZON_BINS)
        for q in sorted({s[0] for s in skipped}):
            hs = sorted({s[1] for s in skipped if s[0] == q},
                        key=lambda h: order.index(h) if h in order else len(order))
            print(f"  {q}: {hs}")

    print(f"\nwritten: {out}")


if __name__ == "__main__":
    main()
