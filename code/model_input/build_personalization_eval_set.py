"""Carves out the personalization fine-tuning eval set E and each
participant's fine-tuning anchor, for the proposal's second research question
(1/3/7 days of per-person history). Consumed by
code/chronos2/chronos2_personalization.py; nothing here touches a model.

WHY THIS FILE EXISTS AT ALL -- the causal boundary
==================================================
Fine-tuning must only ever see the held-out participant's OWN past. A global
fine-tune on the train split is a different experiment. So for each test
participant p:

    t0(p)  = first real glucose timestamp in p's own cleaned record
    E(p)   = p's eligible windows (window_index.csv, unchanged eligibility)
             whose context_start >= t0(p) + HISTORY_BAR_DAYS
    T(p)   = min(context_start) over E(p)              <- the ANCHOR
    history for the k-day arm = p's own bins in [T(p) - k days, T(p))

Three properties this buys, in order of importance:

1. Airtight against the eval horizon. Every window in E(p) starts at or
   after T(p), so every horizon bin is at or after T(p) + 12h, while all
   history ends strictly before T(p). History and horizon are disjoint by
   construction, not by a check that could be got wrong.

2. Nested histories, one shared endpoint. 1d subset 3d subset 7d, all ending
   at the same instant T(p). The only thing that varies across the three arms
   is how MUCH history the model got -- not which period it came from. That
   is what makes "does 7 days beat 3" a meaningful comparison rather than a
   confound between history length and calendar position.

3. Enforced by slicing, not masking. chronos2_personalization.py slices to
   [T-k, T) BEFORE calling .fit(), so post-boundary data is never in the
   process at all and no amount of internal window sampling inside
   Chronos2Pipeline.fit() can reach it. report.md Sec 15.7's reverted
   interpolation is the cautionary precedent: that leak came from a fill
   applied before the context/horizon boundary existed.

WHY THE 7-DAY BAR APPLIES TO EVERY ARM, INCLUDING THE 1-DAY ONE
===============================================================
AI-READI is a ~10-day Dexcom G6 wear (measured: median record 9.91 days), and
a participant's first eligible window sits only ~0.45 days into it. So 7 days
of prior history only exists for windows in the last ~3 days of a record.
Letting each arm use every window it is individually eligible for would give
the three arms three different, progressively later, progressively smaller
window sets -- and any 1d-vs-7d difference would then mix "amount of history"
with "which windows and which participants got scored."

So HISTORY_BAR_DAYS is fixed at the LARGEST history length being tested (7),
for every arm and for the zero-shot reference arms too. The cost is real and
should be quoted with the results: E is roughly a quarter of the full test
window set, and every claim is about windows late in a recording. The
alternative (per-arm eligibility, larger samples, no cross-arm comparison)
was considered and rejected because the research question is a dose-response
one. See CHRONOS2_PERSONALIZATION.md.

The single anchor also means that for p's LATER windows we withhold recent
history a real deployment would have had. That is deliberate and
conservative: it can only understate personalization, never inflate it. The
realistic alternative -- a rolling boundary, one fit per window at
[context_start - k, context_start) -- costs ~4x the fits and is planned only
as a subsample confirmation if the anchored version shows an effect.

t0 SOURCE
=========
t0 comes from final_cleaned_data/glucose/<pid>.csv's first final_timestamp,
i.e. the participant's full cleaned glucose record -- NOT from
build_grid()'s output, which left-joins glucose onto the analysis_dataframe
wearable grid and is therefore bounded by wearable coverage. Phase 1 is
glucose-only, so the wearable bound is irrelevant to it and using the fuller
record gives every participant slightly more usable history. Verified that
load_cleaned_glucose_binned()'s 5-minute bins land on exactly the same
timestamps as the zeroshot_eval export's (0 value mismatches on real bins
across sampled participants), so a history slice and an eval window's context
are on one shared grid.

PHASE 2 (COVARIATES): eval_set_cov_<split>.csv, AND WHY THE ANCHOR IS REUSED
============================================================================
build_cov() emits the same file for the `with_covariates` condition. It does
NOT recompute the anchor from the covariate-eligible window set. It reuses the
glucose_only anchor T(p) computed above, for one reason that matters:

  E_cov(p) is a strict SUBSET of E_gluc(p) -- the with_covariates eligibility
  rule (heart_rate + respiratory_rate coverage, windows.py) is the glucose
  rule plus two more channels, on the identical non-overlapping window grid,
  and the 7-day bar is applied identically. So min(context_start) over
  E_cov(p) is at or after T(p), and T(p) is still a valid (indeed strictly
  conservative) causal boundary for every covariate window.

Recomputing a later, covariate-specific anchor would give each participant
slightly more recent history -- but it would also mean the phase-2 covariate
arms and the phase-1 glucose-only arm fit on DIFFERENT calendar history, so a
covariate-vs-glucose-only difference would confound channels with anchor
position. The whole point of phase 2 is that channels are the only thing that
varies. Reusing T(p) keeps that true across phases, not just within phase 2.

Asserted, not assumed: build_cov() checks context_start >= anchor for every
covariate window, so if the subset relation above ever stops holding the
build crashes rather than emitting a leaky eval set.

OUTPUT
======
model_input/personalization/eval_set_<split>.csv, one row per (horizon,
window) in E:

    person_id, horizon, window_id, key, context_start, target_start,
    t0, offset_days, anchor, history_start_1d, history_start_3d,
    history_start_7d

`key` is built to be byte-identical to chronos2_matched_comparison.
load_window_keys()'s own key (person_id + "_" + str(context_start), UTC
tz-aware), by calling that function rather than reformatting timestamps here
-- so the file can be fed straight to compare_arms(restrict_keys=...) to
re-score any existing zero-shot arm on E without re-running it.
"""
import os
import subprocess
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
# Sibling modules live in adjacent topic directories under code/.
_CODE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _d in ("analysis", "baselines", "cgmlsm", "chronos2", "meal_detection", "model_input", "ssmcgm", "timesfm"):
    sys.path.insert(0, os.path.join(_CODE, _d))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from chronos2_matched_comparison import load_window_keys  # noqa: E402

REPO_ROOT = "/path/to/aireadi-collab"
GLUCOSE_DIR = f"{REPO_ROOT}/final_cleaned_data/glucose"
WINDOW_INDEX_PATH = f"{REPO_ROOT}/model_input/window_index.csv"
OUTPUT_DIR = f"{REPO_ROOT}/model_input/personalization"

# Fixed at the largest history length tested, for every arm. See the module
# docstring's "WHY THE 7-DAY BAR APPLIES TO EVERY ARM".
#
# CHRONOS2_PERS_HISTORY_BAR_DAYS overrides it, for the SECOND, PARALLEL family
# of results the docstring's fallback describes: one 2-arm experiment per k,
# each on its own larger k-eligible window set. Lowering the bar to B admits
# every window >= B days into a participant's record, which is a much larger
# set -- at the cost that a B=1 set and a B=3 set contain different windows and
# different participants, so arms across them are NOT comparable to each other.
# That is the whole trade, and it is why this is a separate family rather than
# a replacement: the primary analysis keeps bar=7 for all arms so 1d/3d/7d stay
# mutually comparable.
#
# Only k <= bar is meaningful: a k-day history cannot fit inside a record that
# is only guaranteed to extend `bar` days before the window.
HISTORY_BAR_DAYS = int(os.environ.get("CHRONOS2_PERS_HISTORY_BAR_DAYS", "7"))
HISTORY_DAYS = tuple(k for k in (1, 3, 7) if k <= HISTORY_BAR_DAYS)
# bar=7 keeps the original filenames untouched so nothing downstream moves.
BAR_SUFFIX = "" if HISTORY_BAR_DAYS == 7 else f"_bar{HISTORY_BAR_DAYS}d"

# Phase 1 is glucose-only. See CHRONOS2_PERSONALIZATION.md for why: the
# with_covariates eligibility rule (heart_rate + respiratory_rate coverage)
# stacked on top of the 7-day bar cuts E to 409 windows / 194 participants at
# 1h, versus 1,024 / 260 for glucose_only, and the covariate effects this
# project has been resolving are 0.05-0.5 mg/dL -- not recoverable at that
# sample size. Covariates are phase 2.
CONDITION = "glucose_only"
HORIZONS = ("30min", "1h", "2h", "3h", "4h")


def first_glucose_timestamp(person_id):
    """t0(p), read from the first data line of the participant's cleaned
    glucose file rather than by parsing the whole file. These are ~29k-row
    files and this runs for every participant in the split; `head -2` keeps
    it to two lines each. Returns a tz-aware UTC Timestamp, or None if the
    file is missing or has no data rows.

    Relies on final_cleaned_data/glucose/*.csv being sorted by
    final_timestamp. Checked rather than assumed -- see verify_t0_sorted()."""
    path = f"{GLUCOSE_DIR}/{person_id}.csv"
    if not os.path.exists(path):
        return None
    head = subprocess.run(["head", "-2", path], capture_output=True, text=True).stdout
    lines = head.strip().split("\n")
    if len(lines) < 2:
        return None
    cols = lines[0].split(",")
    if "final_timestamp" not in cols:
        return None
    value = lines[1].split(",")[cols.index("final_timestamp")]
    ts = pd.to_datetime(value, utc=True, errors="coerce")
    return None if pd.isna(ts) else ts


def verify_t0_sorted(person_ids, n_check=25):
    """first_glucose_timestamp() trusts that the glucose files are
    chronologically sorted. That is true of how build_finalized_glucose
    writes them, but a wrong t0 would silently shift every anchor and every
    history slice, so it is checked directly on a sample instead of assumed.
    Raises rather than warning: a bad t0 invalidates the causal boundary,
    which is not something to proceed past with a printed warning."""
    for pid in list(person_ids)[:n_check]:
        path = f"{GLUCOSE_DIR}/{pid}.csv"
        if not os.path.exists(path):
            continue
        ts = pd.read_csv(path, usecols=["final_timestamp"])["final_timestamp"]
        ts = pd.to_datetime(ts, utc=True, errors="coerce").dropna()
        if ts.empty:
            continue
        if ts.iloc[0] != ts.min():
            raise SystemExit(
                f"glucose file for {pid} is not chronologically sorted "
                f"(first row {ts.iloc[0]}, min {ts.min()}) -- "
                "first_glucose_timestamp()'s head-2 shortcut is invalid, "
                "switch it to a full read before trusting any anchor"
            )
    print(f"t0 sort check passed on {min(n_check, len(person_ids))} participants", file=sys.stderr)


def build(split):
    index = pd.read_csv(WINDOW_INDEX_PATH, dtype={"person_id": str})
    split_pids = sorted(index.loc[index["recommended_split"] == split, "person_id"].unique())
    if not split_pids:
        raise SystemExit(f"no participants with recommended_split={split!r}")
    verify_t0_sorted(split_pids)

    t0_by_pid = {pid: first_glucose_timestamp(pid) for pid in split_pids}
    missing = [p for p, t in t0_by_pid.items() if t is None]
    if missing:
        print(f"WARNING: no t0 for {len(missing)} participants, excluded: {missing[:5]}...",
              file=sys.stderr)

    eval_dir = (f"{REPO_ROOT}/model_input/zeroshot_eval" if split == "test"
                else f"{REPO_ROOT}/model_input/zeroshot_eval_{split}")

    rows = []
    for horizon in HORIZONS:
        # keys come from load_window_keys so `key` is byte-identical to what
        # compare_arms() will build for the same window; context_start is the
        # min bin_start over the window's is_context rows, which is the same
        # quantity window_index.csv calls context_start but re-derived from
        # the export the model actually reads.
        keys = load_window_keys(horizon, CONDITION, eval_dir=eval_dir).reset_index()
        keys["t0"] = keys["person_id"].map(t0_by_pid)
        keys = keys.dropna(subset=["t0"])
        keys["offset_days"] = (keys["context_start"] - keys["t0"]).dt.total_seconds() / 86400.0

        eligible = keys[keys["offset_days"] >= HISTORY_BAR_DAYS].copy()
        if eligible.empty:
            print(f"WARNING: horizon={horizon}: 0 windows clear the {HISTORY_BAR_DAYS}-day bar",
                  file=sys.stderr)
            continue
        # The anchor is per (participant, horizon-set)? No -- per participant,
        # shared across horizons, so that ONE fine-tuned model per
        # (participant, k) serves every horizon. Taking the min across all
        # horizons' eligible windows would let a 30min window sit earlier than
        # the anchor derived from 4h windows; taking the max across horizons
        # would discard windows needlessly. Resolved by computing the anchor
        # once, below, over the union of every horizon's eligible windows.
        eligible["horizon"] = horizon
        rows.append(eligible)

    if not rows:
        raise SystemExit(f"no eligible windows at all for split={split}")
    e = pd.concat(rows, ignore_index=True)

    # ONE anchor per participant, over the union of all horizons' eligible
    # windows -- the earliest window any horizon will be scored on. Every
    # eligible window at every horizon is therefore at or after the anchor,
    # so a single fine-tuned model per (participant, k) is leak-free for all
    # five horizons simultaneously. Doing it per horizon instead would mean
    # five fits per (participant, k) for no scientific gain.
    anchors = e.groupby("person_id")["context_start"].min().rename("anchor")
    e = e.merge(anchors, left_on="person_id", right_index=True, how="left")
    for k in HISTORY_DAYS:
        e[f"history_start_{k}d"] = e["anchor"] - pd.Timedelta(days=k)

    # Invariants. These are the causal boundary, so they are asserted, not
    # printed: if any of them fails the whole experiment is invalid and the
    # right outcome is a crash, not a results file with a warning above it.
    assert (e["context_start"] >= e["anchor"]).all(), \
        "a window starts before its own participant's anchor"
    max_k = max(HISTORY_DAYS)
    assert (e["anchor"] - e[f"history_start_{max_k}d"]
            == pd.Timedelta(days=max_k)).all()
    # the 7-day history must fit inside the record, which is exactly what the
    # 7-day bar guarantees -- re-checked here against t0 directly rather than
    # trusting the filter above
    assert (e[f"history_start_{max_k}d"] >= e["t0"]).all(), \
        f"a {max_k}-day history slice would start before the participant's record does"

    e = e.sort_values(["person_id", "horizon", "context_start"]).reset_index(drop=True)
    out_cols = ["person_id", "horizon", "window_id", "key", "context_start", "t0",
                "offset_days", "anchor"] + [f"history_start_{k}d" for k in HISTORY_DAYS]
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = f"{OUTPUT_DIR}/eval_set_{split}{BAR_SUFFIX}.csv"
    e[out_cols].to_csv(out_path, index=False)

    print(f"\n{out_path}", file=sys.stderr)
    print(f"  {e['person_id'].nunique()} participants, anchor bar = "
          f"{HISTORY_BAR_DAYS} days after each participant's own t0", file=sys.stderr)
    per_h = e.groupby("horizon").agg(windows=("key", "nunique"),
                                      participants=("person_id", "nunique"))
    print(per_h.to_string(), file=sys.stderr)
    span = (e.groupby("person_id")["context_start"].max()
            - e.groupby("person_id")["anchor"].first()).dt.total_seconds() / 86400
    print(f"  eval span after anchor: median {span.median():.2f} d, max {span.max():.2f} d",
          file=sys.stderr)
    return e


def build_cov(split):
    """Phase 2's eval set: the same construction restricted to the
    `with_covariates` condition, reusing the glucose_only anchors. See the
    module docstring's "PHASE 2" section for why the anchor is not
    recomputed."""
    gluc_path = f"{OUTPUT_DIR}/eval_set_{split}{BAR_SUFFIX}.csv"
    if not os.path.exists(gluc_path):
        raise SystemExit(f"{gluc_path} missing -- run build({split!r}) first; "
                          "the covariate eval set reuses its anchors")
    gluc = pd.read_csv(gluc_path, dtype={"person_id": str})
    anchors = (gluc.groupby("person_id")
               .agg(anchor=("anchor", "first"), t0=("t0", "first")))
    anchors["anchor"] = pd.to_datetime(anchors["anchor"], utc=True)
    anchors["t0"] = pd.to_datetime(anchors["t0"], utc=True)

    eval_dir = (f"{REPO_ROOT}/model_input/zeroshot_eval" if split == "test"
                else f"{REPO_ROOT}/model_input/zeroshot_eval_{split}")

    rows = []
    for horizon in HORIZONS:
        keys = load_window_keys(horizon, "with_covariates", eval_dir=eval_dir).reset_index()
        keys["t0"] = keys["person_id"].map(anchors["t0"])
        # a participant absent from the glucose_only eval set has no anchor and
        # therefore no leak-free fit; dropped rather than anchored from the
        # covariate windows alone, which would break the shared-anchor property
        keys = keys.dropna(subset=["t0"])
        keys["offset_days"] = (keys["context_start"] - keys["t0"]).dt.total_seconds() / 86400.0
        eligible = keys[keys["offset_days"] >= HISTORY_BAR_DAYS].copy()
        if eligible.empty:
            print(f"WARNING: horizon={horizon}: 0 with_covariates windows clear "
                  f"the {HISTORY_BAR_DAYS}-day bar", file=sys.stderr)
            continue
        eligible["horizon"] = horizon
        rows.append(eligible)
    if not rows:
        raise SystemExit(f"no eligible with_covariates windows for split={split}")
    e = pd.concat(rows, ignore_index=True)
    e = e.merge(anchors["anchor"], left_on="person_id", right_index=True, how="left")
    for k in HISTORY_DAYS:
        e[f"history_start_{k}d"] = e["anchor"] - pd.Timedelta(days=k)

    # The causal boundary, re-checked against the REUSED anchor. If the
    # E_cov-subset-of-E_gluc relation the anchor reuse rests on ever breaks,
    # this is where it surfaces.
    assert (e["context_start"] >= e["anchor"]).all(), \
        "a with_covariates window starts before its participant's glucose_only " \
        "anchor -- E_cov is no longer a subset of E_gluc, recompute the anchor"
    assert (e[f"history_start_{max(HISTORY_DAYS)}d"] >= e["t0"]).all()

    e = e.sort_values(["person_id", "horizon", "context_start"]).reset_index(drop=True)
    out_cols = ["person_id", "horizon", "window_id", "key", "context_start", "t0",
                "offset_days", "anchor"] + [f"history_start_{k}d" for k in HISTORY_DAYS]
    out_path = f"{OUTPUT_DIR}/eval_set_cov_{split}{BAR_SUFFIX}.csv"
    e[out_cols].to_csv(out_path, index=False)
    print(f"\n{out_path}", file=sys.stderr)
    print(f"  {e['person_id'].nunique()} participants (anchors reused from "
          f"eval_set_{split}{BAR_SUFFIX}.csv)", file=sys.stderr)
    print(e.groupby("horizon").agg(windows=("key", "nunique"),
                                    participants=("person_id", "nunique")).to_string(),
          file=sys.stderr)
    return e


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if a != "--cov"]
    cov = "--cov" in sys.argv[1:]
    for split in args or ["test"]:
        if cov:
            build_cov(split)
        else:
            build(split)
