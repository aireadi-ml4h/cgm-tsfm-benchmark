"""Ranks per-participant fine-tuning configs on val, and reports the donor
negative control alongside every config.

SELECTION CRITERION: own_vs_zs -- how much better the personalized model is
than the same base with no fine-tuning. That is the deliverable.

CORRECTED 2026-09-07. An earlier version of this script selected on the
OWN-MINUS-DONOR increment, reasoning that own-vs-zeroshot partly rewards
whichever recipe most aggressively adapts to the population rather than to the
person. That concern is real, but maximising own-minus-donor is the wrong
answer to it, and the val data shows why: own-minus-donor can be inflated by
DAMAGING THE DONOR ARM rather than by improving the personalized one. At k=7d
it selected lr=2e-3/100 steps, where a stranger's history is 3.5 mg/dL WORSE
than zero-shot and the own-history gain is only +1.93 -- while ranking on
own_vs_zs selects lr=2e-3/25 steps, where the own-history gain is +3.78, twice
as large. The gap-maximising criterion picked a recipe that personalizes half
as well because it is more destructive when pointed at the wrong person.

So the donor arm is a NEGATIVE CONTROL -- a validity check on whether a gain is
person-specific -- not a quantity to maximise. Both numbers are reported for
every config:
  own_vs_zs        -- the effect, and the selection criterion.
  donor_vs_zs      -- what a stranger's history reproduces.
  own_minus_donor  -- the person-specific part, reported NOT optimised.

NOTE FOR REPORTING: at the selected configs a donor's history is strongly
POSITIVE (~+2.1 to +2.4 mg/dL), i.e. fine-tuning on ANY participant's recent
glucose helps this base substantially and fine-tuning on your own helps more.
That is the OPPOSITE of the Chronos-2 arm, where a donor's history was worse
than zero-shot at every horizon and the control therefore "passed" cleanly.
For CGM-LSM the honest statement is two-part: total gain vs zero-shot, and the
smaller donor-controlled portion that is attributable to the person. Do not
report the total as if it were all personalization.

Ranked SEPARATELY PER HISTORY LENGTH; each k gets its own frozen recipe.
P-values are author-reference only and must not label anything.
"""
import glob
import os
import re
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cgmlsm_matched_comparison import compare_arms, common_scored_keys  # noqa: E402

REPO_ROOT = os.environ["AIREADI_ROOT"]
VAL_ROOT = os.environ.get(
    "CGMLSM_PERS_VAL_ROOT",
    os.path.join(REPO_ROOT, "cgmlsm_personalization_results",
                 "valperk_finetune_sweep4_ep2_lr1e-2"),
)
ZEROSHOT = os.path.join(VAL_ROOT, "zeroshot_ref", "zs_glucose_only")
HORIZONS = ("30min", "1h", "2h")
# Per-k layout: k<K>_lr<LR>_s<STEPS>/p<K> paired with
# donor_k<K>_lr<LR>_s<STEPS>/p<K>_shuffled.
COMBO_RE = re.compile(r"^k(?P<k>\d+d)_lr(?P<lr>[0-9.e+-]+)_s(?P<steps>\d+)$")


def discover():
    """Every (k, lr, steps) cell that has BOTH an own-history and a matching
    donor arm. A cell missing either one is skipped loudly rather than ranked
    on a half-comparison -- the selection criterion is a difference between the
    two, so one arm alone cannot produce it."""
    out = []
    for path in sorted(glob.glob(os.path.join(VAL_ROOT, "k*_lr*_s*"))):
        name = os.path.basename(path)
        m = COMBO_RE.match(name)
        if not m:
            continue
        k = m.group("k")
        own = os.path.join(path, f"p{k}")
        donor = os.path.join(VAL_ROOT, f"donor_{name}", f"p{k}_shuffled")
        if not (os.path.isdir(own) and os.path.isdir(donor)):
            print(f"  skipping {name}: missing own or donor arm")
            continue
        out.append({"combo": name, "k": k, "own": own, "donor": donor,
                    "lr": float(m.group("lr")), "steps": int(m.group("steps"))})
    return out


def main():
    if not os.path.isdir(ZEROSHOT):
        raise SystemExit(f"zero-shot reference missing at {ZEROSHOT}")
    combos = discover()
    if not combos:
        raise SystemExit(f"no paired own/donor combos under {VAL_ROOT}")
    print(f"ranking {len(combos)} configs in {VAL_ROOT}\n")

    rows = []
    for c in combos:
        for horizon in HORIZONS:
            keys = common_scored_keys([ZEROSHOT, c["own"], c["donor"]], horizon)
            if not keys:
                continue
            keys = pd.MultiIndex.from_tuples(sorted(keys))
            trio = {}
            for label, a, b in (("own_vs_zs", ZEROSHOT, c["own"]),
                                ("donor_vs_zs", ZEROSHOT, c["donor"]),
                                ("own_vs_donor", c["donor"], c["own"])):
                r = compare_arms(a, b, horizon, restrict_keys=keys, label_a="a", label_b="b")
                if r is not None:
                    trio[label] = r
            if "own_vs_donor" not in trio:
                continue
            rows.append({
                "combo": c["combo"], "k": c["k"], "lr": c["lr"], "steps": c["steps"],
                "horizon": horizon,
                "n_participants": trio["own_vs_donor"]["n_participants"],
                "n_bins": trio["own_vs_donor"]["n_bins_scored_matched"],
                "mae_zeroshot": trio["own_vs_zs"]["mae_a"] if "own_vs_zs" in trio else None,
                "mae_own": trio["own_vs_zs"]["mae_b"] if "own_vs_zs" in trio else None,
                "mae_donor": trio["donor_vs_zs"]["mae_b"] if "donor_vs_zs" in trio else None,
                "own_vs_zs": trio["own_vs_zs"]["delta_mae"] if "own_vs_zs" in trio else None,
                "donor_vs_zs": trio["donor_vs_zs"]["delta_mae"] if "donor_vs_zs" in trio else None,
                "own_vs_donor": trio["own_vs_donor"]["delta_mae"],
                "own_vs_donor_ci_lo": trio["own_vs_donor"]["delta_mae_ci_lo"],
                "own_vs_donor_ci_hi": trio["own_vs_donor"]["delta_mae_ci_hi"],
                "own_vs_donor_win_rate": trio["own_vs_donor"]["win_rate_b"],
                "own_vs_donor_p": trio["own_vs_donor"]["cluster_by_participant_p"],
                "n_nonnumeric_own": trio["own_vs_zs"]["n_nonnumeric_b"] if "own_vs_zs" in trio else None,
            })

    if not rows:
        raise SystemExit("no comparisons produced")
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(VAL_ROOT, "val_rank_own_minus_donor.csv"), index=False)

    print("=== per config and horizon (POSITIVE = the later arm is better) ===")
    print(out[["combo", "horizon", "n_participants", "mae_zeroshot", "mae_own", "mae_donor",
               "own_vs_zs", "donor_vs_zs", "own_vs_donor",
               "own_vs_donor_ci_lo", "own_vs_donor_ci_hi"]].round(3).to_string(index=False))

    # RANKED SEPARATELY PER HISTORY LENGTH. One global winner would impose a
    # single recipe across k, which is the exact failure the per-k step budgets
    # were designed to avoid.
    best_per_k = {}
    for k in sorted(out["k"].unique(), key=lambda x: int(x.rstrip("d"))):
        sub = out[out["k"] == k]
        rank = (sub.groupby(["combo", "lr", "steps"])
                   .agg(mean_own_vs_zs=("own_vs_zs", "mean"),
                        mean_donor_vs_zs=("donor_vs_zs", "mean"),
                        mean_own_minus_donor=("own_vs_donor", "mean"))
                   .reset_index()
                   .sort_values("mean_own_vs_zs", ascending=False)
                   .reset_index(drop=True))
        print(f"\n=== k={k}: ranked on mean own_vs_zs over 30min/1h/2h "
              f"(donor control shown, not optimised) ===")
        print(rank.round(4).to_string(index=False))
        b = rank.iloc[0]
        best_per_k[k] = b
        print(f"  best at k={k}: lr={b['lr']:g}, steps={b['steps']}, "
              f"own_vs_zs = {b['mean_own_vs_zs']:+.4f} mg/dL "
              f"(donor {b['mean_donor_vs_zs']:+.4f}, "
              f"person-specific part {b['mean_own_minus_donor']:+.4f})")
        if len(rank) >= 3:
            spread = float(b["mean_own_vs_zs"] - rank["mean_own_vs_zs"].iloc[2])
            note = ("  <- knife-edge; val is not resolving these, treat the winner as "
                    "arbitrary among the top few" if spread < 0.15 else "")
            print(f"  spread best-to-3rd: {spread:.4f} mg/dL{note}")
        lrs = sorted(rank["lr"].unique())
        stps = sorted(rank["steps"].unique())
        edges = []
        if len(lrs) > 1 and b["lr"] in (lrs[0], lrs[-1]):
            edges.append(f"lr={b['lr']:g} on the {'low' if b['lr'] == lrs[0] else 'high'} edge of {lrs}")
        if len(stps) > 1 and b["steps"] in (stps[0], stps[-1]):
            edges.append(f"steps={b['steps']} on the "
                         f"{'low' if b['steps'] == stps[0] else 'high'} edge of {stps}")
        for e in edges:
            print(f"  WARNING: {e}")
        if not edges:
            print("  interior to the grid on both axes")

    print("\n=== proposed frozen config ===")
    for k, b in best_per_k.items():
        flag = "" if b["mean_own_minus_donor"] > 0 else "   <- NO person-specific gain at this k"
        print(f"  k={k}: lr={b['lr']:g}, steps={b['steps']}"
              f"  (own_vs_zs {b['mean_own_vs_zs']:+.4f}, of which "
              f"person-specific {b['mean_own_minus_donor']:+.4f}){flag}")

    print("\nNOTE: recipes are tuned PER k, so no separate k-transfer check is\n"
          "outstanding -- that was only required when one k's recipe was going to be\n"
          "reused at another.")
    print("CGMLSM_PERS_VAL_RANK_CONTROL_OK")


if __name__ == "__main__":
    main()
