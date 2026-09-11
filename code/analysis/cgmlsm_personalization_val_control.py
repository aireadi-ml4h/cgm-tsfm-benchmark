"""The negative control for the CGM-LSM personalization arm, on val.

THE QUESTION THIS ANSWERS. The val sweep found a very large gain from
per-participant fine-tuning (+24.6 mg/dL matched at 2h at lr=2e-4/50 steps).
There are two rival explanations and they support completely different claims:

  (A) ADAPTATION TO THE PERSON. Fine-tuning on this participant's own recent
      glucose teaches the model their individual dynamics/level/variability.
      This is personalization, and it is the claim RQ2 is about.

  (B) FINISHING THE POPULATION ADAPTATION. The base checkpoint is only lightly
      adapted to AI-READI (GlucoFM-Bench pretrain + ONE epoch at lr=5e-6). Any
      50 steps at lr=2e-4 on any AI-READI participant's glucose might close
      most of that gap. Then the gain is a statement about an underfit base
      checkpoint, not about individuals at all.

These are separated by fitting on a DONOR participant's history and scoring on
this participant's windows. Under (A) the donor arm should be no better than
zero-shot -- possibly worse, since it adapts toward the wrong person. Under (B)
the donor arm should gain about as much as the own-history arm. The Chronos-2
arm ran exactly this control and it passed there (a donor's history was worse
than zero-shot at all 5 horizons), which is what licensed the personalization
claim; CGM-LSM needs its own, because its base is far less adapted and its
effect is ~30x larger.

ALL THREE ARMS ARE SCORED ON ONE COMMON BIN SET (common_scored_keys), not on
each pair's own intersection -- otherwise the three comparisons would sit on
three slightly different denominators and would not be internally consistent.

DECISION RULE, stated in advance so the outcome cannot be reinterpreted after
the fact:
  - donor gain ~= 0 or negative, own-history gain large  -> control PASSES,
    the effect is adaptation to the person.
  - donor gain comparable to own-history gain            -> control FAILS,
    report the effect as base-checkpoint underfitting, not personalization.
  - donor gain clearly positive but clearly smaller      -> BOTH mechanisms are
    present; the personalization-specific part is (own - donor), and that
    difference is what may be attributed to the person.

P-values are author-reference only and must not label anything, per the
project's standing reporting rule.
"""
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cgmlsm_matched_comparison import compare_arms, common_scored_keys  # noqa: E402

REPO_ROOT = "/path/to/aireadi-collab"
VAL_ROOT = os.environ.get(
    "CGMLSM_PERS_VAL_ROOT", os.path.join(REPO_ROOT, "cgmlsm_personalization_results", "val")
)
CONFIG = os.environ.get("CGMLSM_PERS_CONTROL_CONFIG", "lr2e-4_s50")

ZEROSHOT = os.path.join(VAL_ROOT, "zeroshot_ref", "zs_glucose_only")
OWN = os.path.join(VAL_ROOT, CONFIG, "p7d")
DONOR = os.path.join(VAL_ROOT, f"donor_control_{CONFIG}", "p7d_shuffled")
OUTPUT_PATH = os.path.join(VAL_ROOT, f"val_negative_control_{CONFIG}.csv")
HORIZONS = ("30min", "1h", "2h")


def main():
    for name, path in (("zero-shot", ZEROSHOT), ("own-history", OWN), ("donor", DONOR)):
        if not os.path.isdir(path):
            raise SystemExit(f"{name} arm missing at {path}")
    print(f"negative control at config {CONFIG}\n  zero-shot: {ZEROSHOT}\n  own: {OWN}\n"
          f"  donor: {DONOR}\n")

    rows = []
    for horizon in HORIZONS:
        keys = common_scored_keys([ZEROSHOT, OWN, DONOR], horizon)
        if not keys:
            print(f"{horizon}: no bins scored by all three arms, skipped")
            continue
        keys = pd.MultiIndex.from_tuples(sorted(keys))

        for label_b, arm_b in (("own_history", OWN), ("donor_history", DONOR)):
            res = compare_arms(ZEROSHOT, arm_b, horizon, restrict_keys=keys,
                               label_a="zeroshot", label_b=label_b)
            if res is not None:
                res["contrast"] = f"zeroshot_vs_{label_b}"
                res["config"] = CONFIG
                rows.append({k: v for k, v in res.items()})

        # The personalization-specific increment: own history vs a stranger's,
        # holding the fine-tuning recipe and the bin set fixed.
        res = compare_arms(DONOR, OWN, horizon, restrict_keys=keys,
                           label_a="donor_history", label_b="own_history")
        if res is not None:
            res["contrast"] = "donor_vs_own"
            res["config"] = CONFIG
            rows.append(res)

    if not rows:
        raise SystemExit("no comparisons produced")
    out = pd.DataFrame(rows)
    out.to_csv(OUTPUT_PATH, index=False)

    print("=== POSITIVE delta_mae = arm_b better than arm_a ===")
    for contrast in ("zeroshot_vs_own_history", "zeroshot_vs_donor_history", "donor_vs_own"):
        sub = out[out["contrast"] == contrast]
        if sub.empty:
            continue
        print(f"\n{contrast}:")
        cols = ["horizon", "n_participants", "n_bins_scored_matched", "delta_mae",
                "delta_mae_ci_lo", "delta_mae_ci_hi", "win_rate_b", "cluster_by_participant_p"]
        print(sub[cols].round(4).to_string(index=False))

    print("\n=== verdict against the pre-stated decision rule ===")
    for horizon in HORIZONS:
        own = out[(out["contrast"] == "zeroshot_vs_own_history") & (out["horizon"] == horizon)]
        don = out[(out["contrast"] == "zeroshot_vs_donor_history") & (out["horizon"] == horizon)]
        diff = out[(out["contrast"] == "donor_vs_own") & (out["horizon"] == horizon)]
        if own.empty or don.empty:
            continue
        o, d = float(own["delta_mae"].iloc[0]), float(don["delta_mae"].iloc[0])
        share = (d / o * 100.0) if o != 0 else float("nan")
        lo = float(diff["delta_mae_ci_lo"].iloc[0]) if not diff.empty else float("nan")
        hi = float(diff["delta_mae_ci_hi"].iloc[0]) if not diff.empty else float("nan")
        print(f"  {horizon:>5}: own {o:+.3f} | donor {d:+.3f} ({share:.0f}% of own) | "
              f"own-minus-donor CI [{lo:+.3f}, {hi:+.3f}]")
    print("\nRead these against the module docstring's decision rule. A donor gain that is a "
          "large share of the own-history gain means most of the effect is the base checkpoint "
          "being underfit to AI-READI, and must be reported that way.")
    print(f"\nwrote {OUTPUT_PATH}")
    print("CGMLSM_PERS_VAL_CONTROL_OK")


if __name__ == "__main__":
    main()
