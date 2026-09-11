"""THE CGM-LSM PERSONALIZATION TEST-SPLIT REPORT -- the paper output for this arm.

Produces two files in the test results root:
  test_arm_scores.csv -- MAE/RMSE per arm per horizon on ONE common bin set.
                         This is the table that goes in the paper.
  test_report.csv     -- every matched contrast (each arm vs zero-shot, and
                         each own-history arm vs its own donor control) with
                         cluster-by-participant CIs.

REPORTING RULES, inherited from code/CHRONOS2_PERSONALIZATION.md and enforced
here rather than left to the writer:

  1. Report what TSFM papers report: MAE/RMSE per horizon. That is the output.
  2. THERE IS NO VERDICT COLUMN AND DO NOT ADD ONE. Two attempts on the
     Chronos-2 arm both failed -- "helped/hurt on p<0.05", then "|delta| vs
     MDE", which is arithmetically identical to p<0.005 and suppressed 17 of 36
     significant rows including the headline. Report the estimate and its CI
     uncompressed and let the reader judge.
  3. For a null, give the bound ("rules out effects larger than X mg/dL"),
     never "personalization does not help".
  4. p-values (cluster-by-participant) are for the author's reference only and
     must not label anything.
  5. Compare arms ONLY on matched keys. Never diff summary.csv -- arms have
     different window coverage, and CGM-LSM can additionally decline to
     forecast a bin, so two arms' summary MAEs can be averages over different
     bin sets entirely.

ONE COMMON BIN SET ACROSS ALL ARMS. Every score in test_arm_scores.csv is
computed on the bins where EVERY arm produced a numeric prediction and ground
truth exists. That is stricter than pairwise matching and makes the whole table
internally consistent -- any two rows can be compared directly. The cost is
that a participant skipped in one arm (no clean stretch of history to fit on)
drops out of all of them; n_participants is reported so that is visible.

HORIZONS: 30min/1h/2h. There is no 3h/4h block for CGM-LSM -- the architecture
cannot generate that far (n_positions=168 caps it at 24 bins past the context),
which is a hard limit, not a scope decision.

THE DONOR CONTROL MUST NOT BE OVERSOLD. See
slurm/cgmlsm_personalization_config.sh caveat 1: on val, the donor effect swung
+1.69 / -0.07 / -1.15 across three k=7d configs that were statistically
indistinguishable on the total gain. So the own-minus-donor split is noisy even
where the total is stable. This script prints the split, and prints that
warning next to it, because the frozen config happens to be one where the donor
effect lands near zero -- which would otherwise read as a clean control pass.
Unlike Chronos-2, where a donor's history was worse than zero-shot at all five
horizons, CGM-LSM has no such clean separation.
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cgmlsm_matched_comparison import compare_arms, common_scored_keys, load_arm  # noqa: E402

REPO_ROOT = "/path/to/aireadi-collab"
BASE_TAG = os.environ.get("CGMLSM_PERS_BASE_TAG", "finetune_sweep4_ep2_lr1e-2")
TEST_ROOT = os.environ.get(
    "CGMLSM_PERS_TEST_ROOT",
    os.path.join(REPO_ROOT, "cgmlsm_personalization_results", f"test_{BASE_TAG}"),
)
HORIZONS = ("30min", "1h", "2h")
ZS = "zs_glucose_only"
OWN_ARMS = ("p1d", "p3d", "p7d")
DONOR_OF = {"p1d": "p1d_shuffled", "p3d": "p3d_shuffled", "p7d": "p7d_shuffled"}
ARM_ORDER = [ZS, "p1d", "p3d", "p7d", "p1d_shuffled", "p3d_shuffled", "p7d_shuffled"]


def arm_dir(arm):
    return os.path.join(TEST_ROOT, arm)


def main():
    present = [a for a in ARM_ORDER if os.path.isdir(arm_dir(a))]
    missing = [a for a in ARM_ORDER if a not in present]
    if ZS not in present:
        raise SystemExit(f"zero-shot arm missing under {TEST_ROOT}; nothing to compare against")
    if missing:
        print(f"WARNING: arms not present, excluded from the common bin set: {missing}")
    print(f"reporting {len(present)} arms from {TEST_ROOT}\n")

    score_rows, contrast_rows = [], []
    for horizon in HORIZONS:
        keys = common_scored_keys([arm_dir(a) for a in present], horizon)
        if not keys:
            print(f"{horizon}: no bins scored by every arm, skipped")
            continue
        keys_idx = pd.MultiIndex.from_tuples(sorted(keys))

        # --- the paper table: every arm on the one common bin set ---
        for arm in present:
            df = load_arm(arm_dir(arm), horizon)
            sel = df[pd.MultiIndex.from_frame(df[["person_id", "window_id", "bin_start"]])
                     .isin(keys)]
            sel = sel[sel["is_numeric"] & sel["has_truth"]]
            score_rows.append({
                "arm": arm, "horizon": horizon,
                "mae_mgdl": round(float(sel["abs_error"].mean()), 4),
                "rmse_mgdl": round(float(np.sqrt((sel["abs_error"] ** 2).mean())), 4),
                "n_bins": len(sel),
                "n_windows": sel["window_id"].nunique(),
                "n_participants": sel["person_id"].nunique(),
                "is_negative_control": arm.endswith("_shuffled"),
            })

        # --- contrasts, all on the same bin set ---
        for arm in present:
            if arm == ZS:
                continue
            r = compare_arms(arm_dir(ZS), arm_dir(arm), horizon, restrict_keys=keys_idx,
                             label_a="zeroshot", label_b="arm")
            if r:
                r.update({"contrast": f"{arm}_vs_zeroshot", "horizon": horizon, "arm": arm})
                contrast_rows.append(r)
        for own, donor in DONOR_OF.items():
            if own in present and donor in present:
                r = compare_arms(arm_dir(donor), arm_dir(own), horizon, restrict_keys=keys_idx,
                                 label_a="donor", label_b="own")
                if r:
                    r.update({"contrast": f"{own}_vs_{donor}", "horizon": horizon, "arm": own})
                    contrast_rows.append(r)

    if not score_rows:
        raise SystemExit("no scores produced")
    scores = pd.DataFrame(score_rows)
    contrasts = pd.DataFrame(contrast_rows)
    scores.to_csv(os.path.join(TEST_ROOT, "test_arm_scores.csv"), index=False)
    contrasts.to_csv(os.path.join(TEST_ROOT, "test_report.csv"), index=False)

    print("=== TEST: MAE by arm and horizon (one common bin set; no verdict column, by design) ===")
    print(scores.pivot(index="arm", columns="horizon", values="mae_mgdl")
                .reindex([a for a in ARM_ORDER if a in set(scores["arm"])])
                .reindex(columns=list(HORIZONS)).round(3).to_string())
    print("\n=== TEST: RMSE by arm and horizon ===")
    print(scores.pivot(index="arm", columns="horizon", values="rmse_mgdl")
                .reindex([a for a in ARM_ORDER if a in set(scores["arm"])])
                .reindex(columns=list(HORIZONS)).round(3).to_string())
    cov = scores[scores["arm"] == ZS][["horizon", "n_bins", "n_windows", "n_participants"]]
    print("\ncoverage of the common bin set (identical for every arm by construction):")
    print(cov.to_string(index=False))

    print("\n=== TEST: personalization vs zero-shot (POSITIVE delta_mae = personalization better) ===")
    # delta_mae_clustered_mean is quoted WITH its own CI and p -- the pooled
    # delta_mae is shown alongside but its CI is not the clustered one.
    cols = ["horizon", "arm", "delta_mae", "delta_mae_clustered_mean",
            "delta_mae_ci_lo", "delta_mae_ci_hi", "win_rate_b", "n_participants",
            "delta_mae_cluster_p"]
    zs_c = contrasts[contrasts["contrast"].str.endswith("_vs_zeroshot")]
    print(zs_c[cols].round(4).to_string(index=False))

    print("\n=== TEST: own history vs donor history, the person-specific part ===")
    own_c = contrasts[~contrasts["contrast"].str.endswith("_vs_zeroshot")]
    if not own_c.empty:
        print(own_c[cols].round(4).to_string(index=False))
    print("\nREAD THIS WITH THE TABLE ABOVE: on val, the donor effect swung "
          "+1.69/-0.07/-1.15 across three k=7d configs that were indistinguishable on "
          "the total gain, so this split is NOISY even where the total is stable. Do not "
          "report a near-zero donor effect as a clean control pass; that is what the "
          "Chronos-2 arm could claim and this one cannot.")

    # Denominator watchdog: any arm that declined to forecast more often than
    # the base would otherwise quietly shrink the common bin set.
    print("\n=== non-numeric (declined-to-forecast) predictions per arm ===")
    for horizon in HORIZONS:
        bits = []
        for arm in present:
            df = load_arm(arm_dir(arm), horizon)
            if df is None:
                continue
            n = int((~df["is_numeric"]).sum())
            if n:
                bits.append(f"{arm}={n}")
        print(f"  {horizon}: {', '.join(bits) if bits else 'none in any arm'}")

    print(f"\nwrote {os.path.join(TEST_ROOT, 'test_arm_scores.csv')}")
    print(f"wrote {os.path.join(TEST_ROOT, 'test_report.csv')}")
    print("CGMLSM_PERS_TEST_REPORT_OK")


if __name__ == "__main__":
    main()
