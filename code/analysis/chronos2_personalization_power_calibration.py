"""How large a covariate increment could phase 2 actually detect on 409
windows / 194 participants -- computed from this project's own data, not from
a rule of thumb.

THE QUESTION
============
Phase 2's four arms are compared on E_cov: 409 windows / 194 participants at
1h. CHRONOS2_PERSONALIZATION.md already argues on sample-size grounds that the
0.05-0.5 mg/dL covariate effects the zero-shot sweep resolved are not
recoverable there. That argument deserves a number, because the point of phase
2 is a pre-registered hypothesis that personalization makes covariates matter
MORE than they did zero-shot -- so what matters is not "is 0.07 mg/dL
detectable" (it is not) but "how big does the unlocked effect have to be
before this design can see it."

WHAT THIS COMPUTES
==================
compare_arms()'s primary test is a one-sample t-test on the PER-PARTICIPANT
mean loss differential (`cluster_by_participant_p`). Its minimum detectable
effect at 80% power, alpha=0.05 two-sided is

    MDE = (z_{1-a/2} + z_{power}) * SD(d_by_person) / sqrt(n_participants)
        ~ 2.802 * SD / sqrt(n)

SD(d_by_person) is not a quantity to guess: it is measurable now, on the exact
windows phase 2 will score, from zero-shot arms already on disk. Each of the
three increments phase 2 attributes has an exact zero-shot counterpart pair
(the same reason the mode=none plumbing check works --
slurm/chronos2_personalization_cov_smoke.sbatch):

    arm2 - arm1  isolates predmeal_prob
    arm3 - arm2  isolates oxygen_saturation
    arm4 - arm2  isolates sleep_stage

Reported in TWO units, because they answer different questions:
  MSE  the units the significance test is actually computed in
  MAE  the units every result in this project is quoted in, so the MDE can be
       read against the 0.05-0.5 mg/dL effects the zero-shot sweep resolved and
       against phase 1's own ~0.9-1.5 mg/dL personalization effect

The MAE-scale differential is computed the same way (per-window mean absolute
error, averaged per participant) rather than by converting the MSE number,
because the MSE->MAE map depends on the error distribution.

Also reported: the same calculation on the FULL test window set, so the power
lost to E_cov's 7-day bar plus covariate eligibility is a measured ratio
rather than an assertion.

CAVEAT ON WHAT THIS DOES AND DOES NOT BOUND
===========================================
SD(d_by_person) is measured on ZERO-SHOT arm pairs. If personalization changes
the spread of per-participant covariate effects -- which is exactly what the
"covariates only pay off once you personalize" hypothesis implies -- the
fine-tuned SD will differ, and probably be larger (fine-tuning is stochastic
and adds per-participant variance the zero-shot pair does not have). So these
MDEs are a LOWER BOUND on phase 2's minimum detectable effect: the real one is
at least this large. That direction is the useful one for deciding whether to
run.

This is a power calculation, not a result. It uses only zero-shot arms already
run and reported, and it selects nothing.
"""
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from chronos2_matched_comparison import compare_arms, load_window_keys  # noqa: E402

REPO_ROOT = "/path/to/aireadi-collab"
ZS = f"{REPO_ROOT}/chronos2_zeroshot_results"
ZSF = f"{REPO_ROOT}/chronos2_zeroshot_results_ffillfix"
# CHRONOS2_PERS_HISTORY_BAR_DAYS: measure the MDE on the PARALLEL family's
# window set instead of the primary one. Lowering the eligibility bar is the
# only lever that materially changes phase 2's power (more k values do not --
# every arm scores the identical window set under a fixed bar), so this is the
# knob worth calibrating before deciding whether that family is worth running.
_BAR = int(os.environ.get("CHRONOS2_PERS_HISTORY_BAR_DAYS", "7"))
_BAR_SUFFIX = "" if _BAR == 7 else f"_bar{_BAR}d"
EVAL_SET_COV = (f"{REPO_ROOT}/model_input/personalization/"
                f"eval_set_cov_test{_BAR_SUFFIX}.csv")
HORIZON_BINS = {"30min": 6, "1h": 12, "2h": 24, "3h": 36, "4h": 48}

# 80% power, alpha=0.05 two-sided: 1.95996 + 0.84162
Z_SUM = 2.80158

ARMS = {
    "arm1_5phys": dict(label="arm1", condition="with_covariates",
                       results_dir=f"{ZS}/loo/drop_oxygen_saturation"),
    "arm2_5phys_meal": dict(label="arm2", condition="with_covariates",
                            results_dir=f"{ZSF}/full/with_predmeal_prob_drop_oxysat"),
    "arm3_5phys_meal_oxysat": dict(label="arm3", condition="with_covariates",
                                    results_dir=f"{ZSF}/full/with_predmeal_prob"),
    "arm4_5phys_meal_sleep": dict(label="arm4", condition="with_covariates",
                                   results_dir=f"{ZSF}/full/with_sleep_stage_predmeal_drop_oxysat"),
}
INCREMENTS = [
    ("predmeal_prob", "arm1_5phys", "arm2_5phys_meal"),
    ("oxygen_saturation", "arm2_5phys_meal", "arm3_5phys_meal_oxysat"),
    ("sleep_stage", "arm2_5phys_meal", "arm4_5phys_meal_sleep"),
]


def per_participant_diffs(arm_a, arm_b, horizon, restrict_keys):
    """Per-participant mean loss differential (a - b) in MSE and MAE units, on
    the same matched/paired window set compare_arms() uses. The pairing is
    duplicated from compare_arms deliberately: that function returns only the
    aggregate statistics, and an MDE needs the per-participant spread."""
    keys_a = load_window_keys(horizon, arm_a["condition"])
    keys_b = load_window_keys(horizon, arm_b["condition"])
    common = set(keys_a["key"]) & set(keys_b["key"])
    if restrict_keys is not None:
        common &= set(restrict_keys)
    key_to_person = keys_a.reset_index().set_index("key")["person_id"].to_dict()

    out = {}
    for name, arm, keys in (("a", arm_a, keys_a), ("b", arm_b, keys_b)):
        p = pd.read_csv(f"{arm['results_dir']}/predictions_{horizon}_{arm['condition']}.csv",
                         dtype={"window_id": str})
        p["key"] = p["window_id"].map(keys["key"].to_dict())
        p = p[p["key"].isin(common)].dropna(subset=["actual"]).copy()
        p["sq"] = (p["predictions"] - p["actual"]) ** 2
        p["ab"] = (p["predictions"] - p["actual"]).abs()
        out[name] = p.groupby("key")[["sq", "ab"]].mean()

    paired = out["a"].index.intersection(out["b"].index)
    d = out["a"].loc[paired] - out["b"].loc[paired]
    person = pd.Series({k: key_to_person[k] for k in paired})
    return d.groupby(person).mean(), len(paired)


def mde(series):
    n = len(series)
    sd = float(series.std(ddof=1))
    return Z_SUM * sd / np.sqrt(n), sd, n


def report(restrict_keys, label):
    print(f"\n{'=' * 78}\n{label}\n{'=' * 78}")
    rows = []
    for channel, a_name, b_name in INCREMENTS:
        arm_a, arm_b = ARMS[a_name], ARMS[b_name]
        for horizon, hbins in HORIZON_BINS.items():
            stat = compare_arms(horizon, hbins, arm_a, arm_b, restrict_keys=restrict_keys)
            d_person, n_paired = per_participant_diffs(arm_a, arm_b, horizon, restrict_keys)
            mde_mse, sd_mse, n_p = mde(d_person["sq"])
            mde_mae, sd_mae, _ = mde(d_person["ab"])
            _, p_mae = stats.ttest_1samp(d_person["ab"].to_numpy(), 0.0)
            rows.append({
                "channel": channel, "horizon": horizon,
                "n_windows": n_paired, "n_participants": n_p,
                # sign convention matches compare_arms: differential is a - b,
                # and b is the arm WITH the extra channel, so POSITIVE means
                # the channel helped zero-shot
                "zs_dMAE": float(d_person["ab"].mean()), "zs_dMAE_cluster_p": p_mae,
                "zs_dMSE_cluster_p": stat["cluster_by_participant_p"],
                "sd_person_mae": sd_mae, "mde_dMAE_80pct": mde_mae,
                "sd_person_mse": sd_mse, "mde_dMSE_80pct": mde_mse,
            })
    df = pd.DataFrame(rows)
    pd.set_option("display.width", 220)
    print(df.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    return df


def main():
    e = pd.read_csv(EVAL_SET_COV, dtype={"person_id": str})
    cov_keys = set(e["key"])

    full = report(None, "FULL test window set (what the zero-shot sweep was measured on)")
    sub = report(cov_keys, f"E_cov: phase-2 window set ({_BAR}-day bar + covariate eligibility)")

    print(f"\n{'=' * 78}\nPOWER LOST BY MOVING TO E_cov\n{'=' * 78}")
    m = full.merge(sub, on=["channel", "horizon"], suffixes=("_full", "_ecov"))
    m["mde_ratio"] = m["mde_dMAE_80pct_ecov"] / m["mde_dMAE_80pct_full"]
    print(m[["channel", "horizon", "n_participants_full", "n_participants_ecov",
             "mde_dMAE_80pct_full", "mde_dMAE_80pct_ecov", "mde_ratio"]]
          .to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    out = f"{REPO_ROOT}/chronos2_personalization_results/power_calibration_cov.csv"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    m.to_csv(out, index=False)
    print(f"\nwritten: {out}")
    print("\nRead mde_dMAE_80pct_ecov as: the smallest |dMAE| phase 2's primary\n"
          "(cluster-by-participant) test would detect 80% of the time on that\n"
          "horizon's E_cov population -- and, per the module docstring, a LOWER\n"
          "BOUND on it. An effect smaller than that produces a null that means\n"
          "nothing; report such a result as underpowered, not as 'covariates do\n"
          "not help once you personalize'.")


if __name__ == "__main__":
    main()
