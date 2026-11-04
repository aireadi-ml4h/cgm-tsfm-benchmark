"""Per-participant significance test of the covariate-personalization delta
(NOT the population-level aggregate test already done elsewhere). For each
of the 194 participants in the personalized-condition covariate comparison
(1h horizon, p7d/fit24 glucose_only vs cov_arm1_5phys/fit24 with_covariates,
same arms as investigate_covariate_heterogeneity.py's already-reported
result), tests whether THAT PERSON's own covariate delta is distinguishable
from zero using only their own window-level variance -- a genuinely
different question from every prior check this session (population mean,
cross-condition stability, clustering), directly testing whether the
population-level null is masking real per-person effects that are simply
underpowered individually (user's hypothesis) vs. genuine noise with no
real individual effects at all (prior working conclusion).

Reuses chronos2_personalization_power_calibration.per_participant_diffs'
matching logic (key-based, not a naive window_id join) but stops BEFORE its
final person-level mean, to keep the per-window diffs needed for a
per-participant t-test.
"""
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

REPO_ROOT = os.environ["AIREADI_ROOT"]
sys.path.insert(0, os.path.join(REPO_ROOT, "code"))
sys.path.insert(0, os.path.join(REPO_ROOT, "code/model_input"))
from chronos2_matched_comparison import load_window_keys  # noqa: E402

PERS = f"{REPO_ROOT}/chronos2_personalization_results/test"
PERS_GLU = dict(condition="glucose_only", results_dir=f"{PERS}/p7d/fit24")
PERS_COV = dict(condition="with_covariates", results_dir=f"{PERS}/cov_arm1_5phys/fit24")
HORIZON = "1h"


def per_window_diffs(arm_a, arm_b, horizon):
    """Same matching as per_participant_diffs, but returns one row PER
    WINDOW (key), not aggregated to one row per person -- the per-window
    'ab' (abs error) diffs are what a per-participant t-test needs."""
    keys_a = load_window_keys(horizon, arm_a["condition"])
    keys_b = load_window_keys(horizon, arm_b["condition"])
    common = set(keys_a["key"]) & set(keys_b["key"])
    key_to_person = keys_a.reset_index().set_index("key")["person_id"].to_dict()

    out = {}
    for name, arm, keys in (("a", arm_a, keys_a), ("b", arm_b, keys_b)):
        p = pd.read_csv(f"{arm['results_dir']}/predictions_{horizon}_{arm['condition']}.csv",
                         dtype={"window_id": str})
        p["key"] = p["window_id"].map(keys["key"].to_dict())
        p = p[p["key"].isin(common)].dropna(subset=["actual"]).copy()
        p["ab"] = (p["predictions"] - p["actual"]).abs()
        out[name] = p.groupby("key")["ab"].mean()  # one row per window (mean over horizon bins)

    paired = out["a"].index.intersection(out["b"].index)
    d = (out["a"].loc[paired] - out["b"].loc[paired]).rename("ab_diff").to_frame()
    d["person_id"] = [key_to_person[k] for k in d.index]
    return d


def main():
    d = per_window_diffs(PERS_GLU, PERS_COV, HORIZON)
    print(f"total matched windows: {len(d)}, participants: {d['person_id'].nunique()}", file=sys.stderr)

    rows = []
    for pid, g in d.groupby("person_id"):
        vals = g["ab_diff"].to_numpy()
        n = len(vals)
        if n < 2:
            rows.append({"person_id": pid, "n_windows": n, "mean_diff": vals.mean() if n else np.nan,
                         "t_stat": np.nan, "p_value": np.nan, "testable": False})
            continue
        t, p = stats.ttest_1samp(vals, 0.0)
        rows.append({"person_id": pid, "n_windows": n, "mean_diff": float(vals.mean()),
                     "t_stat": float(t), "p_value": float(p), "testable": True})

    res = pd.DataFrame(rows)
    out_path = f"{REPO_ROOT}/chronos2_personalization_results/covariate_per_participant_significance.csv"
    res.to_csv(out_path, index=False)

    testable = res[res["testable"]]
    n_total = len(res)
    n_testable = len(testable)
    print(f"\n=== window-count distribution (n={n_total} participants) ===", file=sys.stderr)
    print(res["n_windows"].describe().to_string(), file=sys.stderr)
    print(f"\nparticipants with n_windows < 2 (not testable): {n_total - n_testable} "
          f"({100*(n_total - n_testable)/n_total:.1f}%)", file=sys.stderr)
    print(f"participants testable (n_windows >= 2): {n_testable} ({100*n_testable/n_total:.1f}%)", file=sys.stderr)

    if n_testable:
        p05 = (testable["p_value"] < 0.05).sum()
        p10 = (testable["p_value"] < 0.10).sum()
        print(f"\nof {n_testable} testable participants:", file=sys.stderr)
        print(f"  individually significant at p<0.05: {p05} ({100*p05/n_testable:.1f}%)", file=sys.stderr)
        print(f"  individually significant at p<0.10: {p10} ({100*p10/n_testable:.1f}%)", file=sys.stderr)

        chance_05 = 0.05 * n_testable
        print(f"\nchance expectation under true null (5% of {n_testable} testable): {chance_05:.1f} participants", file=sys.stderr)
        print(f"observed p<0.05 count: {p05} vs. chance-expected: {chance_05:.1f} "
              f"(ratio {p05/chance_05:.2f}x)" if chance_05 > 0 else "", file=sys.stderr)

        high_n = testable[testable["n_windows"] >= 5]
        low_n = testable[testable["n_windows"] < 5]
        print(f"\n--- stratified by window count ---", file=sys.stderr)
        if len(high_n):
            hp05 = (high_n["p_value"] < 0.05).sum()
            print(f"n_windows>=5 (n={len(high_n)} participants): {hp05} significant at p<0.05 "
                  f"({100*hp05/len(high_n):.1f}%)", file=sys.stderr)
        if len(low_n):
            lp05 = (low_n["p_value"] < 0.05).sum()
            print(f"n_windows<5 (n={len(low_n)} participants): {lp05} significant at p<0.05 "
                  f"({100*lp05/len(low_n):.1f}%)", file=sys.stderr)

    print(f"\nwritten: {out_path}", file=sys.stderr)
    print("COVARIATE_PER_PARTICIPANT_SIGNIFICANCE_OK", file=sys.stderr)


if __name__ == "__main__":
    main()
