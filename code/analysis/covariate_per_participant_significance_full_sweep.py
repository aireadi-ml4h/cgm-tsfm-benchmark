"""Full sweep of the per-participant covariate-significance test across
EVERY real (arm, horizon, k/family) combination on disk -- generalizes
covariate_per_participant_significance.py (which only covered
cov_arm1_5phys, 1h, k=7/fit24) after it was informally treated as a general
finding when it was actually one narrow slice.

Real combinations found on disk (verified by directory listing, not
assumed):
- 4 nested arms (cov_arm1_5phys, cov_arm2_5phys_meal, cov_arm3_5phys_meal_oxysat,
  cov_arm4_5phys_meal_sleep), each vs. its own glucose-only twin.
- 2 fit lengths (fit6, fit24) per arm.
- 5 horizons (30min/1h/2h/3h/4h) per arm/fit.
- 2 k-value FAMILIES, not 3: covariate arms only exist at k=7
  (chronos2_personalization_results/test/, glucose-only twin is p7d) and at
  k=3 in a separate, larger, purpose-built family
  (chronos2_personalization_results/test_bar3d/, glucose-only twin is p3d) --
  built because bar=7 was underpowered (CHRONOS2_PERSONALIZATION.md). There
  is NO k=1 covariate arm anywhere -- confirmed absent, not an oversight in
  this sweep.

SSM-CGM: confirmed by directory check (ssmcgm_train_results/,
ssmcgm_personalization_results/) to have NO per-participant covariate
comparison data at all -- those directories contain trained checkpoints and
val-only scoring, not a completed with-vs-without-covariates per-window
prediction pair the way Chronos-2's arms have. Its test-set eval is
deliberately held. This is a real, confirmed gap, not tested here.
"""
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

REPO_ROOT = "/path/to/aireadi-collab"
sys.path.insert(0, os.path.join(REPO_ROOT, "code"))
sys.path.insert(0, os.path.join(REPO_ROOT, "code/model_input"))
from chronos2_matched_comparison import load_window_keys  # noqa: E402

HORIZONS = ["30min", "1h", "2h", "3h", "4h"]
FITS = ["fit6", "fit24"]
ARMS = ["cov_arm1_5phys", "cov_arm2_5phys_meal", "cov_arm3_5phys_meal_oxysat", "cov_arm4_5phys_meal_sleep"]
FAMILIES = [
    dict(name="bar7_k7", base_dir=f"{REPO_ROOT}/chronos2_personalization_results/test", glu_arm="p7d"),
    dict(name="bar3_k3", base_dir=f"{REPO_ROOT}/chronos2_personalization_results/test_bar3d", glu_arm="p3d"),
]


def per_window_diffs(glu_dir, glu_cond, cov_dir, cov_cond, horizon):
    keys_a = load_window_keys(horizon, glu_cond)
    keys_b = load_window_keys(horizon, cov_cond)
    common = set(keys_a["key"]) & set(keys_b["key"])
    key_to_person = keys_a.reset_index().set_index("key")["person_id"].to_dict()

    out = {}
    for name, path, cond, keys in (("a", glu_dir, glu_cond, keys_a), ("b", cov_dir, cov_cond, keys_b)):
        fpath = f"{path}/predictions_{horizon}_{cond}.csv"
        if not os.path.exists(fpath):
            return None
        p = pd.read_csv(fpath, dtype={"window_id": str})
        p["key"] = p["window_id"].map(keys["key"].to_dict())
        p = p[p["key"].isin(common)].dropna(subset=["actual"]).copy()
        if p.empty:
            return None
        p["ab"] = (p["predictions"] - p["actual"]).abs()
        out[name] = p.groupby("key")["ab"].mean()

    paired = out["a"].index.intersection(out["b"].index)
    if len(paired) == 0:
        return None
    d = (out["a"].loc[paired] - out["b"].loc[paired]).rename("ab_diff").to_frame()
    d["person_id"] = [key_to_person[k] for k in d.index]
    return d


def test_one_combo(glu_dir, glu_cond, cov_dir, cov_cond, horizon):
    d = per_window_diffs(glu_dir, glu_cond, cov_dir, cov_cond, horizon)
    if d is None:
        return None

    rows = []
    for pid, g in d.groupby("person_id"):
        vals = g["ab_diff"].to_numpy()
        n = len(vals)
        if n < 2:
            continue
        t, p = stats.ttest_1samp(vals, 0.0)
        rows.append({"person_id": pid, "n_windows": n, "p_value": float(p)})
    if not rows:
        return dict(n_participants_total=d["person_id"].nunique(), n_testable=0,
                     n_significant_p05=0, n_significant_p10=0)

    res = pd.DataFrame(rows)
    n_testable = len(res)
    p05 = int((res["p_value"] < 0.05).sum())
    p10 = int((res["p_value"] < 0.10).sum())
    return dict(
        n_participants_total=d["person_id"].nunique(),
        n_testable=n_testable,
        n_significant_p05=p05,
        n_significant_p10=p10,
        chance_expected_p05=round(0.05 * n_testable, 2),
        chance_expected_p10=round(0.10 * n_testable, 2),
        fraction_significant_p05=round(p05 / n_testable, 4),
        fraction_significant_p10=round(p10 / n_testable, 4),
        ratio_vs_chance_p05=round(p05 / (0.05 * n_testable), 3) if n_testable > 0 else np.nan,
    )


def main():
    all_rows = []
    skipped = []

    for fam in FAMILIES:
        for arm in ARMS:
            for fit in FITS:
                glu_dir = f"{fam['base_dir']}/{fam['glu_arm']}/{fit}"
                cov_dir = f"{fam['base_dir']}/{arm}/{fit}"
                if not os.path.isdir(glu_dir) or not os.path.isdir(cov_dir):
                    skipped.append(dict(family=fam["name"], arm=arm, fit=fit, horizon="ALL",
                                         reason="glu_dir or cov_dir missing"))
                    continue
                for horizon in HORIZONS:
                    result = test_one_combo(glu_dir, "glucose_only", cov_dir, "with_covariates", horizon)
                    if result is None:
                        skipped.append(dict(family=fam["name"], arm=arm, fit=fit, horizon=horizon,
                                             reason="no matched prediction data"))
                        continue
                    row = dict(family=fam["name"], arm=arm, fit=fit, horizon=horizon)
                    row.update(result)
                    all_rows.append(row)
                    print(f"{fam['name']} {arm} {fit} {horizon}: "
                          f"n_testable={result['n_testable']}, "
                          f"p05={result.get('n_significant_p05')}/{result.get('chance_expected_p05')} chance",
                          file=sys.stderr)

    out = pd.DataFrame(all_rows)
    out_path = f"{REPO_ROOT}/chronos2_personalization_results/covariate_per_participant_significance_FULL.csv"
    out.to_csv(out_path, index=False)
    print(f"\nwrote {len(out)} combinations to {out_path}", file=sys.stderr)

    if skipped:
        skip_df = pd.DataFrame(skipped)
        skip_path = f"{REPO_ROOT}/chronos2_personalization_results/covariate_per_participant_significance_SKIPPED.csv"
        skip_df.to_csv(skip_path, index=False)
        print(f"skipped {len(skipped)} combinations (no data) -> {skip_path}", file=sys.stderr)

    if len(out):
        flagged = out[out["ratio_vs_chance_p05"] > 2.0]
        print(f"\n=== combinations with >2x chance excess at p<0.05 (n={len(flagged)}) ===", file=sys.stderr)
        if len(flagged):
            print(flagged.to_string(index=False), file=sys.stderr)
        else:
            print("NONE -- no combination exceeded 2x chance expectation.", file=sys.stderr)

        print(f"\n=== overall summary across all {len(out)} combinations ===", file=sys.stderr)
        print(f"mean ratio_vs_chance_p05: {out['ratio_vs_chance_p05'].mean():.3f}", file=sys.stderr)
        print(f"max ratio_vs_chance_p05: {out['ratio_vs_chance_p05'].max():.3f}", file=sys.stderr)
        print(f"min ratio_vs_chance_p05: {out['ratio_vs_chance_p05'].min():.3f}", file=sys.stderr)

    print("COVARIATE_FULL_SWEEP_OK", file=sys.stderr)


if __name__ == "__main__":
    main()
