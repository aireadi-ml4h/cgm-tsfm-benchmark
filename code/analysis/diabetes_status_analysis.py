"""Does diabetes severity/status (study_group) relate to model performance,
covariate benefit, or horizon-specific effects? Broader than the earlier
covariate-heterogeneity check, which only tested study_group against ONE
covariate arm/horizon/k combination (cov_arm1_5phys, 1h, k=7, p=0.71 ns).

Four checks:
  1. Baseline zero-shot difficulty (Chronos-2 + TimesFM, all 5 horizons) by
     study_group -- is forecasting objectively harder for more severe
     diabetes (plausible: more glucose volatility)?
  2. Per-model winner preference by study_group, reusing
     chronos2_zeroshot_results/per_participant_model_winner.csv (already
     computed by a separate, completed fork this session).
  3. Covariate benefit by study_group, swept across ALL 4 nested arms x
     2 fit lengths x 5 horizons x 2 k/bar families (bar7_k7, bar3_k3) --
     reusing the exact combination list already enumerated and verified by
     code/analysis/covariate_per_participant_significance_full_sweep.py, not
     re-deriving it.
  4. heart_rate LOO effect (established: significantly helpful at 3h for
     BOTH models, nowhere else) -- does it hold uniformly across study_group
     or concentrate in one group?

Diabetes subgroups are small (~45-50 each per earlier session checks) --
flagged explicitly wherever n is too small to trust a conclusion.
"""
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

REPO_ROOT = "/path/to/aireadi-collab"
sys.path.insert(0, os.path.join(REPO_ROOT, "code"))
# Sibling modules live in adjacent topic directories under code/.
_CODE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _d in ("analysis", "baselines", "cgmlsm", "chronos2", "meal_detection", "model_input", "ssmcgm", "timesfm"):
    sys.path.insert(0, os.path.join(_CODE, _d))
sys.path.insert(0, os.path.join(REPO_ROOT, "code/model_input"))
from build_participant_grid import load_participants_covariates  # noqa: E402
from chronos2_matched_comparison import load_window_keys  # noqa: E402

HORIZONS = ["30min", "1h", "2h", "3h", "4h"]
C2_DIR = f"{REPO_ROOT}/chronos2_zeroshot_results/full/imputed"
TFM_DIR = f"{REPO_ROOT}/timesfm_zeroshot_results/full/imputed"
OUT_DIR = f"{REPO_ROOT}/chronos2_zeroshot_results"

covs = load_participants_covariates()


def study_group_of(pid):
    return covs.get(str(pid), {}).get("study_group", None)


def per_participant_mae(path):
    df = pd.read_csv(path, dtype={"window_id": str})
    df = df.dropna(subset=["actual"])
    df["person_id"] = df["window_id"].str.rsplit("_", n=1).str[0]
    return df.groupby("person_id")["abs_error"].mean()


def group_test(df, value_col, group_col="study_group"):
    groups = [g[value_col].dropna().values for _, g in df.groupby(group_col) if len(g[value_col].dropna()) >= 2]
    labels = [k for k, g in df.groupby(group_col) if len(g[value_col].dropna()) >= 2]
    sizes = {k: len(g[value_col].dropna()) for k, g in df.groupby(group_col)}
    if len(groups) < 2:
        return dict(f=np.nan, p=np.nan, kw_h=np.nan, kw_p=np.nan, sizes=sizes, means={})
    f, p = stats.f_oneway(*groups)
    h, kw_p = stats.kruskal(*groups)
    means = {lab: float(np.mean(g)) for lab, g in zip(labels, groups)}
    return dict(f=float(f), p=float(p), kw_h=float(h), kw_p=float(kw_p), sizes=sizes, means=means)


print("=" * 70, file=sys.stderr)
print("CHECK 1: baseline zero-shot difficulty by study_group", file=sys.stderr)
print("=" * 70, file=sys.stderr)
rows1 = []
for model_name, base_dir in [("chronos2", C2_DIR), ("timesfm", TFM_DIR)]:
    for h in HORIZONS:
        path = f"{base_dir}/predictions_{h}_glucose_only.csv"
        if not os.path.exists(path):
            print(f"  MISSING: {path}", file=sys.stderr)
            continue
        mae = per_participant_mae(path)
        df = mae.rename("mae").reset_index().rename(columns={"index": "person_id"})
        df["study_group"] = df["person_id"].map(study_group_of)
        df = df.dropna(subset=["study_group"])
        res = group_test(df, "mae")
        print(f"{model_name:10s} {h:6s} n={len(df):4d} ANOVA F={res['f']:.3f} p={res['p']:.4f} "
              f"KW p={res['kw_p']:.4f} sizes={res['sizes']} means={ {k: round(v,2) for k,v in res['means'].items()} }",
              file=sys.stderr)
        rows1.append(dict(check="baseline_difficulty", model=model_name, horizon=h,
                           n=len(df), anova_f=res["f"], anova_p=res["p"], kw_p=res["kw_p"],
                           **{f"mean_{k}": v for k, v in res["means"].items()}))
pd.DataFrame(rows1).to_csv(f"{OUT_DIR}/diabetes_status_check1_baseline_difficulty.csv", index=False)

print("=" * 70, file=sys.stderr)
print("CHECK 2: per-model winner preference by study_group", file=sys.stderr)
print("=" * 70, file=sys.stderr)
winner_path = f"{REPO_ROOT}/chronos2_zeroshot_results/per_participant_model_winner.csv"
rows2 = []
if os.path.exists(winner_path):
    dfw = pd.read_csv(winner_path, dtype={"person_id": str})
    dfw = dfw.dropna(subset=["study_group"])
    res = group_test(dfw, "frac_horizons_c2_wins")
    print(f"frac_horizons_c2_wins by study_group: n={len(dfw)} ANOVA F={res['f']:.3f} p={res['p']:.4f} "
          f"KW p={res['kw_p']:.4f} sizes={res['sizes']} means={ {k: round(v,3) for k,v in res['means'].items()} }",
          file=sys.stderr)
    rows2.append(dict(check="model_winner_pref", n=len(dfw), anova_f=res["f"], anova_p=res["p"], kw_p=res["kw_p"],
                       **{f"mean_{k}": v for k, v in res["means"].items()}))
    # also: consistent_preference (bool) rate by group
    if "consistent_preference" in dfw.columns:
        ct = dfw.groupby("study_group")["consistent_preference"].mean()
        print(f"  consistent_preference rate by group: {ct.to_dict()}", file=sys.stderr)
else:
    print(f"  MISSING: {winner_path} -- other fork's output not found", file=sys.stderr)
pd.DataFrame(rows2).to_csv(f"{OUT_DIR}/diabetes_status_check2_model_winner.csv", index=False)

print("=" * 70, file=sys.stderr)
print("CHECK 3: covariate benefit by study_group, full sweep (4 arms x 2 fits x 5 horizons x 2 families)", file=sys.stderr)
print("=" * 70, file=sys.stderr)
ARMS = ["cov_arm1_5phys", "cov_arm2_5phys_meal", "cov_arm3_5phys_meal_oxysat", "cov_arm4_5phys_meal_sleep"]
FITS = ["fit6", "fit24"]
FAMILIES = [
    dict(name="bar7_k7", base_dir=f"{REPO_ROOT}/chronos2_personalization_results/test", glu_arm="p7d"),
    dict(name="bar3_k3", base_dir=f"{REPO_ROOT}/chronos2_personalization_results/test_bar3d", glu_arm="p3d"),
]


def per_participant_diffs_raw(glu_dir, glu_cond, cov_dir, cov_cond, horizon):
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
    d = (out["a"].loc[paired] - out["b"].loc[paired])
    person = pd.Series({k: key_to_person[k] for k in paired})
    return d.groupby(person).mean()


rows3 = []
n_tested, n_significant_meaningful = 0, 0
for fam in FAMILIES:
    for arm in ARMS:
        for fit in FITS:
            for h in HORIZONS:
                glu_dir = f"{fam['base_dir']}/{fam['glu_arm']}/{fit}"
                cov_dir = f"{fam['base_dir']}/{arm}/{fit}"
                d = per_participant_diffs_raw(glu_dir, "glucose_only", cov_dir, "with_covariates", h)
                if d is None or len(d) < 4:
                    continue
                df = d.rename("delta").reset_index().rename(columns={"index": "person_id"})
                df["study_group"] = df["person_id"].map(study_group_of)
                df = df.dropna(subset=["study_group"])
                if df["study_group"].nunique() < 2:
                    continue
                res = group_test(df, "delta")
                if np.isnan(res["p"]):
                    continue
                n_tested += 1
                sig = res["p"] < 0.05
                if sig:
                    n_significant_meaningful += 1
                rows3.append(dict(check="covariate_benefit", family=fam["name"], arm=arm, fit=fit, horizon=h,
                                   n=len(df), anova_f=res["f"], anova_p=res["p"], kw_p=res["kw_p"],
                                   sizes=str(res["sizes"])))
                if sig:
                    print(f"  ** SIGNIFICANT ** {fam['name']} {arm} {fit} {h}: F={res['f']:.3f} p={res['p']:.4f} "
                          f"means={ {k: round(v,3) for k,v in res['means'].items()} }", file=sys.stderr)
print(f"Check 3 total combinations tested: {n_tested}, significant at p<0.05: {n_significant_meaningful} "
      f"(chance expectation ~{0.05*n_tested:.1f})", file=sys.stderr)
pd.DataFrame(rows3).to_csv(f"{OUT_DIR}/diabetes_status_check3_covariate_benefit_sweep.csv", index=False)

print("=" * 70, file=sys.stderr)
print("CHECK 4: heart_rate LOO effect by study_group (established: helpful at 3h, both models)", file=sys.stderr)
print("=" * 70, file=sys.stderr)
rows4 = []
for model_name, glu_dir, drop_dir_root in [
    ("chronos2", C2_DIR, f"{REPO_ROOT}/chronos2_zeroshot_results/loo/drop_heart_rate"),
    ("timesfm", TFM_DIR, f"{REPO_ROOT}/timesfm_zeroshot_results/loo/drop_heart_rate"),
]:
    for h in HORIZONS:
        glu_path = f"{glu_dir}/predictions_{h}_with_covariates.csv"
        drop_path = f"{drop_dir_root}/predictions_{h}_with_covariates.csv"
        if not os.path.exists(glu_path) or not os.path.exists(drop_path):
            continue
        keys_a = load_window_keys(h, "with_covariates")
        pa = pd.read_csv(glu_path, dtype={"window_id": str}).dropna(subset=["actual"])
        pb = pd.read_csv(drop_path, dtype={"window_id": str}).dropna(subset=["actual"])
        pa["key"] = pa["window_id"].map(keys_a["key"].to_dict())
        pb["key"] = pb["window_id"].map(keys_a["key"].to_dict())
        pa["ab"] = (pa["predictions"] - pa["actual"]).abs()
        pb["ab"] = (pb["predictions"] - pb["actual"]).abs()
        ga = pa.groupby("key")["ab"].mean()
        gb = pb.groupby("key")["ab"].mean()
        common = ga.index.intersection(gb.index)
        if len(common) == 0:
            continue
        # positive delta (drop - full) = dropping heart_rate makes it WORSE = heart_rate helps
        delta = (gb.loc[common] - ga.loc[common])
        key_to_person = keys_a.reset_index().set_index("key")["person_id"].to_dict()
        person = pd.Series({k: key_to_person[k] for k in common})
        d = delta.groupby(person).mean()
        df = d.rename("delta").reset_index().rename(columns={"index": "person_id"})
        df["study_group"] = df["person_id"].map(study_group_of)
        df = df.dropna(subset=["study_group"])
        if df["study_group"].nunique() < 2:
            continue
        res = group_test(df, "delta")
        marker = " ** SIGNIFICANT **" if (not np.isnan(res["p"]) and res["p"] < 0.05) else ""
        print(f"{model_name:10s} {h:6s} n={len(df):4d} ANOVA F={res['f']:.3f} p={res['p']:.4f}{marker} "
              f"sizes={res['sizes']} means={ {k: round(v,3) for k,v in res['means'].items()} }", file=sys.stderr)
        rows4.append(dict(check="heart_rate_loo", model=model_name, horizon=h, n=len(df),
                           anova_f=res["f"], anova_p=res["p"], kw_p=res["kw_p"],
                           **{f"mean_{k}": v for k, v in res["means"].items()}))
pd.DataFrame(rows4).to_csv(f"{OUT_DIR}/diabetes_status_check4_heart_rate_by_group.csv", index=False)

print("DIABETES_STATUS_ANALYSIS_OK", file=sys.stderr)
