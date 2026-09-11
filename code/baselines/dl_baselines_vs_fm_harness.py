"""LSTM / Mamba / LightGBM / MLP vs the foundation models, through the
PROJECT'S OWN evaluation harness.

WHY THIS SUPERSEDES code/baselines/dl_baselines_vs_foundation_models.py. That script
hand-rolled the comparison: it paired on per-participant mean ABSOLUTE error
and ran a single t-test. The harness every other comparison in this project
uses (`chronos2_matched_comparison.compare_arms`) does three things
differently, and all three matter:

  1. It pairs on per-window MEAN SQUARED error, not per-participant MAE. A
     different loss and a different aggregation granularity, so the p-values
     are not interchangeable.
  2. It runs THREE tests, not one: naive paired t, Diebold-Mariano with the
     Harvey-Leybourne-Newbold small-sample correction, and
     cluster-by-participant. Cluster is the primary test, but reporting only
     cluster while every other table in the project reports three is an
     inconsistency a reviewer would notice.
  3. It reports matched RMSE alongside matched MAE. The hand-rolled version
     reported MAE only, which for this project systematically understates
     differences -- the personalization results show RMSE effects roughly
     twice the MAE effects.

`compare_arms` itself is not directly reusable here: it resolves prediction
files as `{results_dir}/predictions_{horizon}_{condition}.csv` and derives
window keys from the zero-shot eval directory, whereas the DL baselines write
`predictions_{horizon}.csv` keyed on (window_id, bin_start) with person_id
inline. So this script reuses the harness's STATISTICS verbatim -- imports
`per_window_mse` and `dm_hln_test` rather than reimplementing them -- while
supplying its own key mapping.

PROBABILISTIC METRICS ARE NOT COMPUTED HERE, and cannot be from the stored
artifacts. Only Chronos-2 writes quantiles (columns 0.1-0.9). TimesFM's
stored predictions are point-only, and LSTM / Mamba / LightGBM / MLP are
point forecasters trained on squared or absolute error, so they have no
predictive distribution to score. A CRPS / pinball / interval-coverage
comparison across these arms would require re-running inference for TimesFM
with quantile output and retraining the DL baselines with a quantile loss.
What IS available from stored files is a Chronos-2-only probabilistic
summary; see code/chronos2_probabilistic_metrics.py.
"""
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

REPO = "/path/to/aireadi-collab"
sys.path.insert(0, os.path.join(REPO, "code"))
# Sibling modules live in adjacent topic directories under code/.
_CODE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _d in ("analysis", "baselines", "cgmlsm", "chronos2", "meal_detection", "model_input", "ssmcgm", "timesfm"):
    sys.path.insert(0, os.path.join(_CODE, _d))
from chronos2_matched_comparison import dm_hln_test, per_window_mse  # noqa: E402

OUT_DIR = f"{REPO}/dl_vs_fm_results"
os.makedirs(OUT_DIR, exist_ok=True)

HORIZON_BINS = {"30min": 6, "1h": 12, "2h": 24}

DL = {
    "lstm": (f"{REPO}/lstm_zeroshot_results/predictions_{{h}}.csv", "predictions"),
    "mamba": (f"{REPO}/mamba_zeroshot_results/predictions_{{h}}.csv", "predictions"),
    "lightgbm": (f"{REPO}/lightgbm_zeroshot_results/predictions_{{h}}.csv", "predictions"),
    "mlp": (f"{REPO}/extra_baselines_results/predictions_{{h}}.csv", "mlp"),
    "circadian": (f"{REPO}/extra_baselines_results/predictions_{{h}}.csv", "circadian"),
}
FM = {
    "timesfm": f"{REPO}/timesfm_zeroshot_results/full/imputed/predictions_{{h}}_glucose_only.csv",
    "chronos2": f"{REPO}/chronos2_zeroshot_results/full/imputed/predictions_{{h}}_glucose_only.csv",
}


def load(path, cols):
    d = pd.read_csv(path, dtype={"window_id": str},
                    usecols=lambda c: c in cols)
    # tz trap: DL files are tz-aware, FM files tz-naive.
    d["bin_start"] = pd.to_datetime(d["bin_start"], utc=True)
    return d


def main():
    rows = []
    for h, hbins in HORIZON_BINS.items():
        for dl_name, (tmpl, pcol) in DL.items():
            dl_path = tmpl.format(h=h)
            if not os.path.exists(dl_path):
                continue
            dl = load(dl_path, {"window_id", "person_id", "bin_start", "actual", pcol})
            if pcol not in dl.columns:
                continue
            dl = dl.rename(columns={pcol: "predictions"}).dropna(subset=["predictions"])

            for fm_name, fm_tmpl in FM.items():
                fm_path = fm_tmpl.format(h=h)
                if not os.path.exists(fm_path):
                    continue
                fm = load(fm_path, {"window_id", "bin_start", "actual", "predictions"})

                # Common (window_id, bin_start) rows, the analogue of the
                # harness's common-key restriction.
                keys = dl[["window_id", "bin_start"]].merge(
                    fm[["window_id", "bin_start"]], on=["window_id", "bin_start"])
                if keys.empty:
                    print(f"  {h} {dl_name} vs {fm_name}: EMPTY overlap", flush=True)
                    continue
                a = dl.merge(keys, on=["window_id", "bin_start"])
                b = fm.merge(keys, on=["window_id", "bin_start"])

                # Ground truth must agree; otherwise the merge is misaligned.
                chk = a[["window_id", "bin_start", "actual"]].merge(
                    b[["window_id", "bin_start", "actual"]],
                    on=["window_id", "bin_start"], suffixes=("_a", "_b")).dropna()
                if len(chk) and float((chk["actual_a"] - chk["actual_b"]).abs().max()) > 1e-6:
                    print(f"  {h} {dl_name} vs {fm_name}: ACTUALS DISAGREE", flush=True)
                    continue

                sa, sb = a.dropna(subset=["actual"]), b.dropna(subset=["actual"])
                sa = sa.assign(abs_error=(sa["predictions"] - sa["actual"]).abs())
                sb = sb.assign(abs_error=(sb["predictions"] - sb["actual"]).abs())

                # Harvest the harness's own per-window MSE + three tests.
                # window_to_key is identity here: window_id already IS the key.
                wk = {w: w for w in set(a["window_id"])}
                loss_a = per_window_mse(a, wk)
                loss_b = per_window_mse(b, wk)
                paired = loss_a.index.intersection(loss_b.index)
                d = loss_a.loc[paired] - loss_b.loc[paired]

                naive_t, naive_p = stats.ttest_1samp(d.to_numpy(), 0.0)
                dm_stat, dm_p, dm_T = dm_hln_test(d.to_numpy(), hbins)
                person_of = a.drop_duplicates("window_id").set_index("window_id")["person_id"]
                d_by_person = d.groupby(person_of.reindex(paired).to_numpy()).mean()
                ct, cp = stats.ttest_1samp(d_by_person.to_numpy(), 0.0)

                rows.append({
                    "horizon": h, "arm_a": dl_name, "arm_b": fm_name,
                    "n_matched_windows": len(paired),
                    f"{dl_name}_mae_matched": round(float(sa["abs_error"].mean()), 4),
                    f"{dl_name}_rmse_matched": round(float(np.sqrt((sa["abs_error"] ** 2).mean())), 4),
                    f"{fm_name}_mae_matched": round(float(sb["abs_error"].mean()), 4),
                    f"{fm_name}_rmse_matched": round(float(np.sqrt((sb["abs_error"] ** 2).mean())), 4),
                    "dl_mae": round(float(sa["abs_error"].mean()), 4),
                    "fm_mae": round(float(sb["abs_error"].mean()), 4),
                    "dl_rmse": round(float(np.sqrt((sa["abs_error"] ** 2).mean())), 4),
                    "fm_rmse": round(float(np.sqrt((sb["abs_error"] ** 2).mean())), 4),
                    "mean_loss_diff_mse": float(d.mean()),
                    "naive_paired_t": float(naive_t), "naive_paired_p": float(naive_p),
                    "dm_hln_stat": float(dm_stat), "dm_hln_p": float(dm_p), "dm_hln_T": dm_T,
                    "n_participants_clustered": int(len(d_by_person)),
                    "cluster_by_participant_t": float(ct),
                    "cluster_by_participant_p": float(cp),
                })
                print(f"  {h} {dl_name} vs {fm_name}: MAE {sa['abs_error'].mean():.4f} vs "
                      f"{sb['abs_error'].mean():.4f} | cluster p={cp:.3g} | dm p={dm_p:.3g} "
                      f"| naive p={naive_p:.3g}", flush=True)

    if not rows:
        sys.exit("no comparisons produced")
    out = pd.DataFrame(rows)
    out.to_csv(f"{OUT_DIR}/dl_vs_fm_harness.csv", index=False)

    print("\n=== agreement between the three tests (vs timesfm) ===", flush=True)
    tf = out[out["arm_b"] == "timesfm"]
    for _, r in tf.iterrows():
        flags = [n for n, p in (("naive", r["naive_paired_p"]),
                                ("dm", r["dm_hln_p"]),
                                ("cluster", r["cluster_by_participant_p"])) if p < 0.05]
        print(f"  {r['horizon']:>5} {r['arm_a']:<10} significant under: "
              f"{', '.join(flags) if flags else 'none'}", flush=True)
    print(f"\nwrote {OUT_DIR}/dl_vs_fm_harness.csv", flush=True)
    print("DL_VS_FM_HARNESS_OK", flush=True)


if __name__ == "__main__":
    main()
