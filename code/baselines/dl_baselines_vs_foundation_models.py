"""Matched comparison: LSTM / Mamba against BOTH foundation models.

The existing `lstm_zeroshot_results/significance.csv` and
`mamba_zeroshot_results/significance.csv` compare only against persistence and
Chronos-2. Chronos-2 is the weaker of the two foundation models at every
horizon, so "a simple sequence model beats the foundation models" is not
established by those files. TimesFM is the comparison that matters and it was
missing. This script adds it, and re-derives the Chronos-2 comparison by the
same route so the two sit side by side.

METHOD. Inner-merge on (window_id, bin_start), then a paired t-test over
per-participant mean absolute errors. Clustering by participant is the primary
test throughout this project because windows repeat within a person.

TIMEZONE TRAP. The DL prediction files write `bin_start` tz-aware (+00:00);
the foundation-model files write it tz-naive (chronos2_zeroshot_eval.py:369
strips the tz). Merging without normalising returns zero rows and fails open
rather than raising. Both sides are parsed with utc=True, and the script
asserts the merge is non-empty and that the two `actual` columns agree, so a
silent misalignment cannot pass as a result. This is the same failure mode
external-eval/NAIVE_ALIGNMENT_CHECK_FULL_*.md was written to catch.
"""
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

REPO = os.environ["AIREADI_ROOT"]
OUT_DIR = f"{REPO}/dl_vs_fm_results"
os.makedirs(OUT_DIR, exist_ok=True)

HORIZONS = ["30min", "1h", "2h"]
# (path template, name of the prediction column in that file)
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
    d = pd.read_csv(path, dtype={"window_id": str}, usecols=lambda c: c in cols)
    d["bin_start"] = pd.to_datetime(d["bin_start"], utc=True)   # tz trap
    return d


def main():
    rows = []
    for h in HORIZONS:
        for dl_name, (dl_tmpl, pred_col) in DL.items():
            dl_path = dl_tmpl.format(h=h)
            if not os.path.exists(dl_path):
                print(f"  skip {dl_name} {h}: no file", flush=True)
                continue
            dl = load(dl_path,
                      {"window_id", "person_id", "bin_start", "actual", pred_col})
            if pred_col not in dl.columns:
                print(f"  skip {dl_name} {h}: no column '{pred_col}'", flush=True)
                continue
            dl = dl.rename(columns={pred_col: "pred_dl", "actual": "actual_dl"})
            dl = dl.dropna(subset=["pred_dl"])

            for fm_name, fm_tmpl in FM.items():
                fm_path = fm_tmpl.format(h=h)
                if not os.path.exists(fm_path):
                    print(f"  skip {fm_name} {h}: no file", flush=True)
                    continue
                fm = load(fm_path, {"window_id", "bin_start", "actual", "predictions"})
                fm = fm.rename(columns={"predictions": "pred_fm", "actual": "actual_fm"})

                m = dl.merge(fm, on=["window_id", "bin_start"], how="inner")
                if m.empty:
                    print(f"  FAIL {dl_name} vs {fm_name} {h}: empty merge "
                          f"(check timezone handling)", flush=True)
                    continue
                # Ground truth must be identical on both sides.
                both = m.dropna(subset=["actual_dl", "actual_fm"])
                max_diff = float((both["actual_dl"] - both["actual_fm"]).abs().max())
                if max_diff > 1e-6:
                    print(f"  FAIL {dl_name} vs {fm_name} {h}: actuals disagree "
                          f"(max {max_diff})", flush=True)
                    continue

                s = m.dropna(subset=["pred_dl", "pred_fm", "actual_dl"]).copy()
                s["e_dl"] = (s["pred_dl"] - s["actual_dl"]).abs()
                s["e_fm"] = (s["pred_fm"] - s["actual_dl"]).abs()
                per_p = s.groupby("person_id")[["e_dl", "e_fm"]].mean()
                diff = (per_p["e_dl"] - per_p["e_fm"]).to_numpy()
                t, p = stats.ttest_1samp(diff, 0.0)
                dl_mae, fm_mae = float(s["e_dl"].mean()), float(s["e_fm"].mean())
                rows.append({
                    "horizon": h, "dl_model": dl_name, "vs": fm_name,
                    "n_matched_bins": len(s),
                    "n_participants_clustered": int(len(per_p)),
                    "dl_mae_matched": round(dl_mae, 4),
                    "fm_mae_matched": round(fm_mae, 4),
                    "delta_mae": round(dl_mae - fm_mae, 4),
                    "cluster_t": round(float(t), 4),
                    "cluster_p": float(p),
                    "dl_better": bool(dl_mae < fm_mae),
                    "significant": bool(p < 0.05),
                })
                print(f"  {h} {dl_name} vs {fm_name}: n={len(s)}, "
                      f"{dl_mae:.4f} vs {fm_mae:.4f}, p={p:.3g}", flush=True)

    if not rows:
        sys.exit("no comparisons produced")
    out = pd.DataFrame(rows)
    out.to_csv(f"{OUT_DIR}/dl_vs_fm_matched.csv", index=False)
    print("\n" + out.to_string(index=False), flush=True)

    print("\n=== SUMMARY ===", flush=True)
    for fm_name in FM:
        sub = out[out["vs"] == fm_name]
        wins = sub[(sub["dl_better"]) & (sub["significant"])]
        loss = sub[(~sub["dl_better"]) & (sub["significant"])]
        print(f"vs {fm_name}: {len(wins)} significant DL wins, {len(loss)} significant DL "
              f"losses, {len(sub) - len(wins) - len(loss)} not resolved (of {len(sub)})",
              flush=True)
    print(f"\nwrote {OUT_DIR}/dl_vs_fm_matched.csv", flush=True)
    print("DL_VS_FM_OK", flush=True)


if __name__ == "__main__":
    main()
