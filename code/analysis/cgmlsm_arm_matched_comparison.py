"""Matched comparison between two CGM-LSM checkpoints' eval outputs, plus the
diagnostics behind model_input/HANDOFF.md's "CORRECTION 2026-09-05" block.

WHY THIS EXISTS: cgmlsm_eval.py drops non-numeric predictions (PAD/UNKNOWN/
MASK/Other_*) before computing MAE/RMSE -- correct on its own terms (a model
predicting "UNKNOWN" has not produced a forecast, and forcing it into a number
would understate its own uncertainty), but it means each checkpoint's
summary.csv MAE is an average over a DIFFERENT set of horizon bins. Diffing two
summary.csv files therefore compares averages over different populations -- the
same trap chronos2_matched_comparison.compare_arms()/restrict_keys exists to
avoid on the Chronos-2 side. This script is that rule applied to CGM-LSM arms.

The from-scratch checkpoint drops 16.2% of 2h bins; the settled T1D-excluded
fine-tuned checkpoint drops ~0. Since the dropped bins are systematically the
LATE steps of a rollout (see the collapse diagnostics below), the from-scratch
2h average is weighted toward short-lead, easier predictions -- so the
as-reported 50.524-vs-49.291 gap understates the settled checkpoint's real
advantage rather than overstating it.

Reads the per-bin predictions_<horizon>.csv files both arms already wrote; does
not re-run any model. CPU-only, seconds to run.

Usage (defaults reproduce the HANDOFF.md correction table exactly):
    python code/analysis/cgmlsm_arm_matched_comparison.py
    CGMLSM_ARM_A=cgmlsm_results/from_scratch \
    CGMLSM_ARM_B=cgmlsm_results/pretrain_no_t1d/finetuned_final \
        python code/analysis/cgmlsm_arm_matched_comparison.py
"""
import os

import numpy as np
import pandas as pd

REPO_ROOT = os.environ["AIREADI_ROOT"]
ARM_A = os.environ.get("CGMLSM_ARM_A", os.path.join(REPO_ROOT, "cgmlsm_results", "from_scratch"))
ARM_B = os.environ.get(
    "CGMLSM_ARM_B", os.path.join(REPO_ROOT, "cgmlsm_results", "pretrain_no_t1d", "finetuned_final")
)
LABEL_A = os.environ.get("CGMLSM_LABEL_A", "from_scratch")
LABEL_B = os.environ.get("CGMLSM_LABEL_B", "pretrain_no_t1d_finetuned")
OUTPUT_PATH = os.environ.get(
    "CGMLSM_ARM_COMPARISON_OUTPUT", os.path.join(REPO_ROOT, "cgmlsm_results", "arm_matched_comparison.csv")
)
HORIZONS = ("30min", "1h", "2h")
HORIZON_BINS = {"30min": 6, "1h": 12, "2h": 24}


def load_arm(arm_dir, horizon):
    """One arm's per-bin predictions for one horizon. `is_numeric` marks bins
    where the model actually produced a forecast; prediction_token is the raw
    vocabulary symbol, so which non-numeric symbol it chose stays inspectable."""
    path = os.path.join(arm_dir, f"predictions_{horizon}.csv")
    df = pd.read_csv(path, dtype={"window_id": str, "prediction_token": str})
    df["is_numeric"] = df["prediction_token"].str.fullmatch(r"\d+").fillna(False)
    # Step index within the rollout: the eval writer emits each window's horizon
    # bins in bin_start order, so cumcount within window_id recovers lead time.
    df["step"] = df.groupby("window_id").cumcount() + 1
    return df


def collapse_diagnostics(df, horizon):
    """Is a non-numeric prediction a correct 'sensor will be missing here'
    forecast, or a generation failure? Distinguished by what the ground truth
    actually was, and by whether the model ever recovers afterward."""
    nonnum = df[~df["is_numeric"]]
    truth_missing = int(nonnum["actual"].isna().sum())
    tokens = nonnum["prediction_token"].value_counts().to_dict()

    by_step = df.groupby("step")["is_numeric"].apply(lambda s: 100.0 * (~s).mean())

    # An absorbing state would mean: once UNKNOWN appears, every later step in
    # that window is also non-numeric (a pure suffix). Checked, not assumed.
    affected, suffix_only = 0, 0
    for _, g in df[df["window_id"].isin(nonnum["window_id"].unique())].groupby("window_id"):
        flags = (~g.sort_values("step")["is_numeric"]).to_numpy()
        affected += 1
        first_bad = int(np.argmax(flags))
        if flags[first_bad:].all():
            suffix_only += 1

    # Mean lead time of the SCORED bins vs the complete horizon's midpoint --
    # the size of the short-lead tilt the dropping introduces.
    scored = df[df["is_numeric"] & df["actual"].notna()]
    complete_ids = set(df["window_id"].unique()) - set(nonnum["window_id"].unique())
    complete = scored[scored["window_id"].isin(complete_ids)]

    return {
        "horizon": horizon,
        "n_bins_total": len(df),
        "n_nonnumeric": len(nonnum),
        "pct_nonnumeric": round(100.0 * len(nonnum) / len(df), 2),
        "nonnumeric_with_truth_ALSO_missing": truth_missing,
        "nonnumeric_with_truth_a_REAL_number": len(nonnum) - truth_missing,
        "nonnumeric_tokens": tokens,
        "n_windows_affected": affected,
        "n_windows_absorbing_suffix": suffix_only,
        "n_windows_recovered": affected - suffix_only,
        "mean_scored_step": round(float(scored["step"].mean()), 2),
        "complete_horizon_midpoint": (HORIZON_BINS[horizon] + 1) / 2,
        "mae_all_scored_bins": round(float(scored["abs_error"].mean()), 3),
        "mae_complete_windows_only": (
            round(float(complete["abs_error"].mean()), 3) if len(complete) else float("nan")
        ),
        "pct_nonnumeric_by_step": {int(k): round(v, 1) for k, v in by_step.items()},
    }


def matched_scores(a, b, horizon):
    """MAE/RMSE for both arms over the intersection of bins where BOTH produced
    a number AND ground truth exists. Merges on (window_id, bin_start) -- the
    same eval windows by construction, since both arms read the same
    cgmlsm_test.csv, so an unmatched key would itself be a bug worth surfacing."""
    keys = ["window_id", "bin_start"]
    m = a[keys + ["is_numeric", "actual", "abs_error"]].merge(
        b[keys + ["is_numeric", "actual", "abs_error"]], on=keys, how="outer",
        suffixes=("_a", "_b"), indicator=True,
    )
    unmatched = int((m["_merge"] != "both").sum())
    if unmatched:
        raise ValueError(
            f"{horizon}: {unmatched} horizon bins present in one arm but not the other; "
            "these arms were not scored on the same windows and are not comparable.")

    both = m[m["is_numeric_a"] & m["is_numeric_b"] & m["actual_a"].notna() & m["actual_b"].notna()]
    return {
        "horizon": horizon,
        "n_matched_bins": len(both),
        f"{LABEL_A}_mae_matched": round(float(both["abs_error_a"].mean()), 3),
        f"{LABEL_A}_rmse_matched": round(float(np.sqrt((both["abs_error_a"] ** 2).mean())), 3),
        f"{LABEL_B}_mae_matched": round(float(both["abs_error_b"].mean()), 3),
        f"{LABEL_B}_rmse_matched": round(float(np.sqrt((both["abs_error_b"] ** 2).mean())), 3),
        "mae_gap_matched_a_minus_b": round(
            float(both["abs_error_a"].mean() - both["abs_error_b"].mean()), 3),
        # Directional asymmetry: who bailed where the other answered.
        f"bins_{LABEL_A}_nonnumeric_{LABEL_B}_answered": int(
            (~m["is_numeric_a"] & m["is_numeric_b"]).sum()),
        f"bins_{LABEL_B}_nonnumeric_{LABEL_A}_answered": int(
            (m["is_numeric_a"] & ~m["is_numeric_b"]).sum()),
    }


def main():
    rows = []
    for horizon in HORIZONS:
        a, b = load_arm(ARM_A, horizon), load_arm(ARM_B, horizon)
        row = matched_scores(a, b, horizon)
        rows.append(row)

        print(f"===== {horizon} =====")
        print(f"  matched bins: {row['n_matched_bins']}")
        print(f"  {LABEL_A:28s} MAE {row[f'{LABEL_A}_mae_matched']:8.3f}  "
              f"RMSE {row[f'{LABEL_A}_rmse_matched']:8.3f}")
        print(f"  {LABEL_B:28s} MAE {row[f'{LABEL_B}_mae_matched']:8.3f}  "
              f"RMSE {row[f'{LABEL_B}_rmse_matched']:8.3f}")
        print(f"  bins {LABEL_A} bailed / {LABEL_B} answered: "
              f"{row[f'bins_{LABEL_A}_nonnumeric_{LABEL_B}_answered']}")
        print(f"  bins {LABEL_B} bailed / {LABEL_A} answered: "
              f"{row[f'bins_{LABEL_B}_nonnumeric_{LABEL_A}_answered']}")
        for label, arm in ((LABEL_A, a), (LABEL_B, b)):
            d = collapse_diagnostics(arm, horizon)
            if d["n_nonnumeric"] == 0:
                print(f"  [{label}] no non-numeric predictions at this horizon")
                continue
            print(f"  [{label}] non-numeric {d['n_nonnumeric']} ({d['pct_nonnumeric']}%), "
                  f"tokens={d['nonnumeric_tokens']}")
            print(f"           ground truth ALSO missing: "
                  f"{d['nonnumeric_with_truth_ALSO_missing']}, "
                  f"was a REAL number: {d['nonnumeric_with_truth_a_REAL_number']}")
            print(f"           windows affected {d['n_windows_affected']}, "
                  f"absorbing (never recovered) {d['n_windows_absorbing_suffix']}, "
                  f"recovered {d['n_windows_recovered']}")
            print(f"           mean scored step {d['mean_scored_step']} of "
                  f"{HORIZON_BINS[horizon]} (complete midpoint "
                  f"{d['complete_horizon_midpoint']}); "
                  f"MAE all-scored {d['mae_all_scored_bins']} vs "
                  f"complete-windows-only {d['mae_complete_windows_only']}")
            print(f"           non-numeric % by step: {d['pct_nonnumeric_by_step']}")

    out = pd.DataFrame(rows)
    out.to_csv(OUTPUT_PATH, index=False)
    print(f"\nwrote {OUTPUT_PATH}")
    print("CGMLSM_ARM_MATCHED_COMPARISON_OK")


if __name__ == "__main__":
    main()
