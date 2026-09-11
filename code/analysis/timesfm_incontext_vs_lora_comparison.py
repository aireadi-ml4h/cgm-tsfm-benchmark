"""Matched, significance-tested comparison of TimesFM's two personalization
mechanisms -- in-context (no weight updates) vs LoRA (per-person fine-tune),
both glucose-only, both already complete (full val+test, k=1/3/7).

Mirrors timesfm_incontext_personalization_val_compare.py's own pattern
exactly (same compare_arms() call, same eval_dir/restrict_keys mechanism),
swapping the zeroshot_ref arm for the lora_{k}d arm at the SAME k, so this
is a same-history-length, same-eval-set, same-population comparison of
mechanism only (in-context vs LoRA), not history length or population.

Runs both SPLIT=val and SPLIT=test in one job (unlike the val_compare
script, which takes SPLIT as an env var and only runs one).
"""
import os
import sys

import pandas as pd

REPO_ROOT = "/path/to/aireadi-collab"
sys.path.insert(0, f"{REPO_ROOT}/code")
from chronos2_matched_comparison import compare_arms  # noqa: E402

HORIZON_BINS = {"30min": 6, "1h": 12, "2h": 24, "3h": 36, "4h": 48}
LORA_SUFFIX = os.environ.get("TIMESFM_LORA_SUFFIX", "")  # set to "_tuned" for the tuned rerun


def run_split(split):
    eval_dir = (f"{REPO_ROOT}/model_input/zeroshot_eval" if split == "test"
                else f"{REPO_ROOT}/model_input/zeroshot_eval_{split}")
    eval_set_path = f"{REPO_ROOT}/model_input/personalization/eval_set_{split}.csv"
    e = pd.read_csv(eval_set_path, dtype={"person_id": str, "window_id": str})

    rows = []
    for k in (1, 3, 7):
        incontext_arm = {
            "label": f"incontext_{k}d",
            "results_dir": f"{REPO_ROOT}/timesfm_personalization_results/{split}/incontext_{k}d",
            "condition": "glucose_only",
            "eval_dir": eval_dir,
        }
        lora_arm = {
            "label": f"lora_{k}d{LORA_SUFFIX}",
            "results_dir": f"{REPO_ROOT}/timesfm_personalization_results/{split}/lora_{k}d{LORA_SUFFIX}",
            "condition": "glucose_only",
            "eval_dir": eval_dir,
        }
        for horizon_name, horizon_bins in HORIZON_BINS.items():
            restrict_keys = set(e.loc[e["horizon"] == horizon_name, "key"])
            row = compare_arms(horizon_name, horizon_bins, incontext_arm, lora_arm, restrict_keys=restrict_keys)
            row["split"] = split
            row["history_days"] = k
            rows.append(row)
    return rows


def main():
    all_rows = []
    for split in ("val", "test"):
        eval_set_path = f"{REPO_ROOT}/model_input/personalization/eval_set_{split}.csv"
        if not os.path.exists(eval_set_path):
            print(f"skip split={split}: no eval_set file", file=sys.stderr)
            continue
        if LORA_SUFFIX and not os.path.exists(
            f"{REPO_ROOT}/timesfm_personalization_results/{split}/lora_1d{LORA_SUFFIX}"
        ):
            print(f"skip split={split}: no lora_*{LORA_SUFFIX} results (tuned rerun was test-only)", file=sys.stderr)
            continue
        all_rows.extend(run_split(split))

    out = pd.DataFrame(all_rows)
    suffix_tag = LORA_SUFFIX or "_untuned"
    out_path = f"{REPO_ROOT}/timesfm_personalization_results/incontext_vs_lora_comparison{suffix_tag}.csv"
    out.to_csv(out_path, index=False)
    print(f"wrote {len(out)} rows to {out_path}", file=sys.stderr)
    cols = [c for c in out.columns if c in (
        "split", "horizon", "history_days", "n_matched_windows", "n_paired_windows",
        "mean_loss_diff_mse", "naive_paired_p", "dm_hln_p",
        "n_participants_clustered", "cluster_by_participant_p")]
    print(out[cols].to_string(index=False), file=sys.stderr)


if __name__ == "__main__":
    main()
