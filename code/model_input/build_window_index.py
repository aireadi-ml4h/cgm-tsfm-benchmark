"""Population-wide window index: for every participant, every horizon
(30min/1h/2h), and both covariate conditions (glucose-only, with-covariates),
extract coverage-eligible non-overlapping windows and record their metadata
(not the data itself -- per-model exporters re-slice build_grid() using this
index so there is one source of truth for which windows are eligible).
"""
import glob
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from build_participant_grid import build_grid, load_participants_covariates  # noqa: E402
from windows import extract_windows, HORIZONS_BINS  # noqa: E402

REPO_ROOT = os.environ["AIREADI_ROOT"]
ANALYSIS_DIR = f"{REPO_ROOT}/analysis_dataframe"
OUTPUT_PATH = f"{REPO_ROOT}/model_input/window_index.csv"


def run():
    covs = load_participants_covariates()
    paths = sorted(glob.glob(f"{ANALYSIS_DIR}/*.csv"))
    print(f"participants found: {len(paths)}", file=sys.stderr)

    rows = []
    for i, path in enumerate(paths):
        pid = os.path.basename(path).replace(".csv", "")
        df = build_grid(pid, covs)
        if df is None:
            continue
        cov = covs.get(str(pid), {})
        split = cov.get("recommended_split", "")
        for horizon_name, horizon_bins in HORIZONS_BINS.items():
            for with_covariates in [False, True]:
                windows = extract_windows(df, horizon_bins, with_covariates=with_covariates)
                for w in windows:
                    rows.append({
                        "person_id": pid,
                        "recommended_split": split,
                        "study_group": cov.get("study_group", ""),
                        "clinical_site": cov.get("clinical_site", ""),
                        "age": cov.get("age"),
                        "horizon": horizon_name,
                        "with_covariates": with_covariates,
                        "context_start": w["context_start"],
                        "target_start": w["target_start"],
                    })
        if (i + 1) % 200 == 0:
            print(f"  processed {i + 1}/{len(paths)}", file=sys.stderr)

    index_df = pd.DataFrame(rows)
    index_df.to_csv(OUTPUT_PATH, index=False)
    print(f"wrote {len(index_df)} window records to {OUTPUT_PATH}", file=sys.stderr)

    summary = index_df.groupby(["horizon", "with_covariates", "recommended_split"]).agg(
        n_windows=("person_id", "size"), n_participants=("person_id", "nunique")
    )
    print(summary, file=sys.stderr)


if __name__ == "__main__":
    run()
