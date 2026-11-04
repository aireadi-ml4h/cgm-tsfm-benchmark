#!/bin/bash
#SBATCH --job-name=chronos2_pers_cov_evalset
#SBATCH --partition=normal
#SBATCH --qos=high_p
#SBATCH --time=01:00:00
#SBATCH --mem=32GB
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/%x_%j.out

# Phase-2 (covariate) prerequisites, both CPU-only, no GPU requested:
#
#   1. build_personalization_eval_set.py --cov test
#      The with_covariates eval set E_cov, anchors REUSED from phase 1's
#      eval_set_test.csv so that a covariate-vs-glucose-only difference cannot
#      be an anchor-position difference. See that script's "PHASE 2" docstring.
#
#   2. measure_covariate_history_coverage.py
#      The feasibility number this whole phase turns on: covariate history
#      comes from build_grid(), which is bounded by the WEARABLE record, while
#      the 7-day bar is defined on the GLUCOSE record. Clearing the bar
#      therefore does NOT imply 7 days of covariate history exist.
#
# Run this before building the covariate fine-tuning path, not after.

set -euo pipefail
REPO_ROOT=${AIREADI_ROOT:?set AIREADI_ROOT to your aireadi-collab checkout}
ml load math py-pandas/2.3.3_py314 py-scipy/1.16.3_py314

cd "$REPO_ROOT/code"
# syntax-only check of the covariate fine-tuning path here, where no GPU is
# held: a SyntaxError should not cost a GPU allocation to discover
python3 -m py_compile chronos2_personalization.py \
    model_input/build_personalization_eval_set.py \
    model_input/measure_covariate_history_coverage.py

cd "$REPO_ROOT/code/model_input"
python3 build_personalization_eval_set.py --cov test
CHRONOS2_PERS_HISTORY_DAYS=7 python3 measure_covariate_history_coverage.py
