#!/bin/bash
# Export the shared zero-shot evaluation windows, in both wearable-imputation
# variants.
#
# Both arms run code/model_input/export_zeroshot_windows.py with identical
# resources. Consolidated from two near-identical scripts.
#
#   default   the shared model_input/zeroshot_eval/*.csv used by every model
#   noimpute  the same window selection (window_index.csv is unchanged) but
#             sourced from the non-LOCF wearables analysis dataframe. Used for
#             the Chronos-2/TimesFM imputation robustness check; writes to its
#             own zeroshot_eval_noimpute/ dir and never touches the shared one.
#
#SBATCH --job-name=export_zeroshot_windows
#SBATCH --partition=normal
#SBATCH --qos=high_p
#SBATCH --time=01:00:00
#SBATCH --mem=16GB
#SBATCH --cpus-per-task=1
#SBATCH --array=0-1
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err

: "${AIREADI_ROOT:?set AIREADI_ROOT to your cgm-tsfm-benchmark checkout}"

# arm | analysis-dataframe dir | zeroshot-eval output dir  ("-" = script default)
ARMS=(
  "default|-|-"
  "noimpute|analysis_dataframe_noimpute_wearables|model_input/zeroshot_eval_noimpute"
)
IFS='|' read -r ARM_NAME ANALYSIS_SUBDIR OUT_SUBDIR <<< "${ARMS[${SLURM_ARRAY_TASK_ID:-0}]}"
[ "$ANALYSIS_SUBDIR" != "-" ] && export ANALYSIS_DIR_OVERRIDE=${AIREADI_ROOT}/$ANALYSIS_SUBDIR
[ "$OUT_SUBDIR" != "-" ] && export ZEROSHOT_EVAL_OUTPUT_DIR_OVERRIDE=${AIREADI_ROOT}/$OUT_SUBDIR
echo "arm=$ARM_NAME analysis=$ANALYSIS_SUBDIR out=$OUT_SUBDIR"

[ -f /etc/profile.d/modules.sh ] && source /etc/profile.d/modules.sh
ml devel python/3.12.1
# The venv's python3 needs the loaded Python's libpython on the library path.
export LD_LIBRARY_PATH=$(dirname "$(dirname "$(command -v python3)")")/lib:$LD_LIBRARY_PATH
VENV=${AIREADI_ROOT}/venv/bin/python3

"$VENV" ${AIREADI_ROOT}/code/model_input/export_zeroshot_windows.py
