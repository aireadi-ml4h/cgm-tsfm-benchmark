#!/bin/bash
#SBATCH --job-name=timesfm_lora_pers_full
#SBATCH --partition=normal
#SBATCH --qos=high_p
#SBATCH --time=1-00:00:00
#SBATCH --mem=32GB
#SBATCH --cpus-per-task=4
#SBATCH -G 1
#SBATCH --output=/scratch/users/%u/logs/%x_%j.out

set -euo pipefail
REPO_ROOT=${AIREADI_ROOT:?set AIREADI_ROOT to your aireadi-collab checkout}
export HF_HOME=$SCRATCH/hf_cache
mkdir -p "$HF_HOME"

# NOT loading py-huggingface-hub or py-safetensors modules -- see
# timesfm_lora_env_fix2.sbatch's comment: both are too old for
# transformers==5.15.0 and were confirmed to shadow the venv's own
# pip-installed compatible versions when loaded alongside them.
ml load math py-pytorch/2.9.1_py314 py-numpy/2.3.5_py314 py-pandas/2.3.3_py314 py-pyarrow/22.0.0_py314
source "$REPO_ROOT/venv-timesfm/bin/activate"

: "${TIMESFM_LORA_SPLIT:?set TIMESFM_LORA_SPLIT to val or test}"
export TIMESFM_LORA_SPLIT

# Full run: all three history-day tiers, sequentially, full population
# (no TIMESFM_LORA_MAX_PARTICIPANTS cap), real NUM_STEPS=100 (script default,
# not the smoke test's 10). The script tracks a per-output-dir
# participants_done_shard0.csv and skips anyone already scored, so a
# requeue/interruption resumes instead of restarting -- same guarantee
# requested earlier this session for the SSM-CGM runs.
for days in 1 3 7; do
  echo "=== history_days=$days split=$TIMESFM_LORA_SPLIT ===" >&2
  export TIMESFM_LORA_HISTORY_DAYS=$days
  python "$REPO_ROOT/code/timesfm/timesfm_lora_personalization.py"
done
