#!/bin/bash
#SBATCH --job-name=cgmlsm_ft_seed_stability
#SBATCH --partition=normal
#SBATCH --qos=high_p
#SBATCH --time=03:00:00
#SBATCH --mem=32GB
#SBATCH --cpus-per-task=4
#SBATCH -G 1
#SBATCH --output=/scratch/users/%u/logs/%x_%j.out

# IS THE CHOSEN BASE RECIPE STABLE, OR IS IT LUCK NEXT TO A CLIFF?
#
# Sweeps 3-5 picked 2 epochs at lr=1e-2 (+36.6% mean relative MAE vs the old
# settled checkpoint on val windows). That learning rate is 10-100x higher than
# is normal for AdamW on a transformer, and the surrounding evidence says it is
# sitting immediately next to a failure boundary:
#   - lr=2e-2 (one rung up) destroys the model completely: 100% PAD tokens,
#     zero scoreable predictions, at both 2 and 3 epochs.
#   - the SAME lr=1e-2 with one extra epoch already declines to forecast on
#     0.39%/0.56% of 1h/2h bins, and with two extra on 1.61%/3.48%.
# Every result so far comes from ONE seed (42), which was hardcoded until now.
# A single-seed result adjacent to a cliff is not evidence of a stable optimum.
#
# This runs the chosen recipe at three additional seeds, plus the safer rung
# (2 epochs at lr=5e-3, which was +18.0% with far less degeneration) at the
# same seeds, so the comparison is like-for-like.
#
# HOW TO READ IT, stated before running:
#   - lr=1e-2 numbers tight across seeds        -> recipe is usable; freeze it.
#   - lr=1e-2 scatters, or any seed collapses   -> the 36.6% is partly luck.
#     Fall back to lr=5e-3, which trades ~18 points of relative improvement for
#     a recipe that is not balanced on an edge. A fragile base would also make
#     the personalization arm built on top of it irreproducible, which is worse
#     than a smaller but solid gain.
set -euo pipefail
REPO_ROOT=${AIREADI_ROOT:?set AIREADI_ROOT to your aireadi-collab checkout}
SCRATCH_BASE="$SCRATCH/aireadi_cgmlsm"
PRETRAINED="$SCRATCH_BASE/pretrain_glucofmbench_no_t1d_v1/final"
VAL_WINDOWS="$REPO_ROOT/model_input/cgmlsm/cgmlsm_val_windows.csv"
RESULTS_ROOT="$REPO_ROOT/cgmlsm_results/finetune_seed_stability"

ml load math py-pytorch/2.4.1_py312
source "$REPO_ROOT/venv-cgmlsm/bin/activate"

test -f "$VAL_WINDOWS" || { echo "missing $VAL_WINDOWS" >&2; exit 1; }
mkdir -p "$RESULTS_ROOT"

for LR in 1e-2 5e-3; do
  for SEED in 7 123 2024; do
    TAG="ep2_lr${LR}_seed${SEED}"
    CKPT="$SCRATCH_BASE/finetune_seedcheck_${TAG}"
    EVALOUT="$RESULTS_ROOT/$TAG"

    if [ -f "$EVALOUT/summary.csv" ]; then
      echo "===== skip $TAG ====="
      continue
    fi

    echo "===== $TAG: fine-tuning ====="
    if [ ! -f "$CKPT/final/model.safetensors" ]; then
      CGMLSM_PRETRAINED_CHECKPOINT_DIR="$PRETRAINED" \
      CGMLSM_FINETUNE_OUTPUT_DIR_OVERRIDE="$CKPT" \
      CGMLSM_NUM_EPOCHS=2 \
      CGMLSM_LEARNING_RATE="$LR" \
      CGMLSM_SEED="$SEED" \
        python "$REPO_ROOT/code/cgmlsm/cgmlsm_finetune_aireadi.py"
    fi

    echo "===== $TAG: evaluating on VAL windows ====="
    CGMLSM_CHECKPOINT_DIR="$CKPT/final" \
    CGMLSM_EVAL_INPUT="$VAL_WINDOWS" \
    CGMLSM_EVAL_OUTPUT_DIR="$EVALOUT" \
      python "$REPO_ROOT/code/cgmlsm/cgmlsm_eval.py"
  done
done

echo "=== seed-stability summary (val windows) ==="
for d in "$RESULTS_ROOT"/*/; do
  echo "--- $(basename "$d")"
  cut -d, -f1,2,3,4,5,8 "$d/summary.csv" 2>/dev/null || echo "   no summary"
done

echo "CGMLSM_FT_SEED_STABILITY_OK"
