#!/bin/bash
#SBATCH --job-name=cgmlsm_pers_test
#SBATCH --partition=normal
#SBATCH --qos=high_p
#SBATCH --time=06:00:00
#SBATCH --mem=32GB
#SBATCH --cpus-per-task=4
#SBATCH -G 1
#SBATCH --output=logs/%x_%j.out

# THE TEST-SPLIT PERSONALIZATION RUN. All 7 arms, one job, one shared config.
#
# SOURCES slurm/cgmlsm_personalization_config.sh AND CALLS pers_set_k PER ARM.
# An earlier version of this file did NEITHER, and would have run every arm on
# code/cgmlsm/cgmlsm_personalization.py's module defaults -- lr=5e-6, 50 steps, and
# the OLD ep1/lr=5e-6 base checkpoint. That is precisely the recipe the val
# work showed does essentially nothing (+0.18/+0.36/+3.03 mg/dL, and actively
# harmful at 30min with more steps). It would have produced a near-null
# test-split result indistinguishable from a real finding. Do not remove the
# `source` line or the pers_set_k calls.
#
# ARMS:
#   zs_glucose_only   -- k=0, no fine-tuning. The baseline every other arm is
#                        compared against. Must be produced HERE: the base
#                        checkpoint's own test-window numbers were scored on
#                        cgmlsm_test.csv's windows, NOT the personalization
#                        eval set's windows/participants, so they cannot serve.
#   p1d / p3d / p7d          -- own history, per-k frozen recipe.
#   p1d/p3d/p7d_shuffled     -- donor history at the SAME recipe. Negative
#                        control, not a target; see the config file's caveat 1
#                        about the instability of this decomposition.
#
# HORIZONS: 30min/1h/2h only. 3h/4h is impossible for this architecture
# (n_positions=168 caps generation at 24 bins past the context), so unlike the
# Chronos-2 arm there is no separate 3h/4h block to report.
#
# All arms write into ONE output root; the script namespaces by arm, and each
# arm resumes per (participant, arm) from its own done_shard0.csv, so a requeue
# picks up where it stopped. ~7 arms x ~264 participants at a few minutes per
# arm.
set -euo pipefail
REPO_ROOT=${AIREADI_ROOT:?set AIREADI_ROOT to your cgm-tsfm-benchmark checkout}
ml load math py-pytorch/2.4.1_py312
source "$REPO_ROOT/venv-cgmlsm/bin/activate"

# THE frozen config. Sets base checkpoint, batch size, stride, seed, and
# provides pers_set_k for the per-k learning rate / step count.
source "$REPO_ROOT/slurm/cgmlsm_personalization_config.sh"

BASE_TAG="$(basename "$(dirname "$CGMLSM_PERS_BASE_CKPT")")"
export CGMLSM_PERS_SPLIT=test
export CGMLSM_PERS_OUTPUT_ROOT="$REPO_ROOT/cgmlsm_personalization_results/test_${BASE_TAG}"
echo "base: $CGMLSM_PERS_BASE_CKPT"
echo "out:  $CGMLSM_PERS_OUTPUT_ROOT"

# Zero-shot first, so the baseline exists before anything is compared to it.
# pers_set_k is called anyway rather than left at a module default, so no
# stale value can appear in this arm's recorded provenance -- k=0 skips the
# fit path entirely, so the values are unused.
pers_set_k 7d
export CGMLSM_PERS_ARMS=zs_glucose_only
echo "===== TEST arm: zs_glucose_only ====="
python "$REPO_ROOT/code/cgmlsm/cgmlsm_personalization.py"

for K in 1d 3d 7d; do
  pers_set_k "$K"
  echo "----- k=$K frozen recipe: lr=$CGMLSM_PERS_LR steps=$CGMLSM_PERS_NUM_STEPS -----"

  export CGMLSM_PERS_ARMS="p${K}"
  echo "===== TEST arm: p${K} (own history) ====="
  python "$REPO_ROOT/code/cgmlsm/cgmlsm_personalization.py"

  export CGMLSM_PERS_ARMS="p${K}_shuffled"
  echo "===== TEST arm: p${K}_shuffled (donor control) ====="
  python "$REPO_ROOT/code/cgmlsm/cgmlsm_personalization.py"
done

echo "CGMLSM_PERS_TEST_ARMS_OK"
