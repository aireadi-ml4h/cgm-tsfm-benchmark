#!/bin/bash
#SBATCH --job-name=ssmcgm_train_paper
#SBATCH --partition=normal
#SBATCH --qos=high_p
#SBATCH --time=6-00:00:00
#SBATCH --mem=200GB
#SBATCH --cpus-per-task=8
#SBATCH -G 1
#SBATCH --output=logs/%x_%j.out

# Fresh training run on the population-wide "paper" mode split
# (model_input/ssmcgm/ssmcgm_{train,val,test}_paper.csv -- literal replication
# of SSM-CGM's own per-participant last-2h-test/penultimate-2h-val holdout,
# applied uniformly across the whole population). Same model config/GPU as
# ssmcgm_train_real.sbatch's real disjoint-split run, but mem bumped from
# 96GB to 200GB (2026-09-06): the full-2h-withheld-region fix (§11 in
# report.md) grew this run's real validation set from 1,752 to 22,736
# windows (~13x), and the original 96GB allocation OOM-killed a dataloader
# worker at the end of epoch 0 (confirmed: 1,195 oom_kill events, job
# 42285611) -- a real resource-sizing issue caused by that fix, not a bug in
# the fix itself. No SSMCGM_RESUME_CKPT -- distinct population/split, must
# train from scratch.
set -euo pipefail
REPO_ROOT=${AIREADI_ROOT:?set AIREADI_ROOT to your aireadi-collab checkout}
ml load math py-pytorch/2.4.1_py312
ml load devel py-pandas/2.2.1_py312 py-pyarrow/18.1.0_py312
source "$REPO_ROOT/venv-ssmcgm/bin/activate"

export SSMCGM_NUM_GPUS=1
export SSMCGM_TRAIN_CSV="$REPO_ROOT/model_input/ssmcgm/ssmcgm_train_paper.csv"
export SSMCGM_VAL_CSV="$REPO_ROOT/model_input/ssmcgm/ssmcgm_val_paper.csv"
export SSMCGM_OUTPUT_DIR="$REPO_ROOT/ssmcgm_train_results/paper_mode"
python "$REPO_ROOT/code/ssmcgm/ssmcgm_train_real.py"
