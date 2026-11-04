#!/bin/bash
#SBATCH --job-name=dl_vs_fm
#SBATCH --partition=owners
#SBATCH --qos=normal
#SBATCH --time=00:30:00
#SBATCH --mem=32GB
#SBATCH --cpus-per-task=2
#SBATCH --output=logs/%x_%j.out
# Matched LSTM/Mamba vs TimesFM AND Chronos-2. The existing per-model
# significance.csv files only compare against Chronos-2, the weaker of the two
# foundation models, so the headline claim was not established. CPU-only.
set -euo pipefail
REPO_ROOT=${AIREADI_ROOT:?set AIREADI_ROOT to your aireadi-collab checkout}
ml load devel py-numpy/1.26.3_py312 py-pandas/2.2.1_py312
ml load math py-scipy/1.12.0_py312
cd "$REPO_ROOT"
python3 code/baselines/dl_baselines_vs_foundation_models.py
