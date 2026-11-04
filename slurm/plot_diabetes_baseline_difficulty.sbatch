#!/bin/bash
#SBATCH --job-name=plot_diabetes_baseline
#SBATCH --partition=normal
#SBATCH --time=00:10:00
#SBATCH --mem=8GB
#SBATCH --cpus-per-task=2
#SBATCH --output=logs/%x_%j.out
set -euo pipefail
ml load math py-pytorch/2.4.1_py312
ml load devel py-pandas/2.2.1_py312
source ${AIREADI_ROOT}/venv/bin/activate
pip install --quiet --user matplotlib 2>/dev/null || true
python ${AIREADI_ROOT}/code/analysis/plot_diabetes_baseline_difficulty.py
