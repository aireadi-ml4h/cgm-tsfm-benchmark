#!/bin/bash
#SBATCH --job-name=extra_baselines
#SBATCH --partition=normal
#SBATCH --qos=normal
#SBATCH --time=03:00:00
#SBATCH --mem=32GB
#SBATCH --cpus-per-task=8
#SBATCH --output=logs/%x_%j.out
#
# FIX 2026-09-09: the previous version did `pip install lightgbm` inside
# venv-chronos. There is no lightgbm wheel for the py314 stack, so pip fell
# back to building from source and died with
#   FileNotFoundError: .../py-pytorch/2.9.1_py314/bin/cmake
# (job 42565550, FAILED in 25s).
#
# the cluster ships `py-lightgbm/4.5.0_py312` in the `math` category (verified
# with `ml spider py-lightgbm`), so load the module instead of building.
# That forces the whole stack to py312 rather than py314, hence the changed
# module versions below -- mixing a py312 extension into a py314 interpreter
# would not import.
#
# No venv is activated: this baseline needs only pandas/numpy/scipy/lightgbm,
# all available as modules. venv-chronos exists for the Chronos-2 stack and
# is irrelevant here.
#
# qos=normal (was high_p) so a CPU-only baseline does not consume one of the
# four high_p slots the GPU training jobs share.

set -euo pipefail
REPO_ROOT=${AIREADI_ROOT:?set AIREADI_ROOT to your aireadi-collab checkout}
ml load devel py-numpy/1.26.3_py312 py-pandas/2.2.1_py312
ml load math py-scipy/1.12.0_py312 py-scikit-learn/1.5.1_py312 py-lightgbm/4.5.0_py312

python3 -c "import lightgbm, sklearn, pandas, numpy, scipy; print('lightgbm', lightgbm.__version__, 'sklearn', sklearn.__version__)"
python3 "$REPO_ROOT/code/baselines/extra_baselines.py"
