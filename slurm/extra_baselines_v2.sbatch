#!/bin/bash
#SBATCH --job-name=extra_baselines_v2
#SBATCH --partition=normal
#SBATCH --qos=high_p
#SBATCH --time=06:00:00
#SBATCH --mem=64GB
#SBATCH --cpus-per-task=8
#SBATCH -G 1
#SBATCH --output=logs/%x_%j.out
#
# Circadian (seasonal-naive) baseline and the paper's MLP: a torch MLP trained
# on absolute error. MLP_IMPL=torch MLP_LOSS=l1 are also the script's defaults;
# they are set explicitly below so the job states what it runs. The original
# squared-error sklearn MLP is MLP_IMPL=sklearn MLP_LOSS=l2 (add
# MLP_OUTPUT_SUFFIX=_sklearn_l2 so it doesn't overwrite this run's output).
#
# A GPU is requested because the paper's run used one, and the torch MLP uses
# it when present. The model is small enough to train on CPU, but results may
# then differ slightly from the paper's.
#
# Modules match the job that produced the paper's run: the py312 stack (fixed
# by py-lightgbm/4.5.0_py312, since there is no lightgbm wheel for py314; the
# feature builder is shared with lightgbm_zeroshot_baseline.py) plus
# py-pytorch/2.4.1_py312. No venv is activated: everything is available as
# modules.

set -euo pipefail
REPO_ROOT=${AIREADI_ROOT:?set AIREADI_ROOT to your cgm-tsfm-benchmark checkout}
ml load devel py-numpy/1.26.3_py312 py-pandas/2.2.1_py312
ml load math py-scipy/1.12.0_py312 py-scikit-learn/1.5.1_py312 py-lightgbm/4.5.0_py312
ml load math py-pytorch/2.4.1_py312

python3 -c "import torch, lightgbm, sklearn, pandas, numpy, scipy; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), 'lightgbm', lightgbm.__version__, 'sklearn', sklearn.__version__)"
MLP_IMPL=torch MLP_LOSS=l1 python3 "$REPO_ROOT/code/baselines/extra_baselines_v2.py"
