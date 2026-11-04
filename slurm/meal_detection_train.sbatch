#!/bin/bash
#SBATCH --job-name=meal_detection_train
#SBATCH --partition=normal
#SBATCH --qos=high_p
#SBATCH --time=04:00:00
#SBATCH --mem=32GB
#SBATCH --cpus-per-task=4
#SBATCH -G 1
#SBATCH --output=logs/%x_%j.out

set -euo pipefail
REPO_ROOT=${AIREADI_ROOT:?set AIREADI_ROOT to your cgm-tsfm-benchmark checkout}
# Reuses venv-timesfm's pytorch/pandas/numpy install, but WITHOUT loading
# the py-jax module: py-jax pulls in cudnn/9.13.1, which conflicts with the
# cudnn/9.14.0 py-pytorch/2.9.1_py314 was built against -- harmless for
# Chronos-2/TimesFM (neither uses a cuDNN-accelerated RNN kernel) but fatal
# here (nn.LSTM.flatten_parameters() hits it immediately), confirmed
# directly (RuntimeError: cuDNN version incompatibility, job 41131511).
# py-scikit-learn loads fine standalone, no jax dependency.
ml load math py-pytorch/2.9.1_py314 py-huggingface-hub/0.36.0_py314 py-safetensors/0.7.0_py314 py-numpy/2.3.5_py314 py-pandas/2.3.3_py314 py-pyarrow/22.0.0_py314
ml load math py-scikit-learn/1.8.0_py314
source "$REPO_ROOT/venv-timesfm/bin/activate"

export CGMACROS_DIR=/path/to/cgmacros/1.0.0/extracted/CGMacros

python "$REPO_ROOT/code/meal_detection/train.py"
