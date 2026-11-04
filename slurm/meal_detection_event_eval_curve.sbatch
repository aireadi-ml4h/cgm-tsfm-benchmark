#!/bin/bash
#SBATCH --job-name=meal_detection_pr_curve
#SBATCH --partition=normal
#SBATCH --qos=high_p
#SBATCH --time=00:30:00
#SBATCH --mem=16GB
#SBATCH --cpus-per-task=4
#SBATCH -G 1
#SBATCH --output=logs/%x_%j.out

set -euo pipefail
REPO_ROOT=${AIREADI_ROOT:?set AIREADI_ROOT to your aireadi-collab checkout}
ml load math py-pytorch/2.9.1_py314 py-huggingface-hub/0.36.0_py314 py-safetensors/0.7.0_py314 py-numpy/2.3.5_py314 py-pandas/2.3.3_py314 py-pyarrow/22.0.0_py314
ml load math py-scikit-learn/1.8.0_py314
source "$REPO_ROOT/venv-timesfm/bin/activate"
export CGMACROS_DIR=/path/to/cgmacros/1.0.0/extracted/CGMacros
python "$REPO_ROOT/code/meal_detection/evaluate_event_level_curve.py"
