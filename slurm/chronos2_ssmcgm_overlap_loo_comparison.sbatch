#!/bin/bash
#SBATCH --job-name=chronos2_ssmcgm_overlap_loo_comparison
#SBATCH --partition=normal
#SBATCH --qos=high_p
#SBATCH --time=01:00:00
#SBATCH --mem=16GB
#SBATCH --cpus-per-task=1
#SBATCH --output=/scratch/users/%u/logs/%x_%j.out
#SBATCH --error=/scratch/users/%u/logs/%x_%j.err

set -euo pipefail
REPO_ROOT=${AIREADI_ROOT:?set AIREADI_ROOT to your aireadi-collab checkout}
ml load math py-pytorch/2.9.1_py314 py-transformers/4.57.3_py314 py-einops/0.8.1_py314 py-pandas/2.3.3_py314 py-pyarrow/22.0.0_py314
source "$REPO_ROOT/venv-chronos/bin/activate"

# 6 arms x 5 horizons x 3 references = 90 rows. Walltime/mem well above the
# ~8 min the equivalent 140-row zeroshot_matched_comparison_all run took at
# 16GB, per the standing rule about doubling estimates.
python "$REPO_ROOT/code/analysis/chronos2_ssmcgm_overlap_loo_comparison.py"
