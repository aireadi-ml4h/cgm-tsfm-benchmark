#!/bin/bash
#SBATCH --job-name=ssmcgm_train_personalized
#SBATCH --partition=normal
#SBATCH --qos=high_p
#SBATCH --time=6-00:00:00
#SBATCH --mem=96GB
#SBATCH --cpus-per-task=8
#SBATCH -G 1
#SBATCH --output=logs/%x_%j.out

# Single GPU from the start (ssmcgm_train_real.py's real history: two
# distinct DDP failures before settling on single-GPU for reliability).
# MAX_EPOCHS=10/patience=6 (script defaults) are already lower than
# ssmcgm_train_real.py's 30/15, a direct response to that run reaching its
# best generalization at epoch 0 and overfitting from there -- 6 days is a
# safety margin against Slurm requeues (three real ones on the other run),
# not the expected runtime. Resume via SSMCGM_PERS_RESUME_CKPT=<output_dir>/
# last.ckpt if needed.
set -euo pipefail
REPO_ROOT=${AIREADI_ROOT:?set AIREADI_ROOT to your aireadi-collab checkout}
ml load math py-pytorch/2.4.1_py312
ml load devel py-pandas/2.2.1_py312 py-pyarrow/18.1.0_py312
source "$REPO_ROOT/venv-ssmcgm/bin/activate"

export SSMCGM_PERS_NUM_GPUS=1
# SSMCGM_PERS_RESUME_CKPT intentionally left unset for a fresh run --
# os.environ.get() then returns None, which trainer.fit(ckpt_path=None)
# treats as "start fresh." An empty string here would NOT be equivalent
# (trainer.fit(ckpt_path="") is an invalid path, not "no resume").
python "$REPO_ROOT/code/ssmcgm/ssmcgm_train_personalized.py"
