#!/bin/bash
#SBATCH --job-name=chronos2_pers_report
#SBATCH --partition=normal
# TWO PARTITIONS, and Slurm starts this wherever a slot frees first. This is a
# CPU-only job, so the public `normal` partition is a real option for it --
# the owner partition's 28 GPUs are the scarce resource and its CPU queue sits behind
# them, while `normal` has ~700 idle cores and PreemptMode=OFF. Note this is
# the `normal` PARTITION, which is a different thing from the `normal` QOS on
# the line below; both happen to be named normal.
#
# Do NOT copy the multi-partition list to the GPU arm scripts: `normal` has
# zero GPUs, and the only GPU-bearing alternative (`owners`) is
# PreemptMode=REQUEUE with GraceTime=0.
#
# QOS: normal. MaxJobsPU=4 on high_p is shared across ALL your high_p jobs, so
# submitting this tiny CPU job under high_p would either be blocked by, or
# block, one of the GPU arm tasks. It is a 7-minute pandas job -- it does not
# need a priority boost, it needs to not consume a slot.
#SBATCH --qos=normal
#SBATCH --time=01:00:00
# 2 CPUs / 16GB deliberately. Slurm enforces an 8GB/core cap in the owner partition, so
# asking for 64GB on 4 CPUs gets silently bumped to 9 CPUs and then waits on
# (Priority) for a wider slot than the job ever needed. Measured peak is well
# under 16GB -- this reads prediction CSVs per arm, not the model.
#SBATCH --cpus-per-task=2
#SBATCH --mem=16GB
#SBATCH --output=logs/%x_%A.out

# TEST-SPLIT REPORTING DRIVER.
#
# Runs every comparison in code/analysis/chronos2_personalization_test_report.py's
# COMPARISONS list -- phase 1 (history length + negative control), phase 2
# primary (covariate arms at bar=7), and the bar=3 parallel family -- and
# writes chronos2_personalization_results/test_report.csv.
#
# SAFE TO RUN WHILE ARMS ARE STILL IN FLIGHT. Any comparison whose arms are
# not fully fitted is labelled "INCOMPLETE (n/expected participants fitted)"
# and gets no statistical verdict. That gate exists because an earlier run
# emitted a confident-looking "null, UNDERPOWERED" row computed from the 5
# participants a mid-run arm happened to have on disk.
#
# Comparisons are made with chronos2_matched_comparison.compare_arms() and
# restrict_keys, never by diffing summary.csv -- arms have different window
# coverage and a summary diff would silently compare different window sets.

set -euo pipefail
REPO_ROOT=${AIREADI_ROOT:?set AIREADI_ROOT to your aireadi-collab checkout}
ml load math py-pandas/2.3.3_py314 py-scipy/1.16.3_py314 py-pyarrow/22.0.0_py314

# venv-chronos, NOT the repo's `venv`, and activating it is not optional.
#
# sbatch defaults to --export=ALL, so this job inherits the login shell's
# environment, and ~/.bashrc auto-activates `venv` -- which is built against
# python/3.12.1. Loading the py314 modules above unloads python/3.12.1 and
# takes libpython3.12.so.1.0 with it, so that inherited `python` is left first
# on PATH and broken (job 42086965: "error while loading shared libraries:
# libpython3.12.so.1.0"). Stripping the venv instead is NOT the fix -- bare
# `python` then falls through to /usr/bin/python, which is 2.7.5 here, and
# pandas dies on an f-string (job 42088561: "SyntaxError: invalid syntax").
#
# venv-chronos is built against python/3.14.2 and so matches the _py314
# modules. Activating it puts a working `python` first on PATH. Module
# PYTHONPATH takes precedence over venv site-packages on the cluster, which is
# what supplies pandas/scipy/pyarrow here.
source "$REPO_ROOT/venv-chronos/bin/activate"

# Fail loudly here rather than 200 lines into pandas if the toolchain is wrong.
python -c "import sys, pandas, scipy; print('python', sys.version.split()[0], '| pandas', pandas.__version__, '| scipy', scipy.__version__)"

python "$REPO_ROOT/code/analysis/chronos2_personalization_test_report.py"
