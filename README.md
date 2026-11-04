# Benchmarking and Personalizing Time-Series Foundation Models for Continuous Glucose Forecasting

Code accompanying the ML4H 2026 Findings Track submission of the same title.

We benchmark two general-purpose time-series foundation models (TSFMs) —
**Chronos-2** and **TimesFM 2.5** — against two CGM-specific models
(**CGM-LSM**, **SSM-CGM**), four trained-from-scratch baselines (LSTM, Mamba,
LightGBM, MLP), and three naive baselines (persistence, linear extrapolation,
circadian) on the AI-READI cohort, at forecast horizons from 30 minutes to
4 hours. We then test two adaptation mechanisms: wearable covariates, and
personalization on an individual's recent history (in-context vs. LoRA vs.
full fine-tuning).

## Use of this code

Copyright (c) 2026 The Authors. All rights reserved.

This code is published so that the analyses in the accompanying paper can be
inspected and verified. No license to use, copy, modify, or redistribute it is
granted at this time; a license will be added on publication.

## Data availability

This repository is code only.

The AI-READI dataset is available under a data use agreement at
<https://fairhub.io/>. The auxiliary meal-probability detector was trained on
CGMacros, which is obtained separately. Running these scripts requires your own
copy of both datasets under their respective terms.

## Setup

These scripts were written for a Slurm HPC cluster with an Lmod module system.
Absolute paths appear as placeholders: point `AIREADI_ROOT` at your checkout for
the Slurm scripts, and the `REPO_ROOT` constant at the top of each Python file.
Point `AIREADI_SCRATCH` at a directory with room for large files: the Hugging
Face cache, CGM-LSM checkpoints and the CGMacros download. Submit jobs from the
repository root; Slurm writes job logs to `logs/`, which must exist first.

```bash
export AIREADI_ROOT=/your/path/to/this/checkout
export AIREADI_SCRATCH=/your/path/for/large/files
git submodule update --init --recursive
mkdir -p logs
sbatch slurm/chronos2_zeroshot_arms.sbatch
```

The four upstream model repositories are pinned as submodules at the commits
evaluated in the paper: `chronos-forecasting` at `7dc4435`, `timesfm` at
`3dae50b`, `SSM-CGM` at `16b5a3d`, `cgmlsm` at `ad32a6c`. Published checkpoints
are `amazon/chronos-2` (revision `29ec3766`) and
`google/timesfm-2.5-200m-pytorch`; SSM-CGM and CGM-LSM are trained here.

Five virtual environments are used, since the four models have incompatible
dependencies. `requirements/*.txt` records each one; the scientific stack comes
from cluster modules, which the `ml load` line at the top of each `.sbatch`
file specifies exactly.

| Environment | Python | Used by |
|---|---|---|
| `venv-chronos` | 3.14.2 | Chronos-2 |
| `venv-timesfm` | 3.14.2 | TimesFM 2.5 |
| `venv-ssmcgm` | 3.12.1 | SSM-CGM |
| `venv-cgmlsm` | 3.12.1 | CGM-LSM |
| `venv` | 3.12.1 | window export, baselines, statistics |

## Layout

```
code/
  chronos2/         Chronos-2 zero-shot evaluation and personalization
  timesfm/          TimesFM 2.5 zero-shot evaluation and personalization
  cgmlsm/           CGM-LSM pretraining, fine-tuning, evaluation
  ssmcgm/           SSM-CGM training, evaluation, variable importance
  baselines/        LSTM, Mamba, LightGBM, MLP, and naive baselines
  analysis/         matched comparisons, significance tests, severity strata
  model_input/      eligible-window construction and per-model input builders
  meal_detection/   CNN-BiLSTM meal detector, trained on CGMacros
  external/         the four upstream model repositories, as submodules
slurm/              one job script per experiment arm
requirements/       pip freeze per environment
```

Scripts take the cleaned analysis dataframe as their starting point — a
five-minute UTC grid of glucose and wearable channels. The preprocessing rules
that produce it, and the selected hyperparameters, are specified in the paper's
appendix.

Where a family of arms differs only in which covariates are supplied, the arms
are consolidated into one array job with the per-arm configuration documented at
the top of the file (`*_arms.sbatch`). Run a single arm with
`sbatch --array=<n> <script>`.

Most scripts carry a module docstring explaining the design decisions behind
them; those are the best entry point for a given experiment.

## Evaluation protocol

- AI-READI v3, official recommended participant split (1,576/352/352); 291 test
  participants contribute at least one eligible window.
- 12 h of context (144 five-minute bins) for every model. Horizons 30 min, 1 h,
  2 h for cross-model comparison; Chronos-2 and TimesFM additionally at 3 h
  and 4 h.
- Glucose is never imputed. Each required channel must have ≤5% missingness and
  no gap longer than 60 min, assessed separately over context and horizon.
- MAE in mg/dL pooled over observed forecast bins. Pairwise 95% CIs from
  one-sample t intervals over participant-level mean loss differences.
- Hyperparameters and model selection used validation data only. Every
  configuration was run once with seed 42.

## Anonymization

This code is released for anonymous review, so paths appear as placeholders and
some comments refer to "a collaborator" or "the owner partition" where a name
would otherwise appear.
