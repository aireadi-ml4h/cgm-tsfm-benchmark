"""Seed manifest written by the trained baselines when SAMPLE_SEED or
TRAIN_SEED is set explicitly (see code/seed_stability/README.md).

Records the participant sample and a hash of the early-stopping holdout
indices, so check_seed_manifests.py can verify that arm A ("init only") keeps
the sample and holdout fixed and arm B ("full resample") changes them. Default
runs (neither env var set) never call this, so their output directories are
unchanged.
"""
import hashlib
import json
import os

import numpy as np


def seed_env_set():
    return "SAMPLE_SEED" in os.environ or "TRAIN_SEED" in os.environ


def write_seed_manifest(out_dir, sample_seed, train_seed, sample_pids, holdout_idx=None, n_rows=None):
    m = {"sample_seed": int(sample_seed), "train_seed": int(train_seed),
         "sample_pids": [str(p) for p in sample_pids]}
    if holdout_idx is not None:
        h = np.asarray(holdout_idx, dtype=np.int64)
        m.update({"n_rows": int(n_rows), "n_holdout": int(len(h)),
                  "holdout_sha256": hashlib.sha256(h.tobytes()).hexdigest()})
    with open(os.path.join(out_dir, "seed_manifest.json"), "w") as f:
        json.dump(m, f, indent=1)
