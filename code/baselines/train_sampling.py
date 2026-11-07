"""Training-participant sampling shared by the four trained baselines
(lightgbm_zeroshot_baseline.py, extra_baselines_v2.py [MLP],
lstm_zeroshot_baseline.py, mamba_zeroshot_baseline.py).

TRAIN_SAMPLE_MODE (env), added 2026-11-03 for the FMTS camera-ready
(code/training_size_curve/README.md):
  random      (default) rng.choice over the whole train pool -- the original
              call, byte-for-byte. When the caller passes its own rng (LSTM/Mamba
              reuse it afterwards for the early-stopping holdout), that rng is
              used, so the stream after sampling is also unchanged.
  stratified  keep the train split's study-group proportions at every N.
  balanced    N/4 participants from each study group.
stratified/balanced are NESTED: each group is permuted once by its own rng
seeded from (seed, group index), and a sample takes the first n_g of each
permutation, so a smaller N is a subset of a larger N at the same seed.
"""
import os
import sys

import numpy as np

TRAIN_SAMPLE_MODE = os.environ.get("TRAIN_SAMPLE_MODE", "random")


def stratified_counts(group_sizes, n, mode):
    """Per-group sample sizes summing to n. stratified: largest-remainder
    apportionment to the group shares, floor 1 per group. balanced: n/k each."""
    groups = sorted(group_sizes)
    total = sum(group_sizes.values())
    if mode == "balanced":
        assert n % len(groups) == 0, f"balanced needs n divisible by {len(groups)}"
        per = n // len(groups)
        assert per <= min(group_sizes.values()), f"balanced n/{len(groups)}={per} exceeds smallest group"
        return {g: per for g in groups}
    if n >= total:
        return dict(group_sizes)
    raw = {g: n * group_sizes[g] / total for g in groups}
    counts = {g: max(1, int(np.floor(raw[g]))) for g in groups}
    order = sorted(groups, key=lambda g: raw[g] - np.floor(raw[g]), reverse=True)
    i = 0
    while sum(counts.values()) < n:
        g = order[i % len(order)]
        if counts[g] < group_sizes[g]:
            counts[g] += 1
        i += 1
    while sum(counts.values()) > n:  # only if the floor of 1 overshot
        g = max(groups, key=lambda g: counts[g])
        counts[g] -= 1
    return counts


def sample_train_participants(train_pids_all, widx, n, seed, mode=None, rng=None):
    mode = mode or TRAIN_SAMPLE_MODE
    if mode == "random":
        rng = rng if rng is not None else np.random.default_rng(seed)
        return rng.choice(train_pids_all, size=min(n, len(train_pids_all)), replace=False)
    if mode not in ("stratified", "balanced"):
        raise SystemExit(f"TRAIN_SAMPLE_MODE must be random|stratified|balanced, got {mode!r}")
    grp = (widx.drop_duplicates("person_id").set_index("person_id")["study_group"]
           .loc[list(train_pids_all)])
    groups = sorted(grp.unique())
    counts = stratified_counts(grp.value_counts().to_dict(), n, mode)
    picked = []
    for gi, g in enumerate(groups):
        pids = np.array(sorted(grp.index[grp == g]))
        perm = np.random.default_rng([seed, gi]).permutation(pids)
        picked.extend(perm[:counts[g]])
    print(f"TRAIN_SAMPLE_MODE={mode} n={len(picked)} counts={counts}", file=sys.stderr, flush=True)
    return np.array(picked)
