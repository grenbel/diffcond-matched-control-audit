#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Stage-2.5: offline precompute of per-sample proxy difficulty for PCN.

Rationale:
  PCN has no native per-sample crop-ratio, so we cannot use an oracle difficulty
  on PCN. A median-constant fallback would be a degenerate negative control
  ("DiffCond without signal degenerates" is tautological and does not test
  DiffCond). We instead define a pre-specified, model-independent, objective
  geometry proxy. This script implements that.

Proxy definition:
  difficulty_proxy(sample) = 95-th percentile of NN-distance(complete -> partial)
  i.e. for each point in the complete point cloud, find its nearest-neighbor
  distance in the partial cloud; take the 95-th percentile. A higher value
  means more of the complete shape is far from the partial observation, i.e.
  the partial is a poorer covering of the complete -> harder to reconstruct.

Binning:
  - Compute the above distance for every PCN train sample (all 8 renderings
    per model), collect the distribution.
  - Thresholds = train-set 33.3% and 66.7% percentiles (tertile bins).
  - Apply the SAME thresholds to the test set (pre-specified, no snooping).

Output:
  data/PCN/difficulty_proxy.json = {
    "meta": {...},
    "thresholds_from_train": [t_low, t_high],
    "distance": {"train": {...}, "test": {...}},
    "difficulty": {"train": {...}, "test": {...}},
  }

Key format:
  train: "<taxonomy_id>/<model_id>/<rand_idx:02d>"  (8 renderings per model)
  test:  "<taxonomy_id>/<model_id>"                 (only rand_idx=0 for test)

Usage (on a GPU machine, from the repository root):
  python tools/compute_pcn_proxy_difficulty.py \
      [--config cfgs/dataset_configs/PCN.yaml] \
      [--out data/PCN/difficulty_proxy.json] \
      [--percentile 95] [--device cuda] [--limit N] [--subsets train,test]

Expected runtime: ~5-10 min on RTX 5090 for full PCN (~225K train + 1200 test).
"""

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import yaml
from easydict import EasyDict as edict

# Import IO for pcd loading + PCNDataset for file list
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from datasets.io import IO  # noqa: E402


def load_pcd(path):
    """Load a point cloud file, return (N, 3) float32 numpy array."""
    return IO.get(path).astype(np.float32)


def nn_dist_p95(partial_np, complete_np, device, percentile=95):
    """95-th percentile NN distance complete -> partial.
    partial_np: (M, 3), complete_np: (N, 3).
    Returns: scalar float.
    """
    p = torch.from_numpy(partial_np).to(device)      # (M, 3)
    c = torch.from_numpy(complete_np).to(device)     # (N, 3)
    # cdist(c, p): (N, M)
    d = torch.cdist(c.unsqueeze(0), p.unsqueeze(0))[0]
    # For each complete point, min distance to partial
    min_d = d.min(dim=1).values                      # (N,)
    # percentile over N
    return torch.quantile(min_d, percentile / 100.0).item()


def build_pcn_file_list(config_path, subset):
    """Read PCN dataset config + category JSON, enumerate all (taxonomy, model, rand_idx)."""
    with open(config_path, 'r') as f:
        base = edict(yaml.safe_load(f))
    with open(base.CATEGORY_FILE_PATH, 'r') as f:
        cats = json.loads(f.read())
    n_renderings = 8 if subset == 'train' else 1
    samples = []
    for dc in cats:
        tax = dc['taxonomy_id']
        for m in dc[subset]:
            for ri in range(n_renderings):
                partial_path = base.PARTIAL_POINTS_PATH % (subset, tax, m, ri)
                gt_path = base.COMPLETE_POINTS_PATH % (subset, tax, m)
                key = f'{tax}/{m}' if n_renderings == 1 else f'{tax}/{m}/{ri:02d}'
                samples.append((key, partial_path, gt_path))
    return samples, base


def compute_subset(samples, device, percentile, limit=None, log_every=2000):
    """Iterate samples, compute NN-dist per sample. Returns dict {key: float}."""
    result = {}
    errors = defaultdict(int)
    t0 = time.time()
    n_total = len(samples) if limit is None else min(limit, len(samples))
    for i, (key, partial_path, gt_path) in enumerate(samples):
        if limit is not None and i >= limit:
            break
        try:
            p = load_pcd(partial_path)
            g = load_pcd(gt_path)
        except Exception as e:
            errors[type(e).__name__] += 1
            continue
        # Some renderings have <10 points due to occlusion; skip if degenerate
        if p.shape[0] < 10 or g.shape[0] < 10:
            errors['too_few_points'] += 1
            continue
        try:
            d = nn_dist_p95(p, g, device, percentile=percentile)
        except Exception as e:
            errors[f'gpu_{type(e).__name__}'] += 1
            continue
        result[key] = d
        if (i + 1) % log_every == 0 or (i + 1) == n_total:
            elapsed = time.time() - t0
            rate = (i + 1) / max(elapsed, 1e-6)
            eta = (n_total - i - 1) / max(rate, 1e-6)
            print(f'  [{i+1}/{n_total}] rate={rate:.1f}/s  ETA={eta/60:.1f}min', flush=True)
    elapsed = time.time() - t0
    print(f'  done in {elapsed/60:.1f}min  '
          f'({len(result)}/{n_total} ok, errors={dict(errors) if errors else "none"})')
    return result, dict(errors)


def assign_tertile(values_dict, t_low, t_high):
    """Map each sample value to a tertile index {0=easy, 1=median, 2=hard}."""
    out = {}
    for k, v in values_dict.items():
        if v < t_low:
            out[k] = 0
        elif v < t_high:
            out[k] = 1
        else:
            out[k] = 2
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default='cfgs/dataset_configs/PCN.yaml',
                    help='PCN dataset base config')
    ap.add_argument('--out', default='data/PCN/difficulty_proxy.json')
    ap.add_argument('--percentile', type=float, default=95.0,
                    help='Percentile for NN-distance metric (default 95)')
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--limit', type=int, default=None,
                    help='Process at most this many samples per subset (for smoke testing)')
    ap.add_argument('--subsets', default='train,test',
                    help='comma-separated subsets to process')
    args = ap.parse_args()

    device = torch.device(args.device if args.device == 'cpu' or torch.cuda.is_available()
                          else 'cpu')
    if args.device == 'cuda' and device.type == 'cpu':
        print('WARNING: cuda requested but not available; using CPU (will be SLOW)')

    subsets = [s.strip() for s in args.subsets.split(',') if s.strip()]
    out_dir = os.path.dirname(args.out) or '.'
    os.makedirs(out_dir, exist_ok=True)

    print(f'=== Stage-2.5: PCN proxy difficulty precompute ===')
    print(f'  config     : {args.config}')
    print(f'  out        : {args.out}')
    print(f'  percentile : {args.percentile}')
    print(f'  device     : {device}')
    print(f'  subsets    : {subsets}')
    if args.limit is not None:
        print(f'  limit      : {args.limit} samples per subset (SMOKE MODE)')
    print()

    # 1) Enumerate + compute each subset
    per_subset_distances = {}
    per_subset_errors = {}
    for subset in subsets:
        print(f'--- Subset: {subset} ---')
        samples, base = build_pcn_file_list(args.config, subset)
        print(f'  enumerated {len(samples)} sample entries')
        dists, errs = compute_subset(samples, device, args.percentile, args.limit)
        per_subset_distances[subset] = dists
        per_subset_errors[subset] = errs
        print()

    # 2) Compute tertile thresholds from TRAIN only (pre-specified, no snooping).
    # Require train subset present so thresholds are never 0/0
    # (which would make test labels meaningless).
    if 'train' not in per_subset_distances or not per_subset_distances['train']:
        raise RuntimeError(
            "Train subset is required to compute tertile thresholds. "
            "Either include 'train' in --subsets (default) or provide "
            "pre-computed thresholds via a follow-up feature."
        )
    train_vals = np.array(list(per_subset_distances['train'].values()), dtype=np.float64)
    t_low = float(np.percentile(train_vals, 100.0 / 3))
    t_high = float(np.percentile(train_vals, 200.0 / 3))
    n_train = len(train_vals)
    print(f'--- Tertile thresholds (train-only, n={n_train}) ---')
    print(f'  min    = {train_vals.min():.5f}')
    print(f'  p33.3  = {t_low:.5f}')
    print(f'  median = {np.median(train_vals):.5f}')
    print(f'  p66.7  = {t_high:.5f}')
    print(f'  max    = {train_vals.max():.5f}')
    print(f'  mean   = {train_vals.mean():.5f}')
    print(f'  std    = {train_vals.std():.5f}')

    # 3) Assign difficulty bins to all subsets using the train thresholds
    per_subset_difficulty = {}
    for subset, dists in per_subset_distances.items():
        diff = assign_tertile(dists, t_low, t_high)
        per_subset_difficulty[subset] = diff
        # Quick bin-size report
        counts = [sum(1 for v in diff.values() if v == k) for k in (0, 1, 2)]
        total = sum(counts)
        if total > 0:
            pct = [100 * c / total for c in counts]
            print(f'  {subset}: easy={counts[0]} ({pct[0]:.1f}%)  '
                  f'median={counts[1]} ({pct[1]:.1f}%)  '
                  f'hard={counts[2]} ({pct[2]:.1f}%)  total={total}')

    # 4) Save JSON
    output = {
        'meta': {
            'percentile': args.percentile,
            'metric': 'NN-distance complete->partial, p95',
            'binning': 'train-set tertile thresholds',
            'source_config': args.config,
            'device': str(device),
            'generated_at': time.strftime('%Y-%m-%d %H:%M:%S'),
        },
        'thresholds_from_train': [t_low, t_high],
        'distance': per_subset_distances,
        'difficulty': per_subset_difficulty,
        'errors': per_subset_errors,
    }
    with open(args.out, 'w') as f:
        json.dump(output, f, indent=1, sort_keys=True)
    total_entries = sum(len(d) for d in per_subset_difficulty.values())
    print()
    print(f'=== Saved: {args.out} ({total_entries} difficulty entries, '
          f'{os.path.getsize(args.out)/1e6:.2f} MB) ===')


if __name__ == '__main__':
    main()
