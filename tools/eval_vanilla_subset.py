#!/usr/bin/env python3
"""Evaluate a vanilla AdaPoinTr checkpoint on a controlled subset of
ShapeNet-55 test (mirroring eval_dtr.py's sampling) so we can compare it
apples-to-apples against the DiffCond DTR probe.

Usage:
    # Match eval_dtr.py --n-samples 100 --n-viewpoints 2 --modes hard
    python tools/eval_vanilla_subset.py \
        --config cfgs/ShapeNet55_models/AdaPoinTr_baseline_smoke30_diffcrop.yaml \
        --ckpt experiments/<vanilla-ft-run>/<fine-tuned-checkpoint>.pth \
        --n-samples 100 --n-viewpoints 2 --modes hard

Outputs category-balanced (F-Score, CDL1, CDL2, EMD) per mode, aggregated
the same way as eval_dtr.py so results are directly comparable.
"""
import argparse
import json
import os
import pathlib
import sys
from collections import defaultdict

import numpy as np
import torch

THIS_DIR = pathlib.Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parent
os.chdir(str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT))

from utils.config import cfg_from_yaml_file
from utils import misc
from utils.metrics import Metrics
from tools.builder import model_builder, dataset_builder, load_model

CROP_RATIO = {'easy': 1/4, 'median': 1/2, 'hard': 3/4}

VIEWPOINT_CHOICES = [
    torch.tensor([1.0, 1.0, 1.0]), torch.tensor([1.0, 1.0, -1.0]),
    torch.tensor([1.0, -1.0, 1.0]), torch.tensor([-1.0, 1.0, 1.0]),
    torch.tensor([-1.0, -1.0, 1.0]), torch.tensor([-1.0, 1.0, -1.0]),
    torch.tensor([1.0, -1.0, -1.0]), torch.tensor([-1.0, -1.0, -1.0]),
]


class _Args:
    """Minimal args object for builder compatibility."""
    def __init__(self, config_path, ckpt_path, exp_name='vanilla_subset'):
        self.config = config_path
        self.ckpts = ckpt_path
        self.exp_name = exp_name
        self.use_gpu = True
        self.distributed = False
        self.local_rank = 0
        self.num_workers = 4
        self.experiment_path = './experiments/vanilla_subset_tmp'
        self.tfboard_path = './experiments/vanilla_subset_tmp/tfboard'
        self.start_ckpts = None
        self.resume = False
        self.test = True
        self.mode = None
        os.makedirs(self.experiment_path, exist_ok=True)


def load_vanilla_model(config_path, ckpt_path):
    config = cfg_from_yaml_file(config_path)
    base_model = model_builder(config.model)
    load_model(base_model, ckpt_path, logger=None)
    base_model = base_model.cuda()
    base_model.eval()
    # Fail fast if caller passes a DiffCond ckpt by mistake: this script is
    # specifically for the Vanilla-FT (matched vanilla AdaPoinTr) same-subset control.
    has_diff = any('diff_cond' in n for n, _ in base_model.named_parameters())
    assert not has_diff, (
        'load_vanilla_model expected a vanilla AdaPoinTr checkpoint but '
        f'found diff_cond_* modules in {ckpt_path}. Use eval_dtr.py for DiffCond.'
    )
    print('  Sanity: no diff_cond_* modules present (vanilla ckpt confirmed).')
    return base_model, config


def evaluate_one_mode(model, dataloader, npoints, mode, viewpoints,
                      n_samples_max, require_emd=True):
    """Category-balanced eval on subset. Mirrors eval_dtr.py's aggregation."""
    num_crop = int(npoints * CROP_RATIO[mode])
    print(f'\n[Mode={mode}] num_crop={num_crop} viewpoints={len(viewpoints)} '
          f'n_samples_max={n_samples_max} require_emd={require_emd}')

    # category_accums[taxonomy_id]['fscore'|'cdl1'|'cdl2'|'emd'] = list of values
    category_accums = defaultdict(
        lambda: {'fscore': [], 'cdl1': [], 'cdl2': [], 'emd': []}
    )
    sample_count = 0
    consumed_samples = []  # (taxonomy_id, model_id) tuples -- subset identity artifact

    with torch.no_grad():
        for batch_idx, (taxonomy_ids, model_ids, data) in enumerate(dataloader):
            gt = data.cuda()
            B = gt.size(0)
            # Aggregation currently assumes B=1 (ShapeNet-55 test loader fixes this,
            # see builder.py:26). Fail fast if this invariant ever changes.
            assert B == 1, (
                f'eval_vanilla_subset.py assumes test batch size 1, got B={B}. '
                'Aggregation logic would silently misattribute metrics across '
                'taxonomies; revisit before relaxing this.'
            )
            taxonomy_id = taxonomy_ids[0] if isinstance(taxonomy_ids[0], str) else taxonomy_ids[0].item()
            model_id = model_ids[0] if isinstance(model_ids[0], str) else model_ids[0]

            if n_samples_max is not None and sample_count >= n_samples_max:
                break

            consumed_samples.append((taxonomy_id, model_id))

            for fp in viewpoints:
                partial, _ = misc.seprate_point_cloud(
                    gt, npoints, num_crop, fixed_points=fp)
                partial = misc.fps(partial, 2048).cuda()

                # Vanilla AdaPoinTr forward (no difficulty kwarg)
                ret = model(partial)
                dense = ret[1]  # (B, num_query*factor, 3)

                m = Metrics.get(dense, gt, require_emd=require_emd)
                m_vals = [x.item() for x in m]
                cat_acc = category_accums[taxonomy_id]
                cat_acc['fscore'].append(m_vals[0])
                cat_acc['cdl1'].append(m_vals[1])
                cat_acc['cdl2'].append(m_vals[2])
                cat_acc['emd'].append(m_vals[3])

            sample_count += B
            if batch_idx % 20 == 0:
                print(f'  [{mode}] processed {sample_count} samples '
                      f'({len(category_accums)} categories)', flush=True)

    # Print first 5 consumed samples for provenance transparency
    if consumed_samples:
        print(f'  [{mode}] first 5 samples consumed:')
        for i, (tax, mid) in enumerate(consumed_samples[:5]):
            print(f'    {i+1}. {tax} / {mid}')

    # Category-balanced aggregation
    per_category_means = {'fscore': [], 'cdl1': [], 'cdl2': [], 'emd': []}
    for tax_id, acc in category_accums.items():
        for metric_name in ['fscore', 'cdl1', 'cdl2', 'emd']:
            values = acc[metric_name]
            if values:
                per_category_means[metric_name].append(float(np.mean(values)))
    summary = {
        'fscore': float(np.mean(per_category_means['fscore'])),
        'cdl1': float(np.mean(per_category_means['cdl1'])),
        'cdl2': float(np.mean(per_category_means['cdl2'])),
        'emd': float(np.mean(per_category_means['emd'])),
        'n_categories': len(category_accums),
        'n_total_samples': sample_count,
        'consumed_samples': [{'taxonomy_id': t, 'model_id': m}
                             for t, m in consumed_samples],
    }
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True,
                        help='Path to vanilla AdaPoinTr yaml config')
    parser.add_argument('--ckpt', required=True,
                        help='Path to ckpt to evaluate')
    parser.add_argument('--n-samples', type=int, default=None,
                        help='Limit number of test samples (default: all 10518)')
    parser.add_argument('--n-viewpoints', type=int, default=8,
                        help='Number of viewpoints 1-8 (default 8)')
    parser.add_argument('--modes', nargs='+', default=['hard'],
                        choices=['easy', 'median', 'hard'],
                        help='Which difficulty modes to evaluate')
    parser.add_argument('--output', default='/tmp/vanilla_subset_results.json')
    parser.add_argument('--no-emd', action='store_true',
                        help='Skip EMD compute (faster but EMD column will be 0)')
    args = parser.parse_args()

    if not os.path.isfile(args.config):
        print(f'ERROR: Config not found: {args.config}'); sys.exit(1)
    if not os.path.isfile(args.ckpt):
        print(f'ERROR: Ckpt not found: {args.ckpt}'); sys.exit(1)

    print(f'Config: {args.config}')
    print(f'Ckpt:   {args.ckpt}')
    print(f'Modes:  {args.modes}')
    print(f'Viewpoints: first {args.n_viewpoints} of 8')
    print(f'N samples:  {args.n_samples or "all"}')
    print(f'Output: {args.output}')

    viewpoints = VIEWPOINT_CHOICES[:args.n_viewpoints]

    builder_args = _Args(args.config, args.ckpt)

    print('\nLoading vanilla AdaPoinTr model...')
    model, config = load_vanilla_model(args.config, args.ckpt)

    print('Building test dataloader...')
    _, test_dataloader = dataset_builder(builder_args, config.dataset.test)
    npoints = config.dataset.test._base_.N_POINTS

    per_mode = {}
    for mode in args.modes:
        per_mode[mode] = evaluate_one_mode(
            model, test_dataloader, npoints, mode, viewpoints,
            args.n_samples, require_emd=not args.no_emd)

    # Print summary
    print('\n' + '=' * 72)
    print('  Vanilla AdaPoinTr Subset Eval (category-balanced)')
    print('=' * 72)
    print(f'  {"Mode":<8} {"F-Score":>9} {"CDL1":>8} {"CDL2":>8} {"EMD":>8}  n_cat n_samp')
    print(f'  {"-"*8} {"-"*9} {"-"*8} {"-"*8} {"-"*8}  ----- ------')
    for mode, m in per_mode.items():
        print(f'  {mode:<8} {m["fscore"]:>9.4f} {m["cdl1"]:>8.3f} '
              f'{m["cdl2"]:>8.3f} {m["emd"]:>8.4f}  {m["n_categories"]:5d} {m["n_total_samples"]:6d}')

    with open(args.output, 'w') as f:
        json.dump(per_mode, f, indent=2)
    print(f'\nSaved JSON to {args.output}')


if __name__ == '__main__':
    main()
