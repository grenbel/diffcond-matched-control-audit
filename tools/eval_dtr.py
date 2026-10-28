#!/usr/bin/env python3
"""Difficulty-aware Test-time Refinement (DTR) Eval

Inference-time ensemble of multiple difficulty conditionings on a trained
DiffCond model. Tests whether ensembling outputs across difficulty levels
beats single-difficulty inference (oracle or predict).

For each (sample, viewpoint, oracle_mode), runs the model with 4
different difficulty conditionings, then aggregates via 5 ensemble
strategies. Per-strategy CDL1/CDL2/F-Score/EMD recorded.

Strategies tested:
  - oracle:    single-difficulty conditioning (oracle = mode label)
  - predict:   single-difficulty conditioning (predictor argmax)
  - mean:      simple mean of dense outputs across 3 difficulties
  - weighted:  softmax-weighted mean using predictor probs
  - oracle_best: choose per-sample best of 3 (upper-bound, requires gt)

Designed as a paper-Section contribution candidate (DTR fallback when
Phase 2.6 verdict is NO-GO/AMBIGUOUS).

Usage:
    # quick eval on 100 samples + 2 viewpoints (sanity)
    python tools/eval_dtr.py --n-samples 100 --n-viewpoints 2 --modes hard

    # full eval (single mode, all samples, all viewpoints) - ~2h
    python tools/eval_dtr.py --modes hard

    # full eval (all 3 modes) - ~6h
    python tools/eval_dtr.py
"""
import argparse
import json
import os
import pathlib
import sys
import numpy as np
import torch
import torch.nn.functional as F

THIS_DIR = pathlib.Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parent
os.chdir(str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT))

from utils.config import cfg_from_yaml_file
from utils import misc
from utils.AverageMeter import AverageMeter
from extensions.chamfer_dist import ChamferDistanceL1, ChamferDistanceL2
from utils.metrics import Metrics
from tools.builder import model_builder, dataset_builder, load_model
from models.AdaPoinTr_DiffCond import DIFF_LEVELS

CROP_RATIO = {'easy': 1/4, 'median': 1/2, 'hard': 3/4}

VIEWPOINT_CHOICES = [
    torch.tensor([1.0, 1.0, 1.0]), torch.tensor([1.0, 1.0, -1.0]),
    torch.tensor([1.0, -1.0, 1.0]), torch.tensor([-1.0, 1.0, 1.0]),
    torch.tensor([-1.0, -1.0, 1.0]), torch.tensor([-1.0, 1.0, -1.0]),
    torch.tensor([1.0, -1.0, -1.0]), torch.tensor([-1.0, -1.0, -1.0]),
]


class _Args:
    def __init__(self, config_path, ckpt_path):
        self.config = config_path
        self.ckpts = ckpt_path
        self.exp_name = 'dtr_eval'
        self.use_gpu = True
        self.distributed = False
        self.local_rank = 0
        self.num_workers = 4
        self.experiment_path = './experiments/dtr_tmp'
        self.tfboard_path = './experiments/dtr_tmp/tfboard'
        self.start_ckpts = None
        self.resume = False
        self.test = True
        self.mode = None
        self.difficulty_source = 'predict'
        os.makedirs(self.experiment_path, exist_ok=True)


def get_underlying(model):
    return model.module if hasattr(model, 'module') else model


def load_diffcond_model(config_path, ckpt_path, force_single_gpu=True):
    """Load DiffCond model. Force single GPU by default to avoid DataParallel
    side-effect attribute issues with the `_last_diff_pred_logits` cache.
    Set CUDA_VISIBLE_DEVICES=0 externally if needed."""
    config = cfg_from_yaml_file(config_path)
    base_model = model_builder(config.model)
    load_model(base_model, ckpt_path, logger=None)
    base_model = base_model.cuda()

    # Sanity check: predictor + diff_cond_proj weights should NOT be all-zero
    # (zero-init is the vanilla-warmstart fallback in builder.load_model).
    # A trained DiffCond ckpt MUST have non-zero values here.
    pctr = base_model.base_model
    pred_w = pctr.diff_predictor[0].weight
    proj_w = pctr.diff_cond_proj.weight
    pred_norm = pred_w.norm().item()
    proj_norm = proj_w.norm().item()
    print(f'  Sanity: predictor[0].weight L2={pred_norm:.4f} '
          f'(must be > 0 for trained ckpt)')
    print(f'  Sanity: diff_cond_proj.weight L2={proj_norm:.4f} '
          f'(must be > 0 for trained DiffCond ckpt)')
    assert pred_norm > 1e-3, \
        f'Predictor weight is essentially zero ({pred_norm}); ckpt load failed?'
    assert proj_norm > 1e-3, \
        f'diff_cond_proj weight is essentially zero ({proj_norm}); maybe loaded vanilla ckpt by mistake?'

    if not force_single_gpu and torch.cuda.device_count() > 1:
        base_model = torch.nn.DataParallel(base_model)
        print(f'  Using DataParallel across {torch.cuda.device_count()} GPUs')
    else:
        print(f'  Single-GPU mode (CUDA_VISIBLE_DEVICES={os.environ.get("CUDA_VISIBLE_DEVICES", "all")})')
    base_model.eval()
    return base_model, config


def run_dtr_for_partial(model, partial, gt):
    """Given partial (B, N, 3) and gt (B, M, 3), run all 4 difficulty conditionings
    once each, return dict of dense outputs per strategy."""
    B = partial.size(0)
    device = partial.device

    outputs = {}  # strategy -> dense (B, num_query*factor, 3)

    # Forward 3 times with oracle difficulty conditioning
    diff_outputs = {}
    for diff_str in ['easy', 'median', 'hard']:
        diff_tensor = torch.full((B,), DIFF_LEVELS[diff_str],
                                 dtype=torch.long, device=device)
        with torch.no_grad():
            ret = model(partial, difficulty=diff_tensor)
        # ret = (coarse_point_cloud, rebuild_points)
        diff_outputs[diff_str] = ret[1]  # dense
    outputs['cond_easy'] = diff_outputs['easy']
    outputs['cond_median'] = diff_outputs['median']
    outputs['cond_hard'] = diff_outputs['hard']

    # Forward with predict mode
    with torch.no_grad():
        ret = model(partial, difficulty='predict')
    outputs['predict'] = ret[1]
    pred_logits = get_underlying(model)._last_diff_pred_logits  # (B, 3)
    pred_probs = F.softmax(pred_logits, dim=-1)  # (B, 3)

    # Mean ensemble
    outputs['mean'] = (diff_outputs['easy'] + diff_outputs['median']
                      + diff_outputs['hard']) / 3.0

    # Predictor-weighted ensemble
    # weight broadcast (B,) -> (B, 1, 1)
    we = pred_probs[:, 0].view(-1, 1, 1)
    wm = pred_probs[:, 1].view(-1, 1, 1)
    wh = pred_probs[:, 2].view(-1, 1, 1)
    outputs['weighted'] = (we * diff_outputs['easy']
                         + wm * diff_outputs['median']
                         + wh * diff_outputs['hard'])

    return outputs


def cdl2_per_sample(pred, gt, chamfer_l2):
    """Per-sample CDL2 (compute one at a time so we can pick best per sample)."""
    B = pred.size(0)
    losses = []
    for i in range(B):
        d = chamfer_l2(pred[i:i+1], gt[i:i+1]).item()
        losses.append(d)
    return torch.tensor(losses, device=pred.device)


def evaluate_strategies(strategies_outputs, gt, chamfer_l1, chamfer_l2, metrics_fn):
    """For each strategy, compute (F-Score, CDL1, CDL2, EMD) on (B, *) batch.
    Returns dict strategy -> dict of mean metrics + num_samples."""
    results = {}
    for name, dense in strategies_outputs.items():
        # Aggregate metrics across batch
        cd_l1 = chamfer_l1(dense, gt).item() * 1000
        cd_l2 = chamfer_l2(dense, gt).item() * 1000
        m = metrics_fn(dense, gt)  # returns list/tensor of [F-Score, CDL1, CDL2, EMD]
        m_vals = [x.item() for x in m]
        results[name] = {
            'fscore': m_vals[0],
            'cdl1': m_vals[1],
            'cdl2': m_vals[2],
            'emd': m_vals[3],
            'cd_l1_avg': cd_l1,
            'cd_l2_avg': cd_l2,
        }
    return results


def evaluate_one_mode(model, dataloader, npoints, mode, viewpoints,
                      n_samples_max, chamfer_l1, chamfer_l2, require_emd=True):
    """Evaluate all DTR strategies for a single difficulty mode.

    Uses category-balanced aggregation matching official ShapeNet-55 protocol
    (runner.py:506-531): per-category mean over samples first, then mean across
    categories (NOT raw sample mean). 55 ShapeNet-55 categories have very
    skewed counts (12-1687 samples each) so raw sample mean would be biased
    toward dense categories..

    EMD is computed via require_emd=True; without that flag Metrics.get
    returns 0 for EMD (utils/metrics.py:50)..

    `oracle_best_cdl2` strategy picks per-sample best of (easy/median/hard) by
    CDL2 only; F-Score/CDL1/EMD are NOT guaranteed upper bounds for it..
    """
    num_crop = int(npoints * CROP_RATIO[mode])
    print(f'\n[Mode={mode}] num_crop={num_crop} viewpoints={len(viewpoints)} '
          f'require_emd={require_emd}')

    strategy_names = ['cond_easy', 'cond_median', 'cond_hard',
                      'predict', 'mean', 'weighted', 'oracle_best_cdl2']

    # Category-balanced accumulators: per-category AverageMeter-like state
    # category_accums[taxonomy_id][strategy] = list of [F, CDL1, CDL2, EMD] tuples
    from collections import defaultdict
    category_accums = defaultdict(
        lambda: {s: {'fscore': [], 'cdl1': [], 'cdl2': [], 'emd': []}
                 for s in strategy_names}
    )
    sample_count = 0

    with torch.no_grad():
        for batch_idx, (taxonomy_ids, model_ids, data) in enumerate(dataloader):
            gt = data.cuda()
            B = gt.size(0)
            # bs=1 for ShapeNet-55 test (builder.py:14-31), so B=1 here
            taxonomy_id = taxonomy_ids[0] if isinstance(taxonomy_ids[0], str) else taxonomy_ids[0].item()

            if n_samples_max is not None and sample_count >= n_samples_max:
                break

            for fp in viewpoints:
                partial, _ = misc.seprate_point_cloud(
                    gt, npoints, num_crop, fixed_points=fp)
                partial = misc.fps(partial, 2048).cuda()

                strategies_outputs = run_dtr_for_partial(model, partial, gt)

                # oracle_best_cdl2: per-sample best of (easy, median, hard) by CDL2
                cdl2_easy = cdl2_per_sample(strategies_outputs['cond_easy'], gt, chamfer_l2)
                cdl2_median = cdl2_per_sample(strategies_outputs['cond_median'], gt, chamfer_l2)
                cdl2_hard = cdl2_per_sample(strategies_outputs['cond_hard'], gt, chamfer_l2)
                stacked = torch.stack([cdl2_easy, cdl2_median, cdl2_hard], dim=0)  # (3, B)
                best_idx = stacked.argmin(dim=0)  # (B,) which difficulty per sample
                oracle_best_dense = torch.zeros_like(strategies_outputs['cond_easy'])
                for i in range(B):
                    if best_idx[i] == 0:
                        oracle_best_dense[i] = strategies_outputs['cond_easy'][i]
                    elif best_idx[i] == 1:
                        oracle_best_dense[i] = strategies_outputs['cond_median'][i]
                    else:
                        oracle_best_dense[i] = strategies_outputs['cond_hard'][i]
                strategies_outputs['oracle_best_cdl2'] = oracle_best_dense

                # Compute metrics for all strategies on this batch+viewpoint
                # (with require_emd=True so EMD is real, not 0)
                for s in strategy_names:
                    dense = strategies_outputs[s]
                    m = Metrics.get(dense, gt, require_emd=require_emd)
                    m_vals = [x.item() for x in m]
                    cat_acc = category_accums[taxonomy_id][s]
                    cat_acc['fscore'].append(m_vals[0])
                    cat_acc['cdl1'].append(m_vals[1])
                    cat_acc['cdl2'].append(m_vals[2])
                    cat_acc['emd'].append(m_vals[3])

            sample_count += B
            if batch_idx % 20 == 0:
                print(f'  [{mode}] processed {sample_count} samples '
                      f'({len(category_accums)} categories so far)', flush=True)

    # Category-balanced aggregation:
    # Step 1: per-category mean over samples
    # Step 2: mean across categories
    summary = {}
    for s in strategy_names:
        per_category_means = {'fscore': [], 'cdl1': [], 'cdl2': [], 'emd': []}
        for tax_id, strat_dict in category_accums.items():
            for metric_name in ['fscore', 'cdl1', 'cdl2', 'emd']:
                values = strat_dict[s][metric_name]
                if values:
                    per_category_means[metric_name].append(float(np.mean(values)))
        summary[s] = {
            'fscore': float(np.mean(per_category_means['fscore'])),
            'cdl1': float(np.mean(per_category_means['cdl1'])),
            'cdl2': float(np.mean(per_category_means['cdl2'])),
            'emd': float(np.mean(per_category_means['emd'])),
            'n_categories': len(category_accums),
            'n_total_samples': sample_count,
        }
    return summary


def print_dtr_table(per_mode_summary):
    print('\n' + '=' * 96)
    print('  DTR Strategy Comparison (CDL2 lower=better, F-Score higher=better)')
    print('=' * 96)

    strategies_order = ['cond_easy', 'cond_median', 'cond_hard',
                        'predict', 'mean', 'weighted', 'oracle_best_cdl2']

    for mode, summary in per_mode_summary.items():
        print(f'\n  -- Mode: {mode} (n_categories={summary["cond_easy"]["n_categories"]}, '
              f'n_samples={summary["cond_easy"]["n_total_samples"]}) --')
        print(f'  {"Strategy":<18} {"F-Score":>9} {"CDL1":>8} {"CDL2":>8} {"EMD":>8}')
        print(f'  {"-"*18} {"-"*9} {"-"*8} {"-"*8} {"-"*8}')
        for s in strategies_order:
            if s not in summary:
                continue
            m = summary[s]
            print(f'  {s:<18} {m["fscore"]:>9.4f} {m["cdl1"]:>8.3f} '
                  f'{m["cdl2"]:>8.3f} {m["emd"]:>8.4f}')

        # Best non-oracle strategy by CDL2; oracle_best_cdl2 is upper bound for CDL2 only.
        non_oracle = {k: v for k, v in summary.items() if k != 'oracle_best_cdl2'}
        best_s = min(non_oracle.keys(), key=lambda k: non_oracle[k]['cdl2'])
        oracle_cdl2 = summary['oracle_best_cdl2']['cdl2']
        gap = summary[best_s]['cdl2'] - oracle_cdl2
        gap_pct = gap / summary[best_s]['cdl2'] * 100
        print(f'  Best non-oracle: {best_s} (CDL2 {summary[best_s]["cdl2"]:.3f})')
        print(f'  oracle_best_cdl2 (CDL2-only upper bound): {oracle_cdl2:.3f} '
              f'(gap from best non-oracle: {gap:+.3f}, {gap_pct:+.2f}%)')
    print('\n' + '=' * 96)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='cfgs/ShapeNet55_models/AdaPoinTr_DiffCond_smoke30.yaml')
    parser.add_argument('--ckpt', default='experiments/<diffcond-run>/ckpt-epoch-030.pth')
    parser.add_argument('--n-samples', type=int, default=None,
                        help='Limit number of test samples (default: all 10518)')
    parser.add_argument('--n-viewpoints', type=int, default=8,
                        help='Number of viewpoints to use (1-8, default 8)')
    parser.add_argument('--modes', nargs='+', default=['easy', 'median', 'hard'],
                        choices=['easy', 'median', 'hard'],
                        help='Which difficulty modes to evaluate')
    parser.add_argument('--output', default='/tmp/dtr_results.json')
    parser.add_argument('--no-emd', action='store_true',
                        help='Skip EMD compute (faster but EMD column will be 0)')
    parser.add_argument('--allow-data-parallel', action='store_true',
                        help='Allow DataParallel (NOT recommended; logits may be unreliable)')
    args = parser.parse_args()

    if not os.path.isfile(args.config):
        print(f'ERROR: Config not found: {args.config}'); sys.exit(1)
    if not os.path.isfile(args.ckpt):
        print(f'ERROR: Ckpt not found: {args.ckpt}'); sys.exit(1)

    print(f'Config:        {args.config}')
    print(f'Ckpt:          {args.ckpt}')
    print(f'Modes:         {args.modes}')
    print(f'Viewpoints:    first {args.n_viewpoints} of 8')
    print(f'N samples:     {args.n_samples or "all"}')
    print(f'Output:        {args.output}')

    viewpoints = VIEWPOINT_CHOICES[:args.n_viewpoints]

    builder_args = _Args(args.config, args.ckpt)

    print('\nLoading DiffCond model...')
    model, config = load_diffcond_model(
        args.config, args.ckpt, force_single_gpu=not args.allow_data_parallel)

    print('Building test dataloader...')
    _, test_dataloader = dataset_builder(builder_args, config.dataset.test)
    npoints = config.dataset.test._base_.N_POINTS

    chamfer_l1 = ChamferDistanceL1()
    chamfer_l2 = ChamferDistanceL2()

    per_mode = {}
    for mode in args.modes:
        per_mode[mode] = evaluate_one_mode(
            model, test_dataloader, npoints, mode, viewpoints,
            args.n_samples, chamfer_l1, chamfer_l2,
            require_emd=not args.no_emd)

    print_dtr_table(per_mode)

    with open(args.output, 'w') as f:
        json.dump(per_mode, f, indent=2)
    print(f'\nWrote JSON results to {args.output}')


if __name__ == '__main__':
    main()
