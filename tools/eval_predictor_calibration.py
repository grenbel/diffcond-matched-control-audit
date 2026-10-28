#!/usr/bin/env python3
"""Difficulty Predictor Calibration Eval

Evaluates how well DiffCond's auxiliary difficulty predictor head
matches the oracle crop-ratio difficulty on ShapeNet-55 official
S/M/H protocol (8 fixed viewpoints x 3 difficulties).

For each (sample, viewpoint, oracle_difficulty) tuple, runs the
DiffCond model with difficulty='predict' and captures the predictor
logits via model.module._last_diff_pred_logits, then aggregates
calibration metrics.

Outputs:
- Top-1 accuracy
- Confusion matrix (3x3)
- Per-class precision / recall / F1
- Pearson & Spearman correlation (oracle vs argmax)
- Brier score (multi-class)
- ECE (Expected Calibration Error, 10-bin)
- Per-difficulty calibration breakdown

Required so we can defend the predictor in the paper as a calibrated,
non-decorative head.

Usage:
    # quick sanity check on 100 samples
    python tools/eval_predictor_calibration.py --n-samples 100

    # full eval (all 10518 samples) - takes ~2 hours on single GPU
    python tools/eval_predictor_calibration.py
"""
import argparse
import json
import os
import pathlib
import sys
import numpy as np
import torch
import torch.nn.functional as F

# Resolve repo root from this file's location 
THIS_DIR = pathlib.Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parent
os.chdir(str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT))

from utils.config import cfg_from_yaml_file
from utils import misc
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
    """Minimal args object for builder compatibility."""
    def __init__(self, config_path, ckpt_path, exp_name='calibration'):
        self.config = config_path
        self.ckpts = ckpt_path
        self.exp_name = exp_name
        self.use_gpu = True
        self.distributed = False
        self.local_rank = 0
        self.num_workers = 4
        self.experiment_path = './experiments/calibration_tmp'
        self.tfboard_path = './experiments/calibration_tmp/tfboard'
        self.start_ckpts = None
        self.resume = False
        self.test = True
        self.mode = None
        self.difficulty_source = 'predict'
        os.makedirs(self.experiment_path, exist_ok=True)


def load_diffcond_model(config_path, ckpt_path, force_single_gpu=True):
    """Load DiffCond model. Force single GPU by default to avoid DataParallel
    side-effect attribute issues with the `_last_diff_pred_logits` cache."""
    config = cfg_from_yaml_file(config_path)
    base_model = model_builder(config.model)
    load_model(base_model, ckpt_path, logger=None)
    base_model = base_model.cuda()

    # Sanity check: predictor head must NOT be all-zero (zero-init is the
    # vanilla-warmstart fallback in builder.load_model). Trained DiffCond ckpt
    # MUST have non-zero predictor weights.
    pctr = base_model.base_model
    pred_w = pctr.diff_predictor[0].weight
    pred_norm = pred_w.norm().item()
    print(f'  Sanity: predictor[0].weight L2={pred_norm:.4f} (must be > 0 for trained ckpt)')
    assert pred_norm > 1e-3, \
        f'Predictor weight is essentially zero ({pred_norm}); ckpt load failed?'

    if not force_single_gpu and torch.cuda.device_count() > 1:
        base_model = torch.nn.DataParallel(base_model)
        print(f'  Using DataParallel across {torch.cuda.device_count()} GPUs')
    else:
        print(f'  Single-GPU mode (CUDA_VISIBLE_DEVICES={os.environ.get("CUDA_VISIBLE_DEVICES", "all")})')
    base_model.eval()
    return base_model, config


def get_underlying(model):
    """Strip DataParallel if present."""
    return model.module if hasattr(model, 'module') else model


def collect_predictions(model, config, args, n_samples_max=None):
    """Iterate test loader, collect (oracle, pred, probs, logits) per (sample, viewpoint, oracle_diff)."""
    # Build test dataloader (same as test_net). dataset_builder returns (sampler, dataloader).
    _, test_dataloader = dataset_builder(args, config.dataset.test)
    npoints = config.dataset.test._base_.N_POINTS

    all_oracle, all_pred = [], []
    all_probs, all_logits = [], []

    sample_count = 0
    with torch.no_grad():
        for batch_idx, (taxonomy_ids, model_ids, data) in enumerate(test_dataloader):
            gt = data.cuda()
            B = gt.size(0)

            if n_samples_max is not None and sample_count >= n_samples_max:
                break

            for diff_str in ['easy', 'median', 'hard']:
                oracle_diff = DIFF_LEVELS[diff_str]
                num_crop = int(npoints * CROP_RATIO[diff_str])

                for fp in VIEWPOINT_CHOICES:
                    partial, _ = misc.seprate_point_cloud(
                        gt, npoints, num_crop, fixed_points=fp)
                    partial = misc.fps(partial, 2048).cuda()

                    _ = model(partial, difficulty='predict')
                    logits = get_underlying(model)._last_diff_pred_logits  # (B, 3)
                    probs = F.softmax(logits, dim=-1)
                    preds = logits.argmax(dim=-1)

                    all_oracle.extend([oracle_diff] * B)
                    all_pred.extend(preds.cpu().tolist())
                    all_probs.append(probs.cpu().numpy())
                    all_logits.append(logits.cpu().numpy())

            sample_count += B
            if batch_idx % 50 == 0:
                print(f'  Processed {sample_count} samples ({batch_idx+1} batches)', flush=True)

    return (np.array(all_oracle, dtype=np.int64),
            np.array(all_pred, dtype=np.int64),
            np.concatenate(all_probs, axis=0),
            np.concatenate(all_logits, axis=0))


def expected_calibration_error(probs, preds, oracle, n_bins=10):
    """Multi-class ECE based on predicted confidence vs accuracy."""
    confidences = probs.max(axis=1)
    accuracies = (preds == oracle).astype(np.float64)
    bin_edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    n_total = len(confidences)
    for i in range(n_bins):
        in_bin = (confidences >= bin_edges[i]) & (confidences < bin_edges[i + 1])
        if i == n_bins - 1:
            in_bin = in_bin | (confidences == bin_edges[i + 1])
        n_in = in_bin.sum()
        if n_in > 0:
            avg_conf = confidences[in_bin].mean()
            avg_acc = accuracies[in_bin].mean()
            ece += abs(avg_conf - avg_acc) * n_in / n_total
    return float(ece)


def brier_multiclass(probs, oracle, n_classes=3):
    """Multi-class Brier score: mean of squared L2 between prob and one-hot."""
    onehot = np.zeros_like(probs)
    onehot[np.arange(len(oracle)), oracle] = 1.0
    return float(np.mean(np.sum((probs - onehot) ** 2, axis=1)))


def per_class_pr(oracle, pred, n_classes=3):
    """Return per-class precision, recall, F1, support."""
    out = {}
    for c in range(n_classes):
        tp = int(((pred == c) & (oracle == c)).sum())
        fp = int(((pred == c) & (oracle != c)).sum())
        fn = int(((pred != c) & (oracle == c)).sum())
        support = int((oracle == c).sum())
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        out[c] = {'precision': precision, 'recall': recall, 'f1': f1, 'support': support}
    return out


def correlations(oracle, pred):
    """Pearson + Spearman (both treated as ordinal 0/1/2)."""
    try:
        from scipy.stats import pearsonr, spearmanr
        pr, _ = pearsonr(oracle, pred)
        sp, _ = spearmanr(oracle, pred)
    except ImportError:
        # Fallback manual Pearson
        o = oracle.astype(np.float64)
        p = pred.astype(np.float64)
        pr = float(np.corrcoef(o, p)[0, 1])
        sp = float('nan')
    return float(pr), float(sp)


def per_oracle_breakdown(oracle, pred, probs):
    """Calibration metrics broken down by oracle difficulty."""
    out = {}
    for diff_idx in range(3):
        mask = (oracle == diff_idx)
        if mask.sum() == 0:
            continue
        sub_probs = probs[mask]
        sub_pred = pred[mask]
        sub_oracle = oracle[mask]
        out[diff_idx] = {
            'n': int(mask.sum()),
            'accuracy': float((sub_pred == sub_oracle).mean()),
            'pred_distribution': {c: int((sub_pred == c).sum()) for c in range(3)},
            'mean_prob_easy': float(sub_probs[:, 0].mean()),
            'mean_prob_median': float(sub_probs[:, 1].mean()),
            'mean_prob_hard': float(sub_probs[:, 2].mean()),
        }
    return out


def report(oracle, pred, probs, logits, output_path=None):
    print('\n' + '=' * 78)
    print('  Difficulty Predictor Calibration Report')
    print('=' * 78)

    n_total = len(oracle)
    print(f'  Total predictions: {n_total}')
    print(f'  Per-oracle counts: easy={int((oracle==0).sum())} '
          f'median={int((oracle==1).sum())} hard={int((oracle==2).sum())}')

    # Top-1 accuracy
    acc = float((pred == oracle).mean())
    print(f'\n  Top-1 Accuracy: {acc*100:.2f}%')

    # Confusion matrix
    print('\n  Confusion Matrix (rows=oracle, cols=predicted):')
    print(f'           {"easy":>8} {"median":>8} {"hard":>8}')
    for r in range(3):
        row = [int(((oracle == r) & (pred == c)).sum()) for c in range(3)]
        labels = ['easy', 'median', 'hard']
        print(f'  {labels[r]:>6}  {row[0]:>8d} {row[1]:>8d} {row[2]:>8d}')

    # Per-class P/R/F1
    pr_results = per_class_pr(oracle, pred)
    print('\n  Per-class metrics:')
    print(f'           {"prec":>8} {"recall":>8} {"f1":>8} {"support":>9}')
    for c, name in enumerate(['easy', 'median', 'hard']):
        m = pr_results[c]
        print(f'  {name:>6}  {m["precision"]:>8.3f} {m["recall"]:>8.3f} '
              f'{m["f1"]:>8.3f} {m["support"]:>9d}')

    # Correlations (ordinal)
    pearson, spearman = correlations(oracle, pred)
    print(f'\n  Pearson correlation:  {pearson:+.4f}')
    print(f'  Spearman correlation: {spearman:+.4f}')

    # Brier + ECE
    brier = brier_multiclass(probs, oracle)
    ece = expected_calibration_error(probs, pred, oracle)
    print(f'\n  Brier score (lower better): {brier:.4f}')
    print(f'  ECE 10-bin   (lower better): {ece:.4f}')

    # Per-oracle breakdown (monotonicity check)
    print('\n  Per-oracle-difficulty breakdown:')
    breakdown = per_oracle_breakdown(oracle, pred, probs)
    for diff_idx, name in enumerate(['easy', 'median', 'hard']):
        if diff_idx not in breakdown:
            continue
        b = breakdown[diff_idx]
        print(f'  oracle={name}: n={b["n"]}, acc={b["accuracy"]*100:.2f}%, '
              f'mean_probs (easy/med/hard) = ({b["mean_prob_easy"]:.3f}, '
              f'{b["mean_prob_median"]:.3f}, {b["mean_prob_hard"]:.3f})')
    print('=' * 78)

    if output_path:
        result = {
            'n_total': n_total,
            'accuracy': acc,
            'pearson': pearson,
            'spearman': spearman,
            'brier': brier,
            'ece': ece,
            'per_class': pr_results,
            'per_oracle': breakdown,
            'confusion_matrix': [
                [int(((oracle == r) & (pred == c)).sum()) for c in range(3)]
                for r in range(3)
            ],
        }
        with open(output_path, 'w') as f:
            json.dump(result, f, indent=2)
        print(f'\n  Wrote JSON results to {output_path}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='cfgs/ShapeNet55_models/AdaPoinTr_DiffCond_smoke30.yaml')
    parser.add_argument('--ckpt', default='experiments/<diffcond-run>/ckpt-epoch-030.pth')
    parser.add_argument('--n-samples', type=int, default=None,
                        help='Limit number of test samples (for quick sanity). Default: all 10518.')
    parser.add_argument('--output', default='/tmp/predictor_calibration.json')
    parser.add_argument('--allow-data-parallel', action='store_true',
                        help='Allow DataParallel (NOT recommended; logits may be unreliable)')
    args = parser.parse_args()

    if not os.path.isfile(args.config):
        print(f'ERROR: Config not found: {args.config}')
        sys.exit(1)
    if not os.path.isfile(args.ckpt):
        print(f'ERROR: Ckpt not found: {args.ckpt}')
        sys.exit(1)

    print(f'Config: {args.config}')
    print(f'Ckpt:   {args.ckpt}')
    print(f'Output: {args.output}')
    if args.n_samples:
        print(f'Limiting to first {args.n_samples} samples (sanity mode)')

    builder_args = _Args(args.config, args.ckpt)

    print('\nLoading DiffCond model + checkpoint...')
    model, config = load_diffcond_model(
        args.config, args.ckpt, force_single_gpu=not args.allow_data_parallel)
    print('Model loaded.')

    print('\nCollecting predictions over test set (3 difficulties x 8 viewpoints x N samples)...')
    oracle, pred, probs, logits = collect_predictions(
        model, config, builder_args, n_samples_max=args.n_samples)

    report(oracle, pred, probs, logits, output_path=args.output)


if __name__ == '__main__':
    main()
