#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Aggregate cross-backbone S/M/H eval results (post-training) into a paper-ready JSON + table.

Scans:
  experiments/UpTrans/ShapeNet55_models/vanilla_seed{42,43,44,45}/smh_results/
  experiments/UpTrans_DiffCond/ShapeNet55_models/diffcond_seed{42,43,44,45}/smh_results/

For each (family, seed, mode), parses the eval_{mode}_summary.txt 'Overall' line
(format: `Overall    F-Score   CDL1   CDL2   EMD`). Computes:
  - Per-family per-mode mean / std over the requested seeds (default 4)
  - DiffCond vs Vanilla delta (abs + % improvement)

CDL1/CDL2: lower is better -> improvement_pct = (vanilla - diffcond) / vanilla * 100
F-Score:   higher is better -> improvement_pct = (diffcond - vanilla) / vanilla * 100

Output JSON: data/results/sn55_paired_results.json
Output TSV:  data/results/sn55_paired_results.tsv

Usage:
  cd /repo/root && python tools/aggregate_results.py
or with explicit seed override (e.g. for ablations):
  python tools/aggregate_results.py --seeds 42,43,44,45
"""
import argparse
import json
import re
import statistics
import sys
from pathlib import Path

MODES = ['easy', 'median', 'hard']
METRICS = ['F-Score', 'CDL1', 'CDL2']
EMD_COLUMN_NAME = 'EMD'  # 4th column; unused if require_emd=False

# Regex for the summary 'Overall' line. Tab-separated with variable whitespace.
OVERALL_RE = re.compile(r'Overall\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)')


def parse_overall(summary_path):
    """Parse 'Overall F CDL1 CDL2 EMD' from an eval_{mode}_summary.txt file."""
    try:
        txt = summary_path.read_text(encoding='utf-8', errors='replace')
    except OSError:
        return None
    m = OVERALL_RE.search(txt)
    if not m:
        return None
    try:
        return {
            'F-Score': float(m.group(1)),
            'CDL1': float(m.group(2)),
            'CDL2': float(m.group(3)),
            EMD_COLUMN_NAME: float(m.group(4)),
        }
    except ValueError:
        return None


def collect_family(root, family_prefix, seeds):
    """Collect per-seed per-mode metrics for one family."""
    result = {}
    for seed in seeds:
        exp_dir = Path(root) / f'{family_prefix}{seed}'
        smh_dir = exp_dir / 'smh_results'
        if not smh_dir.is_dir():
            print(f'  [WARN] missing: {smh_dir}', file=sys.stderr)
            continue
        seed_key = f'seed_{seed}'
        result[seed_key] = {}
        for mode in MODES:
            summary = smh_dir / f'eval_{mode}_summary.txt'
            if not summary.is_file():
                print(f'  [WARN] missing: {summary}', file=sys.stderr)
                continue
            parsed = parse_overall(summary)
            if parsed is None:
                print(f'  [WARN] unparseable: {summary}', file=sys.stderr)
                continue
            result[seed_key][mode] = parsed
    return result


def aggregate_family(per_seed_data):
    """Compute mean/std per (mode, metric) across seeds."""
    agg = {}
    for mode in MODES:
        agg[mode] = {}
        for metric in METRICS:
            vals = [sd.get(mode, {}).get(metric) for sd in per_seed_data.values()]
            vals = [v for v in vals if v is not None]
            if not vals:
                continue
            agg[mode][metric] = {
                'mean': statistics.mean(vals),
                'std': statistics.stdev(vals) if len(vals) > 1 else 0.0,
                'n': len(vals),
                'values': vals,
            }
    return agg


def compute_delta(vanilla_agg, diffcond_agg):
    """Per (mode, metric) compute DiffCond - Vanilla delta + %improvement."""
    delta = {}
    for mode in MODES:
        delta[mode] = {}
        for metric in METRICS:
            v = vanilla_agg.get(mode, {}).get(metric)
            d = diffcond_agg.get(mode, {}).get(metric)
            if not (v and d):
                continue
            vm = v['mean']
            dm = d['mean']
            abs_d = dm - vm
            if metric in ('CDL1', 'CDL2'):
                # lower is better
                pct = (vm - dm) / vm * 100.0 if vm != 0 else 0.0
            else:
                # F-Score: higher is better
                pct = (dm - vm) / vm * 100.0 if vm != 0 else 0.0
            delta[mode][metric] = {
                'vanilla_mean': vm,
                'vanilla_std': v.get('std', 0),
                'diffcond_mean': dm,
                'diffcond_std': d.get('std', 0),
                'abs_delta': abs_d,
                'improvement_pct': pct,
            }
    return delta


def print_table(delta, file=None):
    if file is None:
        file = sys.stdout
    print(f'{"Mode":<8} {"Metric":<8} '
          f'{"Vanilla (mean+/-std)":>25} {"DiffCond (mean+/-std)":>25} '
          f'{"AbsDelta":>10} {"Delta%":>8}', file=file)
    print('-' * 90, file=file)
    for mode in MODES:
        for metric in METRICS:
            d = delta.get(mode, {}).get(metric)
            if d is None:
                continue
            v_str = f'{d["vanilla_mean"]:.4f}+/-{d["vanilla_std"]:.4f}'
            d_str = f'{d["diffcond_mean"]:.4f}+/-{d["diffcond_std"]:.4f}'
            print(f'{mode:<8} {metric:<8} {v_str:>25} {d_str:>25} '
                  f'{d["abs_delta"]:>+10.4f} {d["improvement_pct"]:>+7.2f}%',
                  file=file)


def write_tsv(delta, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as f:
        f.write('mode\tmetric\tvanilla_mean\tvanilla_std\tdiffcond_mean\tdiffcond_std\t'
                'abs_delta\timprovement_pct\n')
        for mode in MODES:
            for metric in METRICS:
                d = delta.get(mode, {}).get(metric)
                if d is None:
                    continue
                f.write(f'{mode}\t{metric}\t{d["vanilla_mean"]:.6f}\t'
                        f'{d["vanilla_std"]:.6f}\t{d["diffcond_mean"]:.6f}\t'
                        f'{d["diffcond_std"]:.6f}\t{d["abs_delta"]:.6f}\t'
                        f'{d["improvement_pct"]:.4f}\n')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default='experiments',
                    help='experiments root directory')
    ap.add_argument('--seeds', default='42,43,44,45',
                    help='comma-separated seeds (default: 4-seed cross-backbone, n=4)')
    ap.add_argument('--out_json', default='data/results/sn55_paired_results.json')
    ap.add_argument('--out_tsv', default='data/results/sn55_paired_results.tsv')
    args = ap.parse_args()

    seeds = [int(s.strip()) for s in args.seeds.split(',') if s.strip()]

    print(f'=== Collecting SN55 vanilla ===')
    van = collect_family(args.root,
                         'UpTrans/ShapeNet55_models/vanilla_seed',
                         seeds)
    print(f'  got {len(van)} seeds: {sorted(van.keys())}')

    print(f'=== Collecting SN55 DiffCond ===')
    dif = collect_family(args.root,
                         'UpTrans_DiffCond/ShapeNet55_models/diffcond_seed',
                         seeds)
    print(f'  got {len(dif)} seeds: {sorted(dif.keys())}')

    if not van and not dif:
        print('No results found. Run training + eval scripts first.', file=sys.stderr)
        sys.exit(1)

    van_agg = aggregate_family(van)
    dif_agg = aggregate_family(dif)
    delta = compute_delta(van_agg, dif_agg)

    print()
    print('=== SN55 Cross-Backbone Results: DiffCond vs Vanilla-FT (paper-ready) ===')
    print()
    print_table(delta)

    # Save artifacts
    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_json, 'w', encoding='utf-8') as f:
        json.dump({
            'per_seed': {'vanilla': van, 'diffcond': dif},
            'aggregate': {'vanilla': van_agg, 'diffcond': dif_agg},
            'delta_diffcond_minus_vanilla': delta,
        }, f, indent=2, sort_keys=True)
    print(f'\nSaved: {args.out_json}')

    write_tsv(delta, args.out_tsv)
    print(f'Saved: {args.out_tsv}')


if __name__ == '__main__':
    main()
