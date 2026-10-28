#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Paired per-seed statistical analysis for the SN55 SMH eval.

Reads `data/results/sn55_paired_results.json` (per-seed values), computes:
  - Per-seed paired deltas (Vanilla vs DiffCond, sign-aware: lower=better for CDL,
    higher=better for F-Score)
  - mean paired delta over the n seeds (% improvement; n auto-detected)
  - std of the paired deltas
  - 95% CI via t-distribution (df = n-1; df=3 for the n=4 cross-backbone runs)
  - Whether CI crosses zero (statistical significance)
  - Whether CI lower bound beats the pre-specified +3% GO-bar method threshold

Writes:
  - `data/results/paired_analysis.tsv` -- full per-seed x per-(mode,metric) data
  - stdout -- summary table for CDL2 (main paper Sec. 5.5, Table 4)

Usage:
    cd /path/to/repo
    python tools/paired_stats.py
"""
import json
import math
import statistics
import sys
from pathlib import Path

JSON_PATH = Path('data/results/sn55_paired_results.json')
TSV_PATH = Path('data/results/paired_analysis.tsv')
GO_BAR_PCT = 3.0  # Pre-specified method-bar threshold (paper Sec. 3 / Sec. 5)

# Two-sided 95% CI t critical values: t.ppf(0.975, df) for df=1..10
# (Pre-computed to avoid scipy dependency. df = n - 1.)
T_CRIT_95 = {
    1: 12.706204736432095,
    2: 4.302652729911275,
    3: 3.182446305284263,
    4: 2.7764451051977987,
    5: 2.5705818356026378,
    6: 2.4469118487496734,
    7: 2.3646242510102993,
    8: 2.3060041350333704,
    9: 2.2621571627409915,
    10: 2.2281388519649385,
}

MODES = ['easy', 'median', 'hard']
METRICS = ['F-Score', 'CDL1', 'CDL2']  # F-Score: higher better; CDL: lower better


def t_crit_95(df):
    """Two-sided 95% CI t critical value for given df. df = n - 1."""
    if df in T_CRIT_95:
        return T_CRIT_95[df]
    if df > 10:
        # Approach 1.96 as df -> inf; df > 10 already very close.
        return 1.96
    raise ValueError(f'df={df} too small (need n >= 2)')


def paired_delta_pct(v, d, metric):
    """Per-seed paired delta as % improvement."""
    if v == 0:
        return 0.0, 0.0
    if metric in ('CDL1', 'CDL2'):
        # lower better
        return (v - d), (v - d) / v * 100.0
    else:
        # F-Score: higher better
        return (d - v), (d - v) / v * 100.0


def main():
    if not JSON_PATH.exists():
        print(f'ERROR: {JSON_PATH} not found. Run aggregate_results.py first.', file=sys.stderr)
        sys.exit(1)
    data = json.loads(JSON_PATH.read_text(encoding='utf-8'))
    vanilla = data['per_seed']['vanilla']
    diffcond = data['per_seed']['diffcond']
    seeds = sorted(vanilla.keys())  # seed_42 ... seed_45 in the released JSON

    n = len(seeds)
    df = n - 1
    if df < 1:
        print(f'ERROR: need n >= 2 seeds, got n={n}', file=sys.stderr)
        sys.exit(1)
    t_crit = t_crit_95(df)
    print(f'n_seeds={n}  df={df}  t_crit(95% two-sided)={t_crit:.6f}')

    rows = [[
        'mode', 'metric', 'seed', 'vanilla', 'diffcond',
        'abs_paired_delta', 'paired_pct_improvement',
        'mean_paired_pct', 'std_paired_pct',
        f'ci95_lo_pct(df={df})', f'ci95_hi_pct(df={df})', 'ci_crosses_zero',
        'go_bar_pct', 'beats_go_bar_lo',
    ]]

    summary = {}  # for stdout printing
    for mode in MODES:
        summary[mode] = {}
        for metric in METRICS:
            v_vals = [vanilla[s][mode][metric] for s in seeds]
            d_vals = [diffcond[s][mode][metric] for s in seeds]
            paired_abs, paired_pct = [], []
            for v, d in zip(v_vals, d_vals):
                a, p = paired_delta_pct(v, d, metric)
                paired_abs.append(a)
                paired_pct.append(p)
            mean_pp = statistics.mean(paired_pct)
            std_pp = statistics.stdev(paired_pct) if len(paired_pct) > 1 else 0.0
            half = t_crit * std_pp / math.sqrt(n) if std_pp > 0 else 0.0
            ci_lo = mean_pp - half
            ci_hi = mean_pp + half
            crosses_zero = (ci_lo < 0 < ci_hi) or (ci_lo == 0 or ci_hi == 0)
            beats = ci_lo > GO_BAR_PCT
            summary[mode][metric] = {
                'paired_pct': paired_pct,
                'mean': mean_pp, 'std': std_pp,
                'ci_lo': ci_lo, 'ci_hi': ci_hi,
                'crosses_zero': crosses_zero, 'beats_go_bar': beats,
            }
            for i, seed in enumerate(seeds):
                rows.append([
                    mode, metric, seed,
                    f'{v_vals[i]:.4f}', f'{d_vals[i]:.4f}',
                    f'{paired_abs[i]:+.4f}', f'{paired_pct[i]:+.4f}',
                    f'{mean_pp:+.4f}', f'{std_pp:.4f}',
                    f'{ci_lo:+.4f}', f'{ci_hi:+.4f}', str(crosses_zero),
                    f'{GO_BAR_PCT}', str(beats),
                ])

    TSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    TSV_PATH.write_text('\n'.join('\t'.join(r) for r in rows) + '\n', encoding='utf-8')
    print(f'Wrote {TSV_PATH} ({len(rows) - 1} data rows)')
    print()

    # Summary for CDL2 (focal metric for main paper Sec. 5.5)
    print(f'\n=== CDL2 paired-seed analysis (DiffCond vs Vanilla, % improvement, n={n}) ===')
    print(f'{"Mode":<8} {f"Per-seed pct (n={n})":<32} {f"{n}-seed mean":>14} {f"95% CI (df={df})":>22} {"crosses 0?":>11} {"beats +3% GO?":>15}')
    for mode in MODES:
        s = summary[mode]['CDL2']
        per_seed_str = ', '.join(f'{p:+.2f}%' for p in s['paired_pct'])
        print(f'{mode:<8} [{per_seed_str:<26}] {s["mean"]:+10.2f}% [{s["ci_lo"]:+6.2f}%, {s["ci_hi"]:+6.2f}%] {str(s["crosses_zero"]):>11} {str(s["beats_go_bar"]):>15}')

    print()
    print(f'=== F-Score paired-seed analysis (n={n}) ===')
    print(f'{"Mode":<8} {f"Per-seed pct (n={n})":<32} {f"{n}-seed mean":>14} {f"95% CI (df={df})":>22} {"crosses 0?":>11}')
    for mode in MODES:
        s = summary[mode]['F-Score']
        per_seed_str = ', '.join(f'{p:+.2f}%' for p in s['paired_pct'])
        print(f'{mode:<8} [{per_seed_str:<26}] {s["mean"]:+10.2f}% [{s["ci_lo"]:+6.2f}%, {s["ci_hi"]:+6.2f}%] {str(s["crosses_zero"]):>11}')


if __name__ == '__main__':
    if any(a in {'-h', '--help'} for a in sys.argv[1:]):
        print(
            'Usage: python tools/paired_stats.py\n'
            '  Reads:  data/results/sn55_paired_results.json (per-seed paired values)\n'
            '  Writes: data/results/paired_analysis.tsv (full per-seed table)\n'
            '  Prints: CDL2 and F-Score paired-seed CI summary to stdout\n'
            '  No CLI arguments are accepted; the script is deterministic.\n'
        )
        sys.exit(0)
    main()
