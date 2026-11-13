"""
Compare results across baseline, extension, and ablation experiments.
Loads metrics.json files from checkpoint directories and produces
comparison DataFrames.

Usage:
    python evaluation/compare.py
    python evaluation/compare.py --results-dir results/
"""
import os
import sys
import json
import argparse
from pathlib import Path
import numpy as np

# Add project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def find_metrics_files(base_dir):
    """Find all metrics.json files in a directory tree."""
    metrics_files = []
    base = Path(base_dir)
    for f in base.rglob('metrics.json'):
        metrics_files.append(f)
    return metrics_files


def load_all_results(results_dir='results'):
    """
    Load all metrics.json files and return a list of dicts.

    Each dict contains:
        model, dataset, seq_len, pred_len, mse, mae, rmse, ...
    """
    results = []
    metrics_files = find_metrics_files(results_dir)

    for mf in metrics_files:
        try:
            with open(mf, 'r') as f:
                data = json.load(f)
            results.append(data)
        except (json.JSONDecodeError, IOError) as e:
            print(f"Warning: Could not load {mf}: {e}")

    return results


def build_comparison_table(results):
    """
    Build a comparison table from results.

    Returns:
        headers: list of column names
        rows: list of row dicts
    """
    if not results:
        return [], []

    rows = []
    for r in results:
        row = {
            'model': r.get('model', 'unknown'),
            'dataset': r.get('dataset', 'unknown'),
            'seq_len': r.get('seq_len', '?'),
            'pred_len': r.get('pred_len', '?'),
            'mse': r.get('mse', float('nan')),
            'mae': r.get('mae', float('nan')),
            'rmse': r.get('rmse', float('nan')),
            'mape': r.get('mape', float('nan')),
            'smape': r.get('smape', float('nan')),
            'params': r.get('total_params', '?'),
            'train_time_min': r.get('train_time_min', '?'),
        }
        rows.append(row)

    # Sort by dataset, pred_len, model
    rows.sort(key=lambda x: (str(x['dataset']), str(x['pred_len']), str(x['model'])))
    return rows


def compute_multi_seed_stats(results):
    """
    For experiments with multiple seeds, compute mean ± std.

    Groups by (model, dataset, pred_len) and aggregates across seeds.
    """
    from collections import defaultdict

    groups = defaultdict(list)
    for r in results:
        key = (r.get('model', ''), r.get('dataset', ''), r.get('pred_len', ''))
        groups[key].append(r)

    stats = []
    for key, group in groups.items():
        model, dataset, pred_len = key
        if len(group) < 2:
            continue

        mse_vals = [r['mse'] for r in group if 'mse' in r]
        mae_vals = [r['mae'] for r in group if 'mae' in r]

        stats.append({
            'model': model,
            'dataset': dataset,
            'pred_len': pred_len,
            'num_seeds': len(group),
            'mse_mean': float(np.mean(mse_vals)),
            'mse_std': float(np.std(mse_vals)),
            'mae_mean': float(np.mean(mae_vals)),
            'mae_std': float(np.std(mae_vals)),
        })

    return stats


def format_comparison_markdown(rows):
    """Format comparison as a markdown table."""
    if not rows:
        return "No results found."

    lines = []
    lines.append("| Model | Dataset | Horizon | MSE | MAE | RMSE |")
    lines.append("|-------|---------|---------|-----|-----|------|")

    for r in rows:
        mse_str = f"{r['mse']:.4f}" if isinstance(r['mse'], float) else str(r['mse'])
        mae_str = f"{r['mae']:.4f}" if isinstance(r['mae'], float) else str(r['mae'])
        rmse_str = f"{r['rmse']:.4f}" if isinstance(r['rmse'], float) else str(r['rmse'])
        lines.append(f"| {r['model']} | {r['dataset']} | {r['pred_len']} | {mse_str} | {mae_str} | {rmse_str} |")

    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description='Compare experiment results')
    parser.add_argument('--results-dir', type=str, default='results',
                        help='Directory containing metrics.json files')
    parser.add_argument('--output', type=str, default=None,
                        help='Output file (default: print to stdout)')
    args = parser.parse_args()

    results = load_all_results(args.results_dir)
    rows = build_comparison_table(results)

    md = format_comparison_markdown(rows)
    print(md)

    if args.output:
        with open(args.output, 'w') as f:
            f.write(md)
        print(f"\nSaved to {args.output}")

    # Check for multi-seed experiments
    stats = compute_multi_seed_stats(results)
    if stats:
        print("\n\nMulti-seed Statistics:")
        print("=" * 80)
        for s in stats:
            print(f"  {s['model']} | {s['dataset']} | horizon={s['pred_len']} | "
                  f"MSE={s['mse_mean']:.4f}±{s['mse_std']:.4f} | "
                  f"MAE={s['mae_mean']:.4f}±{s['mae_std']:.4f} | "
                  f"seeds={s['num_seeds']}")


if __name__ == '__main__':
    main()
