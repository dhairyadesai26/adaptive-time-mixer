"""
Automatic visualization for TimeMixer experiments.

Generates 5 plot types:
1. Forecast vs Actual — time-series overlay
2. Error Comparison Bar Chart — baseline vs extension
3. Horizon Performance Line Plot — MSE across horizons
4. Adaptive Weights Heatmap — scale weight distribution
5. Training Curves — loss convergence

Usage:
    python evaluation/plots.py
    python evaluation/plots.py --results-dir results/ --output-dir results/plots/
"""
import os
import sys
import json
import argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ── Style configuration ──────────────────────────────────────────
plt.rcParams.update({
    'figure.figsize': (12, 6),
    'figure.dpi': 150,
    'font.family': 'sans-serif',
    'font.size': 12,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.grid': True,
    'grid.alpha': 0.3,
})

COLORS = {
    'baseline': '#4A90D9',
    'adaptive': '#E74C3C',
    'ablation_no_context': '#2ECC71',
    'ablation_no_horizon': '#F39C12',
    'ablation_fixed': '#9B59B6',
}


def plot_forecast_vs_actual(actual, predicted, title="Forecast vs Actual",
                             output_path=None, channel_idx=0, start=0, length=200):
    """
    Plot actual vs predicted time series.

    Args:
        actual: np.ndarray [time, channels]
        predicted: np.ndarray [time, channels]
        title: plot title
        output_path: file path to save
        channel_idx: which channel to plot
        start: start index
        length: how many points to show
    """
    fig, ax = plt.subplots(figsize=(14, 5))

    end = min(start + length, len(actual))
    t = np.arange(start, end)
    a = actual[start:end, channel_idx]
    p = predicted[start:end, channel_idx]

    ax.plot(t, a, color=COLORS['baseline'], linewidth=1.5, label='Actual', alpha=0.8)
    ax.plot(t, p, color=COLORS['adaptive'], linewidth=1.5, label='Predicted',
            alpha=0.8, linestyle='--')

    # Shade the error
    ax.fill_between(t, a, p, alpha=0.1, color=COLORS['adaptive'])

    ax.set_xlabel('Time Step')
    ax.set_ylabel('Value')
    ax.set_title(title)
    ax.legend(loc='upper right')

    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, bbox_inches='tight')
        print(f"  Saved: {output_path}")
    plt.close()


def plot_error_comparison(models, mse_values, mae_values, title="Error Comparison",
                           output_path=None):
    """
    Bar chart comparing MSE and MAE across models.

    Args:
        models: list of model names
        mse_values: list of MSE values
        mae_values: list of MAE values
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    x = np.arange(len(models))
    width = 0.5

    colors = [COLORS.get(m.lower().replace(' ', '_'),
                          COLORS['baseline']) for m in models]

    # MSE
    bars1 = axes[0].bar(x, mse_values, width, color=colors, alpha=0.85, edgecolor='white')
    axes[0].set_xlabel('Model')
    axes[0].set_ylabel('MSE')
    axes[0].set_title('Mean Squared Error')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(models, rotation=30, ha='right')
    for bar, val in zip(bars1, mse_values):
        axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.002,
                     f'{val:.4f}', ha='center', va='bottom', fontsize=9)

    # MAE
    bars2 = axes[1].bar(x, mae_values, width, color=colors, alpha=0.85, edgecolor='white')
    axes[1].set_xlabel('Model')
    axes[1].set_ylabel('MAE')
    axes[1].set_title('Mean Absolute Error')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(models, rotation=30, ha='right')
    for bar, val in zip(bars2, mae_values):
        axes[1].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.002,
                     f'{val:.4f}', ha='center', va='bottom', fontsize=9)

    fig.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, bbox_inches='tight')
        print(f"  Saved: {output_path}")
    plt.close()


def plot_horizon_performance(horizons, baseline_mse, extension_mse=None,
                              title="MSE across Forecast Horizons",
                              output_path=None):
    """
    Line plot showing MSE across different forecast horizons.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(horizons, baseline_mse, 'o-', color=COLORS['baseline'],
            linewidth=2, markersize=8, label='TimeMixer (Baseline)')

    if extension_mse is not None:
        ax.plot(horizons, extension_mse, 's--', color=COLORS['adaptive'],
                linewidth=2, markersize=8, label='Adaptive TimeMixer (Ours)')

    ax.set_xlabel('Forecast Horizon')
    ax.set_ylabel('MSE')
    ax.set_title(title)
    ax.legend()
    ax.set_xticks(horizons)

    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, bbox_inches='tight')
        print(f"  Saved: {output_path}")
    plt.close()


def plot_adaptive_weights(weights, scale_names=None, title="Adaptive Scale Weights",
                           output_path=None):
    """
    Heatmap showing how the adaptive mixer weights different scales.

    Args:
        weights: np.ndarray [num_samples, num_scales]
        scale_names: list of scale names
    """
    num_scales = weights.shape[1] if len(weights.shape) > 1 else len(weights)

    if scale_names is None:
        scale_names = [f'Scale {i+1}' for i in range(num_scales)]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Heatmap of weights
    if len(weights.shape) > 1:
        im = axes[0].imshow(weights[:min(50, len(weights))].T, aspect='auto',
                            cmap='YlOrRd', interpolation='nearest')
        axes[0].set_yticks(range(num_scales))
        axes[0].set_yticklabels(scale_names)
        axes[0].set_xlabel('Sample Index')
        axes[0].set_ylabel('Scale')
        axes[0].set_title('Weight Distribution per Sample')
        plt.colorbar(im, ax=axes[0], label='Weight')
    else:
        axes[0].bar(range(num_scales), weights, color=COLORS['adaptive'], alpha=0.85)
        axes[0].set_xticks(range(num_scales))
        axes[0].set_xticklabels(scale_names)
        axes[0].set_ylabel('Weight')
        axes[0].set_title('Average Scale Weights')

    # Distribution of weights
    if len(weights.shape) > 1:
        bp = axes[1].boxplot([weights[:, i] for i in range(num_scales)],
                              labels=scale_names, patch_artist=True)
        colors_list = plt.cm.viridis(np.linspace(0.2, 0.8, num_scales))
        for patch, color in zip(bp['boxes'], colors_list):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        axes[1].set_ylabel('Weight')
        axes[1].set_title('Weight Distribution per Scale')
    else:
        axes[1].pie(weights, labels=scale_names, autopct='%1.1f%%',
                    colors=plt.cm.viridis(np.linspace(0.2, 0.8, num_scales)))
        axes[1].set_title('Scale Weight Proportions')

    fig.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, bbox_inches='tight')
        print(f"  Saved: {output_path}")
    plt.close()


def plot_training_curves(epochs, train_loss, val_loss=None,
                          title="Training Convergence", output_path=None):
    """Plot training and validation loss curves."""
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(epochs, train_loss, '-', color=COLORS['baseline'],
            linewidth=2, label='Train Loss')

    if val_loss is not None:
        ax.plot(epochs, val_loss, '--', color=COLORS['adaptive'],
                linewidth=2, label='Validation Loss')

    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title(title)
    ax.legend()

    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, bbox_inches='tight')
        print(f"  Saved: {output_path}")
    plt.close()


def generate_all_plots(results_dir='results', output_dir='results/plots'):
    """Generate all available plots from results directory."""
    os.makedirs(output_dir, exist_ok=True)

    print(f"Generating plots from {results_dir} → {output_dir}")
    print("=" * 60)

    # Load results
    from evaluation.compare import load_all_results, build_comparison_table
    results = load_all_results(results_dir)
    rows = build_comparison_table(results)

    if not rows:
        print("No results found. Generating demo plots with placeholder data...")
        # Demo: horizon performance
        horizons = [96, 192, 336, 720]
        baseline_mse = [0.462, 0.474, 0.494, 0.529]
        plot_horizon_performance(
            horizons, baseline_mse,
            title="Traffic — MSE vs Horizon (Paper Reference)",
            output_path=os.path.join(output_dir, 'horizon_performance_paper.png')
        )
        # Demo: adaptive weights
        np.random.seed(42)
        weights = np.random.dirichlet([1, 1.5, 2, 2.5], size=50)
        plot_adaptive_weights(
            weights, title="Adaptive Weights (Demo)",
            output_path=os.path.join(output_dir, 'adaptive_weights_demo.png')
        )
        return

    # Group results by dataset
    datasets = {}
    for r in rows:
        ds = r['dataset']
        if ds not in datasets:
            datasets[ds] = []
        datasets[ds].append(r)

    # Generate per-dataset plots
    for ds, ds_rows in datasets.items():
        # Horizon performance
        baseline_rows = [r for r in ds_rows if 'adaptive' not in r['model'].lower()]
        extension_rows = [r for r in ds_rows if 'adaptive' in r['model'].lower()]

        if baseline_rows:
            horizons = sorted(set(int(r['pred_len']) for r in baseline_rows
                                  if r['pred_len'] != '?'))
            if horizons:
                b_mse = [next((r['mse'] for r in baseline_rows
                              if int(r['pred_len']) == h), None) for h in horizons]
                e_mse = None
                if extension_rows:
                    e_mse = [next((r['mse'] for r in extension_rows
                                  if int(r['pred_len']) == h), None) for h in horizons]

                b_mse = [v for v in b_mse if v is not None]
                if len(b_mse) == len(horizons):
                    plot_horizon_performance(
                        horizons, b_mse,
                        extension_mse=e_mse if e_mse and all(v is not None for v in e_mse) else None,
                        title=f"{ds} — MSE across Horizons",
                        output_path=os.path.join(output_dir, f'{ds.lower()}_horizon_performance.png')
                    )

        # Error comparison across models for each horizon
        for h in sorted(set(r['pred_len'] for r in ds_rows if r['pred_len'] != '?')):
            h_rows = [r for r in ds_rows if r['pred_len'] == h]
            if len(h_rows) > 1:
                models = [r['model'] for r in h_rows]
                mse_vals = [r['mse'] for r in h_rows]
                mae_vals = [r['mae'] for r in h_rows]
                plot_error_comparison(
                    models, mse_vals, mae_vals,
                    title=f"{ds} — Horizon {h}",
                    output_path=os.path.join(output_dir, f'{ds.lower()}_h{h}_comparison.png')
                )

    print(f"\nAll plots saved to {output_dir}/")


def main():
    parser = argparse.ArgumentParser(description='Generate experiment plots')
    parser.add_argument('--results-dir', type=str, default='results',
                        help='Directory containing metrics.json files')
    parser.add_argument('--output-dir', type=str, default='results/plots',
                        help='Output directory for plots')
    args = parser.parse_args()

    generate_all_plots(args.results_dir, args.output_dir)


if __name__ == '__main__':
    main()
