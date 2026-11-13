"""
Post-Evaluation: Traffic Dataset — All Horizons (96, 192, 336, 720)

Performs comprehensive evaluation:
  1. Loads each Traffic checkpoint
  2. Runs test-set evaluation with proper train/val/test split
  3. Computes MSE, MAE, RMSE, MAPE, sMAPE
  4. Compares results against TimeMixer paper reference values
  5. Generates visualizations:
      - Horizon performance comparison (ours vs paper)
      - Forecast vs actual overlays (per horizon)
      - Per-channel error distribution
      - Error heatmap across time
  6. Saves results to JSON, CSV, Markdown, and LaTeX

Usage:
    python post_evaluate_traffic.py
    python post_evaluate_traffic.py --device cuda
    python post_evaluate_traffic.py --horizons 96 192
"""
import os
import sys
import json
import time
import hashlib
import argparse
from datetime import datetime
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch

# -- Paths --------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from evaluation.metrics import compute_all_metrics, format_metrics
from inference.predict import load_model, config_from_yaml

# -- Paper reference values ---------------------------------------
PAPER_REFERENCE = {
    96:  {'mse': 0.462, 'mae': 0.285},
    192: {'mse': 0.474, 'mae': 0.292},
    336: {'mse': 0.494, 'mae': 0.302},
    720: {'mse': 0.529, 'mae': 0.322},
}

# -- Checkpoint mapping -------------------------------------------
# Maps pred_len to checkpoint directory name inside checkpoints/
CHECKPOINT_MAP = {
    96:  'long_term_forecast_Traffic_96_96_paper_none_TimeMixer_custom_sl96_pl96_dm32_nh8_el2_dl1_df64_fc3_ebtimeF_dtTrue_paper_0',
    192: 'long_term_forecast_Traffic_96_192_paper_none_TimeMixer_custom_sl96_pl192_dm32_nh8_el2_dl1_df64_fc3_ebtimeF_dtTrue_paper_0',
    336: None,  # Uses standalone .pth
    720: None,  # Uses standalone .pth
}

STANDALONE_CHECKPOINTS = {
    336: 'Traffic_96_336_checkpoint.pth',
    720: 'Traffic_96_720_checkpoint.pth',
}


def find_checkpoint(pred_len):
    """Find the checkpoint path for a given horizon."""
    ckpt_dir = os.path.join(PROJECT_ROOT, 'checkpoints')

    # Check mapped directory first
    mapped = CHECKPOINT_MAP.get(pred_len)
    if mapped:
        path = os.path.join(ckpt_dir, mapped, 'checkpoint.pth')
        if os.path.exists(path):
            return path

    # Check standalone .pth files
    standalone = STANDALONE_CHECKPOINTS.get(pred_len)
    if standalone:
        path = os.path.join(ckpt_dir, standalone)
        if os.path.exists(path):
            return path

    # Fallback: search for any Traffic checkpoint with this horizon
    for entry in os.listdir(ckpt_dir):
        if f'Traffic_96_{pred_len}' in entry or f'Traffic_{pred_len}' in entry:
            full = os.path.join(ckpt_dir, entry)
            if os.path.isdir(full):
                pth = os.path.join(full, 'checkpoint.pth')
                if os.path.exists(pth):
                    return pth
            elif full.endswith('.pth') and os.path.exists(full):
                return full

    return None


def load_traffic_data():
    """
    Load the Traffic dataset and split into train/val/test.
    Uses the standard 7:1:2 split for TimeMixer benchmarks.
    """
    csv_path = os.path.join(PROJECT_ROOT, 'dataset', 'traffic', 'traffic.csv')
    print(f"  Loading: {csv_path}")

    df = pd.read_csv(csv_path)
    print(f"  Shape: {df.shape[0]} rows × {df.shape[1]} columns")

    # Extract numeric data (all columns except 'date')
    date_col = 'date'
    feature_cols = [c for c in df.columns if c != date_col]
    data = df[feature_cols].values.astype(np.float32)

    # Standard 7:1:2 split (as used in TimeMixer paper)
    n = len(data)
    train_end = int(n * 0.7)
    val_end = int(n * 0.8)

    train_data = data[:train_end]
    val_data = data[train_end:val_end]
    test_data = data[val_end:]

    print(f"  Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")

    # Compute normalization stats from training data
    train_mean = train_data.mean(axis=0)
    train_std = train_data.std(axis=0)
    train_std[train_std == 0] = 1.0

    return {
        'train': train_data,
        'val': val_data,
        'test': test_data,
        'full': data,
        'mean': train_mean,
        'std': train_std,
        'feature_cols': feature_cols,
        'dates': df[date_col].values if date_col in df.columns else None,
    }


def create_test_windows(test_data, full_data, test_start_idx,
                         seq_len=96, pred_len=96, stride=1):
    """
    Create sliding window test samples.

    Each sample: (input[seq_len], target[pred_len])
    The input may extend into val set for the first few windows.
    """
    inputs = []
    targets = []

    # We need seq_len context before each prediction window
    # The test data starts at test_start_idx in the full data
    start = test_start_idx  # First test prediction starts here
    end = len(full_data) - pred_len + 1

    for i in range(start, end, stride):
        input_start = i - seq_len
        if input_start < 0:
            continue
        x = full_data[input_start:i]          # [seq_len, features]
        y = full_data[i:i + pred_len]          # [pred_len, features]
        if len(x) == seq_len and len(y) == pred_len:
            inputs.append(x)
            targets.append(y)

    inputs = np.array(inputs)   # [N, seq_len, features]
    targets = np.array(targets)  # [N, pred_len, features]

    return inputs, targets


def build_config(pred_len):
    """Build model config for the Traffic dataset with given horizon."""
    return SimpleNamespace(
        task_name='long_term_forecast',
        seq_len=96,
        label_len=48,
        pred_len=pred_len,
        enc_in=862,
        dec_in=862,
        c_out=862,
        d_model=32,
        n_heads=8,
        e_layers=2,
        d_layers=1,
        d_ff=64,
        factor=3,
        dropout=0.1,
        embed='timeF',
        freq='h',
        channel_independence=1,
        decomp_method='moving_avg',
        moving_avg=25,
        down_sampling_layers=3,
        down_sampling_window=2,
        down_sampling_method='avg',
        use_norm=1,
        use_future_temporal_feature=0,
        output_attention=False,
        use_amp=False,
        top_k=5,
        adaptive_use_horizon=True,
        adaptive_temperature=1.0,
        max_horizon=720,
    )


@torch.no_grad()
def run_evaluation(model, inputs, targets, batch_size=8, device='cpu'):
    """
    Run model on test windows and compute metrics.

    Returns:
        predictions: np.ndarray [N, pred_len, features]
        metrics: dict with MSE, MAE, etc.
        per_sample_mse: np.ndarray [N] — MSE per sample for analysis
    """
    model.eval()
    all_preds = []
    n_samples = len(inputs)

    for start in range(0, n_samples, batch_size):
        end = min(start + batch_size, n_samples)
        batch_x = torch.tensor(inputs[start:end], dtype=torch.float32).to(device)

        # Model expects (x_enc, x_mark_enc, x_dec, x_mark_dec)
        output = model(batch_x, None, None, None)

        # The model with channel_independence=1 outputs [B, pred_len, c_out]
        preds = output.cpu().numpy()
        all_preds.append(preds)

        if (start // batch_size) % 50 == 0:
            print(f"    Batch {start // batch_size + 1}/{(n_samples + batch_size - 1) // batch_size}")

    predictions = np.concatenate(all_preds, axis=0)  # [N, pred_len, features]

    # Compute global metrics
    metrics = compute_all_metrics(predictions, targets)

    # Per-sample MSE for error distribution analysis
    per_sample_mse = np.mean((predictions - targets) ** 2, axis=(1, 2))

    # Per-horizon MSE (average error at each time step in the prediction)
    per_step_mse = np.mean((predictions - targets) ** 2, axis=(0, 2))  # [pred_len]

    # Per-channel MSE
    per_channel_mse = np.mean((predictions - targets) ** 2, axis=(0, 1))  # [features]

    return predictions, metrics, {
        'per_sample_mse': per_sample_mse,
        'per_step_mse': per_step_mse,
        'per_channel_mse': per_channel_mse,
    }


def generate_plots(all_results, traffic_data, output_dir):
    """Generate comprehensive visualization plots."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        'figure.figsize': (14, 6),
        'figure.dpi': 150,
        'font.family': 'sans-serif',
        'font.size': 12,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.grid': True,
        'grid.alpha': 0.3,
    })

    os.makedirs(output_dir, exist_ok=True)
    horizons = sorted(all_results.keys())

    # -- 1. Horizon Performance: Our Results vs Paper -------------
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    our_mse = [all_results[h]['metrics']['mse'] for h in horizons]
    our_mae = [all_results[h]['metrics']['mae'] for h in horizons]
    paper_mse = [PAPER_REFERENCE[h]['mse'] for h in horizons]
    paper_mae = [PAPER_REFERENCE[h]['mae'] for h in horizons]

    # MSE comparison
    axes[0].plot(horizons, paper_mse, 'o--', color='#4A90D9', linewidth=2.5,
                 markersize=10, label='Paper Reference', zorder=5)
    axes[0].plot(horizons, our_mse, 's-', color='#E74C3C', linewidth=2.5,
                 markersize=10, label='Our Reproduction', zorder=5)
    for i, h in enumerate(horizons):
        axes[0].annotate(f'{our_mse[i]:.4f}', (h, our_mse[i]),
                         textcoords="offset points", xytext=(0, 12),
                         ha='center', fontsize=9, color='#E74C3C', fontweight='bold')
        axes[0].annotate(f'{paper_mse[i]:.3f}', (h, paper_mse[i]),
                         textcoords="offset points", xytext=(0, -16),
                         ha='center', fontsize=9, color='#4A90D9')
    axes[0].set_xlabel('Forecast Horizon', fontsize=13)
    axes[0].set_ylabel('MSE', fontsize=13)
    axes[0].set_title('MSE: Ours vs Paper', fontsize=14, fontweight='bold')
    axes[0].legend(fontsize=11)
    axes[0].set_xticks(horizons)

    # MAE comparison
    axes[1].plot(horizons, paper_mae, 'o--', color='#4A90D9', linewidth=2.5,
                 markersize=10, label='Paper Reference', zorder=5)
    axes[1].plot(horizons, our_mae, 's-', color='#E74C3C', linewidth=2.5,
                 markersize=10, label='Our Reproduction', zorder=5)
    for i, h in enumerate(horizons):
        axes[1].annotate(f'{our_mae[i]:.4f}', (h, our_mae[i]),
                         textcoords="offset points", xytext=(0, 12),
                         ha='center', fontsize=9, color='#E74C3C', fontweight='bold')
        axes[1].annotate(f'{paper_mae[i]:.3f}', (h, paper_mae[i]),
                         textcoords="offset points", xytext=(0, -16),
                         ha='center', fontsize=9, color='#4A90D9')
    axes[1].set_xlabel('Forecast Horizon', fontsize=13)
    axes[1].set_ylabel('MAE', fontsize=13)
    axes[1].set_title('MAE: Ours vs Paper', fontsize=14, fontweight='bold')
    axes[1].legend(fontsize=11)
    axes[1].set_xticks(horizons)

    fig.suptitle('Traffic Dataset — Horizon Performance Comparison',
                 fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'traffic_horizon_comparison.png'),
                bbox_inches='tight')
    plt.close()
    print(f"  [OK] Saved: traffic_horizon_comparison.png")

    # -- 2. Per-horizon forecast vs actual (sample overlay) -------
    for h in horizons:
        r = all_results[h]
        preds = r['predictions']
        targets = r['targets']

        fig, axes = plt.subplots(2, 2, figsize=(18, 10))
        channels = [0, 100, 400, 861]  # Sample different channels
        channel_labels = [f'Sensor {c}' for c in channels]

        for ax, ch, ch_label in zip(axes.flat, channels, channel_labels):
            n_show = min(200, len(targets))
            # Show ground truth and predictions for the first sample
            sample_idx = 0
            true_vals = targets[sample_idx, :, ch]
            pred_vals = preds[sample_idx, :, ch]
            t = np.arange(len(true_vals))

            ax.plot(t, true_vals, color='#4A90D9', linewidth=1.5,
                    label='Actual', alpha=0.9)
            ax.plot(t, pred_vals, color='#E74C3C', linewidth=1.5,
                    label='Predicted', alpha=0.9, linestyle='--')
            ax.fill_between(t, true_vals, pred_vals, alpha=0.12, color='#E74C3C')
            ax.set_title(ch_label, fontsize=12, fontweight='bold')
            ax.set_xlabel('Time Step')
            ax.set_ylabel('Value')
            ax.legend(fontsize=9, loc='upper right')

        fig.suptitle(f'Traffic — Forecast vs Actual (Horizon {h})',
                     fontsize=15, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'traffic_h{h}_forecast_vs_actual.png'),
                    bbox_inches='tight')
        plt.close()
        print(f"  [OK] Saved: traffic_h{h}_forecast_vs_actual.png")

    # -- 3. Error distribution per horizon ------------------------
    fig, axes = plt.subplots(1, len(horizons), figsize=(5 * len(horizons), 5))
    if len(horizons) == 1:
        axes = [axes]
    colors = ['#3498DB', '#E74C3C', '#2ECC71', '#F39C12']

    for ax, h, color in zip(axes, horizons, colors):
        per_sample = all_results[h]['analysis']['per_sample_mse']
        ax.hist(per_sample, bins=50, color=color, alpha=0.7, edgecolor='white')
        ax.axvline(np.mean(per_sample), color='black', linestyle='--',
                   linewidth=2, label=f'Mean={np.mean(per_sample):.4f}')
        ax.axvline(np.median(per_sample), color='gray', linestyle=':',
                   linewidth=2, label=f'Median={np.median(per_sample):.4f}')
        ax.set_title(f'Horizon {h}', fontsize=13, fontweight='bold')
        ax.set_xlabel('Sample MSE')
        ax.set_ylabel('Count')
        ax.legend(fontsize=9)

    fig.suptitle('Traffic — Per-Sample Error Distribution',
                 fontsize=15, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'traffic_error_distribution.png'),
                bbox_inches='tight')
    plt.close()
    print(f"  [OK] Saved: traffic_error_distribution.png")

    # -- 4. Error accumulation across prediction steps ------------
    fig, ax = plt.subplots(figsize=(14, 6))
    for h, color in zip(horizons, colors):
        per_step = all_results[h]['analysis']['per_step_mse']
        ax.plot(np.arange(1, len(per_step) + 1), per_step, '-',
                color=color, linewidth=2, label=f'Horizon {h}', alpha=0.85)

    ax.set_xlabel('Prediction Step', fontsize=13)
    ax.set_ylabel('MSE', fontsize=13)
    ax.set_title('Traffic — Error Accumulation Over Prediction Steps',
                 fontsize=15, fontweight='bold')
    ax.legend(fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'traffic_error_accumulation.png'),
                bbox_inches='tight')
    plt.close()
    print(f"  [OK] Saved: traffic_error_accumulation.png")

    # -- 5. Per-channel MSE heatmap (top/bottom channels) ---------
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))

    for ax, h in zip(axes, [horizons[0], horizons[-1]]):
        per_ch = all_results[h]['analysis']['per_channel_mse']
        top_k = 30
        sorted_idx = np.argsort(per_ch)

        # Show best and worst channels
        best = sorted_idx[:top_k]
        worst = sorted_idx[-top_k:]

        combined = np.concatenate([best, worst])
        combined_mse = per_ch[combined]

        colors_bar = ['#2ECC71'] * top_k + ['#E74C3C'] * top_k
        ax.barh(range(len(combined)), combined_mse, color=colors_bar, alpha=0.8)
        ax.set_yticks(range(len(combined)))
        ax.set_yticklabels([f'Ch {i}' for i in combined], fontsize=7)
        ax.set_xlabel('MSE')
        ax.set_title(f'Horizon {h}: Best/Worst {top_k} Channels',
                     fontsize=12, fontweight='bold')
        ax.axvline(np.mean(per_ch), color='black', linestyle='--',
                   linewidth=1.5, label=f'Mean={np.mean(per_ch):.4f}')
        ax.legend(fontsize=9)

    fig.suptitle('Traffic — Per-Channel Error Analysis',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'traffic_per_channel_error.png'),
                bbox_inches='tight')
    plt.close()
    print(f"  [OK] Saved: traffic_per_channel_error.png")

    # -- 6. Summary bar chart -------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    x = np.arange(len(horizons))
    width = 0.35

    axes[0].bar(x - width/2, paper_mse, width, color='#4A90D9',
                label='Paper', alpha=0.85, edgecolor='white')
    axes[0].bar(x + width/2, our_mse, width, color='#E74C3C',
                label='Ours', alpha=0.85, edgecolor='white')
    for i in range(len(horizons)):
        delta = our_mse[i] - paper_mse[i]
        sign = '+' if delta > 0 else ''
        axes[0].text(i + width/2, our_mse[i] + 0.005, f'{sign}{delta:.4f}',
                     ha='center', va='bottom', fontsize=8, fontweight='bold',
                     color='#E74C3C' if delta > 0 else '#2ECC71')
    axes[0].set_xlabel('Horizon')
    axes[0].set_ylabel('MSE')
    axes[0].set_title('MSE Comparison', fontweight='bold')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(horizons)
    axes[0].legend()

    axes[1].bar(x - width/2, paper_mae, width, color='#4A90D9',
                label='Paper', alpha=0.85, edgecolor='white')
    axes[1].bar(x + width/2, our_mae, width, color='#E74C3C',
                label='Ours', alpha=0.85, edgecolor='white')
    for i in range(len(horizons)):
        delta = our_mae[i] - paper_mae[i]
        sign = '+' if delta > 0 else ''
        axes[1].text(i + width/2, our_mae[i] + 0.005, f'{sign}{delta:.4f}',
                     ha='center', va='bottom', fontsize=8, fontweight='bold',
                     color='#E74C3C' if delta > 0 else '#2ECC71')
    axes[1].set_xlabel('Horizon')
    axes[1].set_ylabel('MAE')
    axes[1].set_title('MAE Comparison', fontweight='bold')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(horizons)
    axes[1].legend()

    fig.suptitle('Traffic — Paper vs Our Reproduction',
                 fontsize=15, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'traffic_paper_comparison_bars.png'),
                bbox_inches='tight')
    plt.close()
    print(f"  [OK] Saved: traffic_paper_comparison_bars.png")


def save_results(all_results, output_dir):
    """Save results in JSON, CSV, and Markdown formats."""
    os.makedirs(output_dir, exist_ok=True)
    horizons = sorted(all_results.keys())

    # -- JSON -----------------------------------------------------
    json_results = {}
    for h in horizons:
        r = all_results[h]
        json_results[str(h)] = {
            'metrics': r['metrics'],
            'paper_reference': PAPER_REFERENCE[h],
            'delta_mse': r['metrics']['mse'] - PAPER_REFERENCE[h]['mse'],
            'delta_mae': r['metrics']['mae'] - PAPER_REFERENCE[h]['mae'],
            'n_test_samples': r['n_samples'],
            'checkpoint': r['checkpoint'],
            'inference_time_sec': r['inference_time'],
        }

    json_path = os.path.join(output_dir, 'traffic_evaluation.json')
    with open(json_path, 'w') as f:
        json.dump({
            'dataset': 'Traffic',
            'model': 'TimeMixer (Baseline)',
            'seq_len': 96,
            'n_features': 862,
            'timestamp': datetime.now().isoformat(),
            'results': json_results,
        }, f, indent=2)
    print(f"  [OK] JSON: {json_path}")

    # -- CSV ------------------------------------------------------
    csv_path = os.path.join(output_dir, 'traffic_evaluation.csv')
    with open(csv_path, 'w') as f:
        f.write('horizon,mse,mae,rmse,mape,smape,paper_mse,paper_mae,delta_mse,delta_mae,n_samples,inference_sec\n')
        for h in horizons:
            r = all_results[h]
            m = r['metrics']
            p = PAPER_REFERENCE[h]
            f.write(f"{h},{m['mse']:.6f},{m['mae']:.6f},{m['rmse']:.6f},"
                    f"{m['mape']:.4f},{m['smape']:.4f},"
                    f"{p['mse']:.3f},{p['mae']:.3f},"
                    f"{m['mse'] - p['mse']:.6f},{m['mae'] - p['mae']:.6f},"
                    f"{r['n_samples']},{r['inference_time']:.2f}\n")
    print(f"  [OK] CSV:  {csv_path}")

    # -- Markdown Report ------------------------------------------
    md_path = os.path.join(output_dir, 'traffic_evaluation_report.md')
    with open(md_path, 'w') as f:
        f.write("# Traffic Dataset — Post-Evaluation Report\n\n")
        f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"**Model:** TimeMixer (Baseline)  \n")
        f.write(f"**Input Length:** 96  \n")
        f.write(f"**Features:** 862 traffic sensors  \n\n")

        f.write("## Results Summary\n\n")
        f.write("| Horizon | MSE | MAE | RMSE | Paper MSE | Paper MAE | Delta MSE | Delta MAE |\n")
        f.write("|---------|-----|-----|------|-----------|-----------|-------|-------|\n")
        for h in horizons:
            m = all_results[h]['metrics']
            p = PAPER_REFERENCE[h]
            dm = m['mse'] - p['mse']
            da = m['mae'] - p['mae']
            dm_str = f"{'[+]' if dm <= 0 else '[!]'} {dm:+.4f}"
            da_str = f"{'[+]' if da <= 0 else '[!]'} {da:+.4f}"
            f.write(f"| {h} | {m['mse']:.4f} | {m['mae']:.4f} | {m['rmse']:.4f} | "
                    f"{p['mse']:.3f} | {p['mae']:.3f} | {dm_str} | {da_str} |\n")

        f.write("\n## Detailed Metrics\n\n")
        for h in horizons:
            m = all_results[h]['metrics']
            r = all_results[h]
            f.write(f"### Horizon {h}\n\n")
            f.write(f"- **MSE:** {m['mse']:.6f}\n")
            f.write(f"- **MAE:** {m['mae']:.6f}\n")
            f.write(f"- **RMSE:** {m['rmse']:.6f}\n")
            f.write(f"- **MAPE:** {m['mape']:.4f}%\n")
            f.write(f"- **sMAPE:** {m['smape']:.4f}%\n")
            f.write(f"- **Test Samples:** {r['n_samples']}\n")
            f.write(f"- **Inference Time:** {r['inference_time']:.2f}s\n")
            f.write(f"- **Checkpoint:** `{r['checkpoint']}`\n\n")

        f.write("## Analysis\n\n")
        f.write("### Error Statistics Per Horizon\n\n")
        f.write("| Horizon | Mean Sample MSE | Median Sample MSE | Std Sample MSE | Max Sample MSE |\n")
        f.write("|---------|----------------|-------------------|----------------|----------------|\n")
        for h in horizons:
            ps = all_results[h]['analysis']['per_sample_mse']
            f.write(f"| {h} | {np.mean(ps):.6f} | {np.median(ps):.6f} | "
                    f"{np.std(ps):.6f} | {np.max(ps):.6f} |\n")

        f.write("\n### Per-Channel Error Summary\n\n")
        f.write("| Horizon | Best Channel MSE | Worst Channel MSE | Mean Channel MSE | Channel Std |\n")
        f.write("|---------|-----------------|-------------------|------------------|-------------|\n")
        for h in horizons:
            pc = all_results[h]['analysis']['per_channel_mse']
            f.write(f"| {h} | {np.min(pc):.6f} | {np.max(pc):.6f} | "
                    f"{np.mean(pc):.6f} | {np.std(pc):.6f} |\n")

        f.write("\n## Plots\n\n")
        f.write("- `traffic_horizon_comparison.png` — MSE/MAE across horizons (ours vs paper)\n")
        f.write("- `traffic_h{H}_forecast_vs_actual.png` — Forecast overlays per horizon\n")
        f.write("- `traffic_error_distribution.png` — Per-sample error histograms\n")
        f.write("- `traffic_error_accumulation.png` — Error growth over prediction steps\n")
        f.write("- `traffic_per_channel_error.png` — Best/worst sensor channels\n")
        f.write("- `traffic_paper_comparison_bars.png` — Side-by-side bar comparison\n")

    print(f"  [OK] Report: {md_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Post-evaluation of Traffic dataset across all horizons')
    parser.add_argument('--horizons', type=int, nargs='+',
                        default=[96, 192, 336, 720],
                        help='Horizons to evaluate (default: 96 192 336 720)')
    parser.add_argument('--device', type=str, default='cpu',
                        help='Device (cpu or cuda)')
    parser.add_argument('--batch-size', type=int, default=8,
                        help='Batch size for inference')
    parser.add_argument('--stride', type=int, default=1,
                        help='Stride for test windows (1=all, higher=faster)')
    parser.add_argument('--output-dir', type=str,
                        default='results/traffic_post_eval',
                        help='Output directory')
    args = parser.parse_args()

    print("=" * 70)
    print("  Traffic Dataset — Post-Evaluation")
    print("=" * 70)
    print(f"  Horizons:   {args.horizons}")
    print(f"  Device:     {args.device}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Stride:     {args.stride}")
    print(f"  Output:     {args.output_dir}")

    # -- Step 1: Find checkpoints ---------------------------------
    print(f"\n{'-'*50}")
    print("Step 1: Locating checkpoints")
    print(f"{'-'*50}")

    available = {}
    for h in args.horizons:
        ckpt = find_checkpoint(h)
        if ckpt:
            size_mb = os.path.getsize(ckpt) / (1024 * 1024)
            print(f"  [OK] Horizon {h}: {os.path.basename(ckpt)} ({size_mb:.1f} MB)")
            available[h] = ckpt
        else:
            print(f"  [X] Horizon {h}: NO CHECKPOINT FOUND")

    if not available:
        print("\n  ERROR: No checkpoints found. Cannot proceed.")
        sys.exit(1)

    # -- Step 2: Load data ----------------------------------------
    print(f"\n{'-'*50}")
    print("Step 2: Loading Traffic dataset")
    print(f"{'-'*50}")

    traffic_data = load_traffic_data()

    # -- Step 3: Evaluate each horizon ----------------------------
    print(f"\n{'-'*50}")
    print("Step 3: Running evaluation")
    print(f"{'-'*50}")

    all_results = {}
    test_start_idx = int(len(traffic_data['full']) * 0.8)

    for h, ckpt_path in sorted(available.items()):
        print(f"\n  === Horizon {h} ===")

        # Build config
        config = build_config(h)

        # Create test windows
        print(f"  Creating test windows (seq={config.seq_len}, pred={h}, stride={args.stride})...")
        inputs, targets = create_test_windows(
            traffic_data['test'],
            traffic_data['full'],
            test_start_idx,
            seq_len=config.seq_len,
            pred_len=h,
            stride=args.stride,
        )
        print(f"  Test samples: {len(inputs)}")

        # Normalize using training stats
        mean = traffic_data['mean']
        std = traffic_data['std']
        inputs_norm = (inputs - mean) / std
        targets_norm = (targets - mean) / std

        # Load model
        print(f"  Loading model from: {os.path.basename(ckpt_path)}")
        model = load_model(ckpt_path, config, 'baseline', args.device)

        # Run evaluation
        print(f"  Running inference...")
        t_start = time.time()
        predictions, metrics, analysis = run_evaluation(
            model, inputs_norm, targets_norm,
            batch_size=args.batch_size, device=args.device
        )
        t_elapsed = time.time() - t_start

        print(f"\n  Results for Horizon {h}:")
        print(format_metrics(metrics))
        print(f"  Paper MSE: {PAPER_REFERENCE[h]['mse']:.3f} | "
              f"Ours: {metrics['mse']:.4f} | "
              f"Delta: {metrics['mse'] - PAPER_REFERENCE[h]['mse']:+.4f}")
        print(f"  Paper MAE: {PAPER_REFERENCE[h]['mae']:.3f} | "
              f"Ours: {metrics['mae']:.4f} | "
              f"Delta: {metrics['mae'] - PAPER_REFERENCE[h]['mae']:+.4f}")
        print(f"  Inference time: {t_elapsed:.2f}s")

        all_results[h] = {
            'metrics': metrics,
            'analysis': analysis,
            'predictions': predictions,
            'targets': targets_norm,
            'n_samples': len(inputs),
            'checkpoint': ckpt_path,
            'inference_time': t_elapsed,
        }

        # Free GPU memory
        del model
        if args.device == 'cuda':
            torch.cuda.empty_cache()

    # -- Step 4: Generate plots -----------------------------------
    print(f"\n{'-'*50}")
    print("Step 4: Generating visualizations")
    print(f"{'-'*50}")

    plots_dir = os.path.join(args.output_dir, 'plots')
    generate_plots(all_results, traffic_data, plots_dir)

    # -- Step 5: Save results -------------------------------------
    print(f"\n{'-'*50}")
    print("Step 5: Saving results")
    print(f"{'-'*50}")

    save_results(all_results, args.output_dir)

    # -- Summary --------------------------------------------------
    print(f"\n{'='*70}")
    print("  POST-EVALUATION COMPLETE")
    print(f"{'='*70}")
    print(f"\n  {'Horizon':<10} {'MSE':<10} {'MAE':<10} {'Paper MSE':<12} {'Paper MAE':<12} {'Delta MSE':<10}")
    print(f"  {'-'*64}")
    for h in sorted(all_results.keys()):
        m = all_results[h]['metrics']
        p = PAPER_REFERENCE[h]
        dm = m['mse'] - p['mse']
        print(f"  {h:<10} {m['mse']:<10.4f} {m['mae']:<10.4f} "
              f"{p['mse']:<12.3f} {p['mae']:<12.3f} {dm:<+10.4f}")

    print(f"\n  Results saved to: {args.output_dir}/")
    print(f"  Plots saved to:   {plots_dir}/")
    print("=" * 70)


if __name__ == '__main__':
    main()
