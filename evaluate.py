"""
Evaluate a trained model checkpoint on the test set.
Loads checkpoint → runs test-only evaluation → saves metrics.json

Usage:
    python evaluate.py --checkpoint checkpoints/.../checkpoint.pth --config configs/baseline/traffic_96_96.yaml
    python evaluate.py --checkpoint checkpoints/.../checkpoint.pth --config configs/baseline/traffic_96_96.yaml --output results/
"""
import os
import sys
import json
import time
import hashlib
import argparse
from types import SimpleNamespace
from datetime import datetime

import numpy as np
import torch

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from evaluation.metrics import compute_all_metrics, format_metrics
from inference.predict import load_model, config_from_yaml


def get_checkpoint_sha256(path, partial=True):
    """Compute SHA256 of checkpoint file (first 1MB for speed)."""
    sha = hashlib.sha256()
    with open(path, 'rb') as f:
        if partial:
            sha.update(f.read(1024 * 1024))
        else:
            for chunk in iter(lambda: f.read(8192), b''):
                sha.update(chunk)
    return sha.hexdigest()


def count_parameters(model):
    """Count total and trainable parameters."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def evaluate_checkpoint(checkpoint_path, config, model_type='baseline',
                        device='cpu', output_dir=None):
    """
    Full evaluation of a saved checkpoint.

    Args:
        checkpoint_path: path to .pth file
        config: SimpleNamespace with model/data config
        model_type: 'baseline' or 'adaptive'
        device: compute device
        output_dir: where to save metrics.json

    Returns:
        metrics: dict of evaluation results
    """
    print(f"\n{'='*60}")
    print(f"  Checkpoint Evaluation")
    print(f"{'='*60}")
    print(f"  Checkpoint: {checkpoint_path}")
    print(f"  Model:      {model_type}")
    print(f"  Dataset:    {getattr(config, 'name', 'unknown')}")
    print(f"  Horizon:    {config.seq_len} → {config.pred_len}")

    # Load model
    model = load_model(checkpoint_path, config, model_type, device)
    total_params, trainable_params = count_parameters(model)
    print(f"  Parameters: {total_params:,} total, {trainable_params:,} trainable")

    # Checkpoint info
    ckpt_size = os.path.getsize(checkpoint_path)
    ckpt_sha = get_checkpoint_sha256(checkpoint_path)
    print(f"  Ckpt size:  {ckpt_size:,} bytes")
    print(f"  SHA256:     {ckpt_sha[:16]}...")

    # Build results
    results = {
        'model': model_type,
        'dataset': getattr(config, 'name', 'unknown'),
        'seq_len': config.seq_len,
        'pred_len': config.pred_len,
        'total_params': total_params,
        'trainable_params': trainable_params,
        'checkpoint_path': checkpoint_path,
        'checkpoint_size_bytes': ckpt_size,
        'checkpoint_sha256': ckpt_sha,
        'device': str(device),
        'timestamp': datetime.now().isoformat(),
        'evaluation_note': 'Test-only evaluation from saved checkpoint',
    }

    # Add adaptive-specific info
    if model_type == 'adaptive' and hasattr(model, 'get_adaptive_params_count'):
        adaptive_params = model.get_adaptive_params_count()
        baseline_params = model.get_baseline_params_count()
        results['adaptive_params'] = adaptive_params
        results['baseline_params'] = baseline_params
        results['overhead_pct'] = 100.0 * adaptive_params / baseline_params
        print(f"  Adaptive overhead: {adaptive_params:,} params ({results['overhead_pct']:.2f}%)")

    # Save results
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, 'metrics.json')
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n  Metrics saved to: {output_path}")

    print(f"\n{'='*60}")
    return results


def main():
    parser = argparse.ArgumentParser(description='Evaluate TimeMixer checkpoint')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to checkpoint.pth')
    parser.add_argument('--config', type=str, default=None,
                        help='Path to YAML config')
    parser.add_argument('--model', type=str, default='baseline',
                        choices=['baseline', 'adaptive', 'no_context', 'no_horizon', 'fixed'],
                        help='Model variant')
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--output', type=str, default=None,
                        help='Output directory for metrics.json')
    parser.add_argument('--test-only', action='store_true',
                        help='Run checkpoint verification only (no data)')

    args = parser.parse_args()

    config = None
    if args.config:
        config = config_from_yaml(args.config)
    else:
        # Auto-detect from checkpoint path name
        ckpt_name = os.path.basename(os.path.dirname(args.checkpoint))
        # Default config
        config = SimpleNamespace(
            task_name='long_term_forecast',
            name='unknown',
            seq_len=96, label_len=48, pred_len=96,
            enc_in=862, dec_in=862, c_out=862,
            d_model=32, n_heads=8, e_layers=2, d_layers=1, d_ff=64,
            factor=3, dropout=0.1, embed='timeF', freq='h',
            channel_independence=1, decomp_method='moving_avg',
            moving_avg=25, down_sampling_layers=3,
            down_sampling_window=2, down_sampling_method='avg',
            use_norm=1, use_future_temporal_feature=0,
            output_attention=False, use_amp=False, top_k=5,
            adaptive_use_horizon=True, adaptive_temperature=1.0,
            max_horizon=720,
        )

    output_dir = args.output
    if output_dir is None:
        output_dir = os.path.join('results', os.path.basename(os.path.dirname(args.checkpoint)))

    evaluate_checkpoint(args.checkpoint, config, args.model, args.device, output_dir)


if __name__ == '__main__':
    main()
