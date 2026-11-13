"""
End-to-end prediction pipeline.
CSV → validate → preprocess → model → inverse transform → results

Usage:
    python inference/predict.py --csv data.csv --checkpoint checkpoints/... --pred-len 96
    python inference/predict.py --csv data.csv --model baseline --horizon 96
"""
import os
import sys
import json
import argparse
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from inference.preprocessing import validate_csv, prepare_input, inverse_transform, create_time_features


def load_model(checkpoint_path, config=None, model_type='baseline', device='cpu'):
    """
    Load a trained model from checkpoint.

    Args:
        checkpoint_path: path to .pth file
        config: dict or SimpleNamespace with model configuration
        model_type: 'baseline' or 'adaptive'
        device: 'cpu' or 'cuda'

    Returns:
        model: loaded PyTorch model in eval mode
    """
    if config is None:
        # Default Traffic config
        config = SimpleNamespace(
            task_name='long_term_forecast',
            seq_len=96,
            label_len=48,
            pred_len=96,
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
            adaptive_use_horizon=True,
            adaptive_temperature=1.0,
            max_horizon=720,
            top_k=5,
        )

    if model_type == 'baseline':
        from baseline.models.TimeMixer import Model
        model = Model(config)
    elif model_type == 'adaptive':
        from extension.models.AdaptiveTimeMixer import AdaptiveTimeMixerModel
        model = AdaptiveTimeMixerModel(config)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    # Load checkpoint
    if os.path.exists(checkpoint_path):
        state_dict = torch.load(checkpoint_path, map_location=device, weights_only=False)
        # Handle DataParallel wrapped checkpoints
        new_state_dict = {}
        for k, v in state_dict.items():
            name = k.replace('module.', '')
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict, strict=False)
        print(f"  Loaded checkpoint: {checkpoint_path}")
    else:
        print(f"  Warning: Checkpoint not found at {checkpoint_path}")

    model = model.to(device)
    model.eval()
    return model


def config_from_yaml(yaml_path):
    """Load model configuration from a YAML file."""
    import yaml
    with open(yaml_path, 'r') as f:
        cfg = yaml.safe_load(f)

    # Flatten into SimpleNamespace
    flat = {}
    flat['task_name'] = 'long_term_forecast'
    flat['output_attention'] = False
    flat['use_amp'] = False
    flat['top_k'] = 5

    if 'dataset' in cfg:
        for k, v in cfg['dataset'].items():
            flat[k] = v
    if 'model_params' in cfg:
        for k, v in cfg['model_params'].items():
            flat[k] = v
    if 'training' in cfg:
        for k, v in cfg['training'].items():
            flat[k] = v

    return SimpleNamespace(**flat)


def forecast(csv_path, checkpoint_path, pred_len=96, model_type='baseline',
             config=None, device='cpu', output_csv=None):
    """
    End-to-end forecasting pipeline.

    Args:
        csv_path: path to input CSV
        checkpoint_path: path to model checkpoint
        pred_len: forecast horizon
        model_type: 'baseline' or 'adaptive'
        config: model config (SimpleNamespace or dict). Auto-detected if None.
        device: 'cpu' or 'cuda'
        output_csv: optional path to save predictions

    Returns:
        results: dict with predictions, metrics, weights (if adaptive)
    """
    print(f"\n{'='*60}")
    print(f"TimeMixer Forecasting Pipeline")
    print(f"{'='*60}")

    # Step 1: Validate CSV
    print("\n[1/5] Validating CSV...")
    df = validate_csv(csv_path)

    # Step 2: Prepare input
    print("[2/5] Preparing model input...")
    seq_len = config.seq_len if config else 96
    model_input, prep_info = prepare_input(df, seq_len=seq_len)

    # Update config with actual feature count
    if config is None:
        config = SimpleNamespace(
            task_name='long_term_forecast',
            seq_len=seq_len,
            label_len=seq_len // 2,
            pred_len=pred_len,
            enc_in=prep_info['n_features'],
            dec_in=prep_info['n_features'],
            c_out=prep_info['n_features'],
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
    else:
        config.pred_len = pred_len

    # Step 3: Load model
    print(f"[3/5] Loading {model_type} model...")
    model = load_model(checkpoint_path, config, model_type, device)

    # Step 4: Run inference
    print("[4/5] Running forecast...")
    with torch.no_grad():
        x_enc = torch.tensor(model_input, dtype=torch.float32).to(device)
        x_mark_enc = None  # Simplified — no temporal marks for inference
        x_dec = None
        x_mark_dec = None

        output = model(x_enc, x_mark_enc, x_dec, x_mark_dec)
        predictions = output.cpu().numpy()

    # Step 5: Inverse transform
    print("[5/5] Post-processing predictions...")
    predictions_original = inverse_transform(predictions[0], prep_info['scaler_info'])

    # Compute simple metrics against last known values
    results = {
        'predictions': predictions_original,  # [pred_len, features]
        'input_data': prep_info['raw_input'],
        'column_names': prep_info['column_names'],
        'pred_len': pred_len,
        'model_type': model_type,
    }

    # Get adaptive weights if available
    if model_type == 'adaptive' and hasattr(model, 'get_last_weights'):
        weights = model.get_last_weights()
        if weights is not None:
            results['adaptive_weights'] = weights.cpu().numpy()

    # Save predictions
    if output_csv:
        pred_df = pd.DataFrame(
            predictions_original,
            columns=prep_info['column_names']
        )
        pred_df.to_csv(output_csv, index=False)
        print(f"\n  Predictions saved to: {output_csv}")

    print(f"\n{'='*60}")
    print(f"  Forecast generated: {pred_len} steps × {prep_info['n_features']} features")
    print(f"{'='*60}\n")

    return results


def main():
    parser = argparse.ArgumentParser(description='TimeMixer Inference Pipeline')
    parser.add_argument('--csv', type=str, required=True,
                        help='Path to input CSV file')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint (.pth)')
    parser.add_argument('--config', type=str, default=None,
                        help='Path to YAML config file')
    parser.add_argument('--model', type=str, default='baseline',
                        choices=['baseline', 'adaptive'],
                        help='Model type')
    parser.add_argument('--horizon', type=int, default=96,
                        help='Forecast horizon (pred_len)')
    parser.add_argument('--device', type=str, default='cpu',
                        help='Device (cpu or cuda)')
    parser.add_argument('--output', type=str, default=None,
                        help='Output CSV path for predictions')
    args = parser.parse_args()

    config = None
    if args.config:
        config = config_from_yaml(args.config)

    import pandas as pd  # needed for output_csv
    results = forecast(
        csv_path=args.csv,
        checkpoint_path=args.checkpoint,
        pred_len=args.horizon,
        model_type=args.model,
        config=config,
        device=args.device,
        output_csv=args.output,
    )


if __name__ == '__main__':
    main()
