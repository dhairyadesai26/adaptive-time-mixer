"""
Run Comprehensive Multi-Dataset Benchmark: Baseline vs. Adaptive-FMM
Covers ALL benchmark datasets:
  - ETTh1 (7 variates)
  - ETTh2 (7 variates)
  - ETTm1 (7 variates)
  - ETTm2 (7 variates)
  - Weather (21 variates)
  - Electricity (321 variates)
  - Solar (137 variates)
  - Traffic (862 variates)
"""
import os
import sys
import json
import time
from types import SimpleNamespace
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from evaluation.metrics import compute_all_metrics
from inference.predict import load_model

# -------------------------------------------------------------------------
# Dataset Loaders
# -------------------------------------------------------------------------
def load_dataset_data(dataset_name):
    """Load and normalize any benchmark dataset."""
    base = os.path.join(PROJECT_ROOT, 'dataset')
    
    if dataset_name == 'ETTh1':
        p = os.path.join(base, 'ETT-small', 'ETTh1.csv')
        df = pd.read_csv(p)
        cols = [c for c in df.columns if c != 'date']
        data = df[cols].values.astype(np.float32)
    elif dataset_name == 'ETTh2':
        p = os.path.join(base, 'ETT-small', 'ETTh2.csv')
        df = pd.read_csv(p)
        cols = [c for c in df.columns if c != 'date']
        data = df[cols].values.astype(np.float32)
    elif dataset_name == 'ETTm1':
        p = os.path.join(base, 'ETT-small', 'ETTm1.csv')
        df = pd.read_csv(p)
        cols = [c for c in df.columns if c != 'date']
        data = df[cols].values.astype(np.float32)
    elif dataset_name == 'ETTm2':
        p = os.path.join(base, 'ETT-small', 'ETTm2.csv')
        df = pd.read_csv(p)
        cols = [c for c in df.columns if c != 'date']
        data = df[cols].values.astype(np.float32)
    elif dataset_name == 'Weather':
        p = os.path.join(base, 'weather', 'weather.csv')
        df = pd.read_csv(p)
        cols = [c for c in df.columns if c != 'date']
        data = df[cols].values.astype(np.float32)
    elif dataset_name == 'Electricity':
        p = os.path.join(base, 'electricity', 'electricity.csv')
        df = pd.read_csv(p)
        cols = [c for c in df.columns if c != 'date']
        data = df[cols].values.astype(np.float32)
    elif dataset_name == 'Solar':
        p = os.path.join(base, 'solar', 'solar_AL.txt')
        data = np.loadtxt(p, delimiter=',').astype(np.float32)
    elif dataset_name == 'Traffic':
        p = os.path.join(base, 'traffic', 'traffic.csv')
        df = pd.read_csv(p)
        cols = [c for c in df.columns if c != 'date']
        data = df[cols].values.astype(np.float32)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    n = len(data)
    train_end = int(n * 0.7)
    val_end = int(n * 0.8)
    return {
        'full': data,
        'train': data[:train_end],
        'val': data[train_end:val_end],
        'test': data[val_end:],
        'n_features': data.shape[1],
        'val_start': train_end,
        'test_start': val_end
    }


def create_windows(full_data, start_idx, seq_len=96, pred_len=96, stride=16, max_windows=30):
    inputs, targets = [], []
    end_idx = len(full_data) - pred_len + 1
    for i in range(start_idx, end_idx, stride):
        s = i - seq_len
        if s < 0:
            continue
        x = full_data[s:i]
        y = full_data[i:i + pred_len]
        if len(x) == seq_len and len(y) == pred_len:
            inputs.append(x)
            targets.append(y)
        if len(inputs) >= max_windows:
            break
    return np.array(inputs), np.array(targets)


# -------------------------------------------------------------------------
# Checkpoint & Config Mappings
# -------------------------------------------------------------------------
def get_checkpoint_and_config(dataset_name, pred_len=96):
    ckpt_dir = os.path.join(PROJECT_ROOT, 'checkpoints')
    
    # Defaults
    d_model = 16
    d_ff = 32
    factor = 1
    e_layers = 2
    n_heads = 8

    if dataset_name == 'ETTh1':
        pattern = f'long_term_forecast_ETTh1_96_{pred_len}'
        n_feats = 7
    elif dataset_name == 'ETTh2':
        pattern = f'long_term_forecast_ETTh2_96_{pred_len}'
        n_feats = 7
    elif dataset_name == 'ETTm1':
        pattern = f'long_term_forecast_ETTm1_96_{pred_len}'
        n_feats = 7
    elif dataset_name == 'ETTm2':
        pattern = f'long_term_forecast_ETTm2_96_{pred_len}'
        n_feats = 7
        d_model = 32
        d_ff = 64
    elif dataset_name == 'Weather':
        pattern = f'long_term_forecast_Weather_96_{pred_len}'
        n_feats = 21
        factor = 1 if pred_len == 96 else 3
    elif dataset_name == 'Electricity':
        pattern = f'long_term_forecast_Electricity_96_{pred_len}'
        n_feats = 321
        factor = 3
    elif dataset_name == 'Solar':
        pattern = f'long_term_forecast_solar_96_{pred_len}'
        n_feats = 137
        d_model = 512
        d_ff = 2048
        e_layers = 3
        factor = 3
    elif dataset_name == 'Traffic':
        pattern = f'long_term_forecast_Traffic_96_{pred_len}'
        n_feats = 862
        d_model = 32
        d_ff = 64
        factor = 3
    else:
        return None, None

    # Search checkpoint file
    ckpt_path = None
    for item in os.listdir(ckpt_dir):
        if pattern.lower() in item.lower():
            p = os.path.join(ckpt_dir, item)
            if os.path.isdir(p):
                target = os.path.join(p, 'checkpoint.pth')
                if os.path.exists(target):
                    ckpt_path = target
                    break
            elif p.endswith('.pth'):
                ckpt_path = p
                break
                
    if not ckpt_path and dataset_name == 'Traffic':
        if pred_len == 336:
            ckpt_path = os.path.join(ckpt_dir, 'Traffic_96_336_checkpoint.pth')
        elif pred_len == 720:
            ckpt_path = os.path.join(ckpt_dir, 'Traffic_96_720_checkpoint.pth')

    cfg = SimpleNamespace(
        task_name='long_term_forecast',
        seq_len=96,
        label_len=48,
        pred_len=pred_len,
        enc_in=n_feats,
        dec_in=n_feats,
        c_out=n_feats,
        d_model=d_model,
        n_heads=n_heads,
        e_layers=e_layers,
        d_layers=1,
        d_ff=d_ff,
        factor=factor,
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
    return ckpt_path, cfg


# -------------------------------------------------------------------------
# Multiscale Extraction & Adaptive Training
# -------------------------------------------------------------------------
def extract_representations(model, inputs, targets, batch_size=8):
    model.eval()
    cached = []
    n_samples = len(inputs)
    with torch.no_grad():
        for start in range(0, n_samples, batch_size):
            end = min(start + batch_size, n_samples)
            bx = torch.tensor(inputs[start:end], dtype=torch.float32)
            by = torch.tensor(targets[start:end], dtype=torch.float32)
            B, T, N = bx.shape

            x_enc, _ = model._Model__multi_scale_process_inputs(bx, None)
            x_list = []
            for j, xi in enumerate(x_enc):
                xi = model.normalize_layers[j](xi, 'norm')
                if model.channel_independence == 1:
                    xi = xi.permute(0, 2, 1).contiguous().reshape(B * N, xi.shape[1], 1)
                x_list.append(xi)

            enc_out_list = []
            x_list = model.pre_enc(x_list)
            for j, xi in enumerate(x_list[0]):
                enc_out = model.enc_embedding(xi, None)
                enc_out_list.append(enc_out)

            for j in range(model.layer):
                enc_out_list = model.pdm_blocks[j](enc_out_list)

            dec_out_list = model.future_multi_mixing(B, enc_out_list, x_list)

            scale_contexts = []
            for enc_out in enc_out_list:
                pooled = enc_out.mean(dim=1)
                if model.channel_independence == 1 and N > 1:
                    pooled = pooled.reshape(B, N, -1).mean(dim=1)
                scale_contexts.append(pooled)

            stacked_f = torch.stack(dec_out_list, dim=-1)
            mean = model.normalize_layers[0].mean.clone()
            stdev = model.normalize_layers[0].stdev.clone()
            cached.append((scale_contexts, stacked_f, by, mean, stdev))

    return cached


def run_benchmark_for_dataset(dataset_name, pred_len=96, epochs=5):
    print(f"\n=======================================================", flush=True)
    print(f"  BENCHMARKING: {dataset_name} (Horizon = {pred_len})", flush=True)
    print(f"=======================================================", flush=True)

    ckpt, cfg = get_checkpoint_and_config(dataset_name, pred_len)
    if not ckpt or not os.path.exists(ckpt):
        print(f"  [SKIPPED] Checkpoint not found for {dataset_name}-{pred_len}", flush=True)
        return None

    data_dict = load_dataset_data(dataset_name)
    val_in, val_tgt = create_windows(data_dict['full'], data_dict['val_start'], seq_len=96, pred_len=pred_len, stride=16, max_windows=20)
    test_in, test_tgt = create_windows(data_dict['full'], data_dict['test_start'], seq_len=96, pred_len=pred_len, stride=16, max_windows=30)

    print(f"  Loaded dataset: {data_dict['n_features']} variates. Test samples: {len(test_in)}", flush=True)
    model = load_model(ckpt, cfg, 'adaptive', 'cpu')

    val_cached = extract_representations(model, val_in, val_tgt)
    test_cached = extract_representations(model, test_in, test_tgt)

    num_scales = val_cached[0][1].shape[-1]
    d_model = val_cached[0][0][0].shape[-1]

    # 1. Baseline Evaluation (Uniform Sum)
    test_preds_base, test_tgts = [], []
    with torch.no_grad():
        for scale_ctxs, stacked_f, by, mean, stdev in test_cached:
            base_f = stacked_f.sum(dim=-1)
            base_f_denorm = base_f * stdev + mean
            test_preds_base.append(base_f_denorm.cpu().numpy())
            test_tgts.append(by.cpu().numpy())

    p_base = np.concatenate(test_preds_base, axis=0)
    t = np.concatenate(test_tgts, axis=0)
    base_metrics = compute_all_metrics(p_base, t)

    # 2. Adaptive-FMM Optimization
    scale_projs = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(num_scales)])
    combine = nn.Sequential(nn.Linear(d_model * num_scales, d_model), nn.GELU())
    horizon_encoder = nn.Sequential(
        nn.Linear(1, d_model // 2),
        nn.GELU(),
        nn.Linear(d_model // 2, d_model),
    )
    weight_net = nn.Sequential(
        nn.Linear(d_model * 2, d_model),
        nn.GELU(),
        nn.Dropout(0.1),
        nn.Linear(d_model, d_model // 2),
        nn.GELU(),
        nn.Linear(d_model // 2, num_scales),
    )
    nn.init.zeros_(weight_net[-1].weight)
    nn.init.zeros_(weight_net[-1].bias)

    params = (list(scale_projs.parameters()) + list(combine.parameters()) +
              list(horizon_encoder.parameters()) + list(weight_net.parameters()))
    optimizer = torch.optim.Adam(params, lr=2e-3, weight_decay=1e-4)

    h_val = torch.tensor([[float(pred_len)]])
    for ep in range(epochs):
        for scale_ctxs, stacked_f, by, mean, stdev in val_cached:
            optimizer.zero_grad()
            proj = [scale_projs[j](ctx) for j, ctx in enumerate(scale_ctxs)]
            ctx = combine(torch.cat(proj, dim=-1))
            h_emb = horizon_encoder(h_val).expand(ctx.size(0), -1)
            cond = torch.cat([ctx, h_emb], dim=-1)
            logits = weight_net(cond)
            w = F.softmax(logits, dim=-1) * num_scales
            f = (stacked_f * w.unsqueeze(1).unsqueeze(2)).sum(dim=-1)
            f_denorm = f * stdev + mean
            loss = F.mse_loss(f_denorm, by)
            loss.backward()
            optimizer.step()

    test_preds_adapt = []
    with torch.no_grad():
        for scale_ctxs, stacked_f, by, mean, stdev in test_cached:
            proj = [scale_projs[j](ctx) for j, ctx in enumerate(scale_ctxs)]
            ctx = combine(torch.cat(proj, dim=-1))
            h_emb = horizon_encoder(h_val).expand(ctx.size(0), -1)
            cond = torch.cat([ctx, h_emb], dim=-1)
            logits = weight_net(cond)
            w = F.softmax(logits, dim=-1) * num_scales
            f = (stacked_f * w.unsqueeze(1).unsqueeze(2)).sum(dim=-1)
            f_denorm = f * stdev + mean
            test_preds_adapt.append(f_denorm.cpu().numpy())

    p_adapt = np.concatenate(test_preds_adapt, axis=0)
    adapt_metrics = compute_all_metrics(p_adapt, t)

    delta_mse = (adapt_metrics['mse'] - base_metrics['mse']) / base_metrics['mse'] * 100.0
    print(f"  [BASELINE]     MSE: {base_metrics['mse']:.5f} | MAE: {base_metrics['mae']:.5f}", flush=True)
    print(f"  [ADAPTIVE-FMM] MSE: {adapt_metrics['mse']:.5f} | MAE: {adapt_metrics['mae']:.5f}", flush=True)
    print(f"  [IMPROVEMENT]  Delta MSE: {delta_mse:.2f}%", flush=True)

    return {
        'dataset': dataset_name,
        'horizon': pred_len,
        'n_features': data_dict['n_features'],
        'baseline_mse': float(base_metrics['mse']),
        'baseline_mae': float(base_metrics['mae']),
        'adaptive_mse': float(adapt_metrics['mse']),
        'adaptive_mae': float(adapt_metrics['mae']),
        'delta_mse_pct': float(delta_mse),
        'overhead_params': sum(p.numel() for p in params),
    }


def main():
    print("=" * 70, flush=True)
    print("  EXECUTING COMPREHENSIVE BENCHMARK FOR ALL DATASETS", flush=True)
    print("=" * 70, flush=True)

    benchmark_tasks = [
        ('ETTh1', 96), ('ETTh1', 192),
        ('ETTh2', 96), ('ETTh2', 192),
        ('ETTm1', 96), ('ETTm1', 192),
        ('ETTm2', 96), ('ETTm2', 192),
        ('Weather', 96), ('Weather', 192), ('Weather', 336),
        ('Electricity', 96), ('Electricity', 192),
        ('Solar', 96), ('Solar', 192),
        ('Traffic', 96), ('Traffic', 192), ('Traffic', 336), ('Traffic', 720)
    ]

    all_results = []
    for ds, h in benchmark_tasks:
        res = run_benchmark_for_dataset(ds, pred_len=h, epochs=5)
        if res:
            all_results.append(res)

    # Save to CSV
    df = pd.DataFrame(all_results)
    out_csv = os.path.join(PROJECT_ROOT, 'results', 'all_datasets_benchmark_results.csv')
    df.to_csv(out_csv, index=False)
    print(f"\nAll benchmark results saved to: {out_csv}", flush=True)

    # Save to JSON
    out_json = os.path.join(PROJECT_ROOT, 'results', 'all_datasets_benchmark_summary.json')
    with open(out_json, 'w') as f:
        json.dump(all_results, f, indent=2)

    # Plot Multi-Dataset Comparison
    plots_dir = os.path.join(PROJECT_ROOT, 'results', 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    
    # Filter for H=96 across all datasets for a clean comparison plot
    h96_df = df[df['horizon'] == 96]
    plt.figure(figsize=(12, 6), dpi=300)
    x = np.arange(len(h96_df))
    width = 0.35
    
    # Normalized MSE relative to baseline = 1.0 for visual comparison across heterogeneous scales
    rel_base = np.ones(len(h96_df))
    rel_adapt = h96_df['adaptive_mse'].values / h96_df['baseline_mse'].values

    plt.bar(x - width/2, rel_base, width, label='Baseline TimeMixer (Normalized = 1.0)', color='#94A3B8')
    plt.bar(x + width/2, rel_adapt, width, label='Adaptive-FMM (Proposed)', color='#2563EB')
    plt.axhline(1.0, color='red', linestyle='--', alpha=0.7)
    plt.xticks(x, h96_df['dataset'].values, fontsize=11, fontweight='bold')
    plt.ylabel("Relative Normalized MSE (Lower is Better)", fontsize=12)
    plt.title("Cross-Dataset Generalization: Adaptive-FMM vs. Baseline (Horizon 96)", fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(axis='y', linestyle=':', alpha=0.6)
    plt.tight_layout()
    plot_file = os.path.join(plots_dir, 'all_datasets_benchmark_comparison.png')
    plt.savefig(plot_file)
    plt.close()
    print(f"Cross-dataset comparison figure saved to: {plot_file}", flush=True)

    print("\n" + "=" * 70, flush=True)
    print("  ALL DATASETS BENCHMARK COMPLETED SUCCESSFULLY!", flush=True)
    print("=" * 70, flush=True)


if __name__ == '__main__':
    main()

