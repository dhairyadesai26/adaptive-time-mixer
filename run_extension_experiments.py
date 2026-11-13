"""
Run Extension Experiments & Ablations for Adaptive TimeMixer.

Executes:
  1. Full Baseline vs. Adaptive-FMM on Traffic (96, 192, 336, 720)
  2. Complete Ablation Suite:
       - Baseline (Uniform FMM)
       - Adaptive-FMM (Full model: context + horizon)
       - NoContext Ablation (horizon conditioning only)
       - NoHorizon Ablation (context conditioning only)
       - FixedWeight Ablation (learned static weights)
  3. Multi-Seed Stability Analysis (Seeds 2024, 2025, 2026)
  4. Secondary Validation on Weather (96, 192)
  5. Publication Plots & Tables (CSV, JSON, LaTeX, Markdown)
"""
import os
import sys
import json
import time
import argparse
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
from post_evaluate_traffic import find_checkpoint, build_config, load_traffic_data, create_test_windows


def load_weather_data():
    """Load and prepare Weather dataset."""
    csv_path = os.path.join(PROJECT_ROOT, 'dataset', 'weather', 'weather.csv')
    df = pd.read_csv(csv_path)
    feature_cols = [c for c in df.columns if c != 'date']
    data = df[feature_cols].values.astype(np.float32)

    n = len(data)
    train_end = int(n * 0.7)
    val_end = int(n * 0.8)

    train_data = data[:train_end]
    val_data = data[train_end:val_end]
    test_data = data[val_end:]

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
    }


def build_weather_config(pred_len):
    return SimpleNamespace(
        task_name='long_term_forecast',
        seq_len=96,
        label_len=48,
        pred_len=pred_len,
        enc_in=21,
        dec_in=21,
        c_out=21,
        d_model=16,
        n_heads=8,
        e_layers=2,
        d_layers=1,
        d_ff=32,
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


def extract_multiscale_representations(model, inputs, targets, device='cpu', batch_size=8):
    """
    Pass inputs through model backbone to extract:
    - scale_contexts: [B, num_scales, d_model] pooled representations
    - stacked_forecasts: [B, pred_len, C, num_scales] per-scale forecasts
    - targets: ground truth
    """
    model.eval()
    cached = []
    n_samples = len(inputs)

    with torch.no_grad():
        for start in range(0, n_samples, batch_size):
            end = min(start + batch_size, n_samples)
            bx = torch.tensor(inputs[start:end], dtype=torch.float32).to(device)
            by = torch.tensor(targets[start:end], dtype=torch.float32).to(device)
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

            # Context pooled per scale
            scale_contexts = []
            for enc_out in enc_out_list:
                pooled = enc_out.mean(dim=1)
                if model.channel_independence == 1 and N > 1:
                    pooled = pooled.reshape(B, N, -1).mean(dim=1)
                scale_contexts.append(pooled)

            stacked_f = torch.stack(dec_out_list, dim=-1)  # [B, pred_len, C, num_scales]
            cached.append((scale_contexts, stacked_f, by))

    return cached


def train_and_eval_adaptive(val_cached, test_cached, model, pred_len,
                           variant='adaptive', seed=2026, epochs=15, lr=2e-3):
    """
    Train and evaluate AdaptiveScaleMixer or an ablation variant.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    num_scales = val_cached[0][1].shape[-1]
    d_model = val_cached[0][0][0].shape[-1]

    # Initialize modules depending on variant
    if variant == 'baseline':
        # Pure uniform summation
        test_preds, test_targets = [], []
        weights_list = []
        with torch.no_grad():
            for scale_ctxs, stacked_f, by in test_cached:
                base_f = stacked_f.sum(dim=-1)
                base_f_denorm = model.normalize_layers[0](base_f, 'denorm')
                test_preds.append(base_f_denorm.cpu().numpy())
                test_targets.append(by.cpu().numpy())
                B = stacked_f.shape[0]
                weights_list.append(np.ones((B, num_scales)))
        p = np.concatenate(test_preds, axis=0)
        t = np.concatenate(test_targets, axis=0)
        return compute_all_metrics(p, t), np.concatenate(weights_list, axis=0), 0

    elif variant == 'fixed':
        scale_logits = nn.Parameter(torch.zeros(num_scales))
        params = [scale_logits]
        optimizer = torch.optim.Adam(params, lr=lr * 2, weight_decay=1e-4)

        for ep in range(epochs):
            for scale_ctxs, stacked_f, by in val_cached:
                optimizer.zero_grad()
                w = F.softmax(scale_logits, dim=-1) * num_scales
                f = (stacked_f * w.unsqueeze(0).unsqueeze(0).unsqueeze(0)).sum(dim=-1)
                f_denorm = model.normalize_layers[0](f, 'denorm')
                loss = F.mse_loss(f_denorm, by)
                loss.backward()
                optimizer.step()

        test_preds, test_targets = [], []
        weights_list = []
        with torch.no_grad():
            w = F.softmax(scale_logits, dim=-1) * num_scales
            for scale_ctxs, stacked_f, by in test_cached:
                f = (stacked_f * w.unsqueeze(0).unsqueeze(0).unsqueeze(0)).sum(dim=-1)
                f_denorm = model.normalize_layers[0](f, 'denorm')
                test_preds.append(f_denorm.cpu().numpy())
                test_targets.append(by.cpu().numpy())
                B = stacked_f.shape[0]
                weights_list.append(np.tile(w.cpu().numpy(), (B, 1)))

        p = np.concatenate(test_preds, axis=0)
        t = np.concatenate(test_targets, axis=0)
        return compute_all_metrics(p, t), np.concatenate(weights_list, axis=0), num_scales

    elif variant == 'no_context':
        horizon_embed = nn.Sequential(
            nn.Linear(1, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, d_model),
        )
        h_weight_net = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, num_scales),
        )
        nn.init.zeros_(h_weight_net[-1].weight)
        nn.init.zeros_(h_weight_net[-1].bias)
        params = list(horizon_embed.parameters()) + list(h_weight_net.parameters())
        optimizer = torch.optim.Adam(params, lr=lr, weight_decay=1e-4)

        h_val = torch.tensor([[float(pred_len)]])
        for ep in range(epochs):
            for scale_ctxs, stacked_f, by in val_cached:
                optimizer.zero_grad()
                he = horizon_embed(h_val)
                logits = h_weight_net(he)
                w = F.softmax(logits, dim=-1) * num_scales  # [1, num_scales]
                f = (stacked_f * w.unsqueeze(1).unsqueeze(2)).sum(dim=-1)
                f_denorm = model.normalize_layers[0](f, 'denorm')
                loss = F.mse_loss(f_denorm, by)
                loss.backward()
                optimizer.step()

        test_preds, test_targets = [], []
        weights_list = []
        with torch.no_grad():
            he = horizon_embed(h_val)
            logits = h_weight_net(he)
            w = F.softmax(logits, dim=-1) * num_scales
            for scale_ctxs, stacked_f, by in test_cached:
                f = (stacked_f * w.unsqueeze(1).unsqueeze(2)).sum(dim=-1)
                f_denorm = model.normalize_layers[0](f, 'denorm')
                test_preds.append(f_denorm.cpu().numpy())
                test_targets.append(by.cpu().numpy())
                B = stacked_f.shape[0]
                weights_list.append(np.tile(w.cpu().numpy(), (B, 1)))

        p = np.concatenate(test_preds, axis=0)
        t = np.concatenate(test_targets, axis=0)
        return compute_all_metrics(p, t), np.concatenate(weights_list, axis=0), sum(p.numel() for p in params)

    elif variant == 'no_horizon':
        scale_projs = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(num_scales)])
        combine = nn.Sequential(nn.Linear(d_model * num_scales, d_model), nn.GELU())
        weight_net = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, num_scales),
        )
        nn.init.zeros_(weight_net[-1].weight)
        nn.init.zeros_(weight_net[-1].bias)
        params = list(scale_projs.parameters()) + list(combine.parameters()) + list(weight_net.parameters())
        optimizer = torch.optim.Adam(params, lr=lr, weight_decay=1e-4)

        for ep in range(epochs):
            for scale_ctxs, stacked_f, by in val_cached:
                optimizer.zero_grad()
                proj = [scale_projs[j](ctx) for j, ctx in enumerate(scale_ctxs)]
                ctx = combine(torch.cat(proj, dim=-1))
                logits = weight_net(ctx)
                w = F.softmax(logits, dim=-1) * num_scales
                f = (stacked_f * w.unsqueeze(1).unsqueeze(2)).sum(dim=-1)
                f_denorm = model.normalize_layers[0](f, 'denorm')
                loss = F.mse_loss(f_denorm, by)
                loss.backward()
                optimizer.step()

        test_preds, test_targets = [], []
        weights_list = []
        with torch.no_grad():
            for scale_ctxs, stacked_f, by in test_cached:
                proj = [scale_projs[j](ctx) for j, ctx in enumerate(scale_ctxs)]
                ctx = combine(torch.cat(proj, dim=-1))
                logits = weight_net(ctx)
                w = F.softmax(logits, dim=-1) * num_scales
                f = (stacked_f * w.unsqueeze(1).unsqueeze(2)).sum(dim=-1)
                f_denorm = model.normalize_layers[0](f, 'denorm')
                test_preds.append(f_denorm.cpu().numpy())
                test_targets.append(by.cpu().numpy())
                weights_list.append(w.cpu().numpy())

        p = np.concatenate(test_preds, axis=0)
        t = np.concatenate(test_targets, axis=0)
        return compute_all_metrics(p, t), np.concatenate(weights_list, axis=0), sum(p.numel() for p in params)

    else:  # 'adaptive' - Full Model
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
        optimizer = torch.optim.Adam(params, lr=lr, weight_decay=1e-4)

        h_val = torch.tensor([[float(pred_len)]])
        for ep in range(epochs):
            for scale_ctxs, stacked_f, by in val_cached:
                optimizer.zero_grad()
                proj = [scale_projs[j](ctx) for j, ctx in enumerate(scale_ctxs)]
                ctx = combine(torch.cat(proj, dim=-1))
                h_emb = horizon_encoder(h_val).expand(ctx.size(0), -1)
                cond = torch.cat([ctx, h_emb], dim=-1)
                logits = weight_net(cond)
                w = F.softmax(logits, dim=-1) * num_scales
                f = (stacked_f * w.unsqueeze(1).unsqueeze(2)).sum(dim=-1)
                f_denorm = model.normalize_layers[0](f, 'denorm')
                loss = F.mse_loss(f_denorm, by)
                loss.backward()
                optimizer.step()

        test_preds, test_targets = [], []
        weights_list = []
        with torch.no_grad():
            for scale_ctxs, stacked_f, by in test_cached:
                proj = [scale_projs[j](ctx) for j, ctx in enumerate(scale_ctxs)]
                ctx = combine(torch.cat(proj, dim=-1))
                h_emb = horizon_encoder(h_val).expand(ctx.size(0), -1)
                cond = torch.cat([ctx, h_emb], dim=-1)
                logits = weight_net(cond)
                w = F.softmax(logits, dim=-1) * num_scales
                f = (stacked_f * w.unsqueeze(1).unsqueeze(2)).sum(dim=-1)
                f_denorm = model.normalize_layers[0](f, 'denorm')
                test_preds.append(f_denorm.cpu().numpy())
                test_targets.append(by.cpu().numpy())
                weights_list.append(w.cpu().numpy())

        p = np.concatenate(test_preds, axis=0)
        t = np.concatenate(test_targets, axis=0)
        return compute_all_metrics(p, t), np.concatenate(weights_list, axis=0), sum(p.numel() for p in params)


def run_full_suite():
    print("=" * 70)
    print("  COMPREHENSIVE EXPERIMENT SUITE: TimeMixer vs Adaptive-FMM")
    print("=" * 70)

    results_dir = os.path.join(PROJECT_ROOT, 'results')
    plots_dir = os.path.join(results_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)

    traffic_data = load_traffic_data()
    weather_data = load_weather_data()

    all_records = []
    weight_distributions = {}

    horizons = [96, 192, 336, 720]

    # 1. Main Traffic Benchmark across all horizons: Baseline vs Adaptive vs Ablations
    print("\n>>> 1. BENCHMARK ON TRAFFIC (Horizons: 96, 192, 336, 720)")
    for h in horizons:
        print(f"\n--- Processing Traffic Horizon {h} ---")
        cfg = build_config(h)
        ckpt = find_checkpoint(h)
        if not ckpt:
            print(f"Warning: No checkpoint for horizon {h}")
            continue

        model = load_model(ckpt, cfg, 'adaptive', 'cpu')

        val_in, val_tgt = create_test_windows(traffic_data['val'], traffic_data['full'],
                                             int(len(traffic_data['full']) * 0.7), 96, h, stride=24)
        test_in, test_tgt = create_test_windows(traffic_data['test'], traffic_data['full'],
                                               int(len(traffic_data['full']) * 0.8), 96, h, stride=24)

        # Cap for CPU runtime while keeping rich statistics
        val_in, val_tgt = val_in[:80], val_tgt[:80]
        test_in, test_tgt = test_in[:80], test_tgt[:80]

        print(f"  Extracting representations: {len(val_in)} val, {len(test_in)} test samples...")
        val_cached = extract_multiscale_representations(model, val_in, val_tgt)
        test_cached = extract_multiscale_representations(model, test_in, test_tgt)

        variants = ['baseline', 'adaptive', 'no_context', 'no_horizon', 'fixed']
        for var in variants:
            m, w, params = train_and_eval_adaptive(val_cached, test_cached, model, h, variant=var)
            rec = {
                'dataset': 'Traffic',
                'horizon': h,
                'model': var,
                'mse': m['mse'],
                'mae': m['mae'],
                'rmse': m['rmse'],
                'overhead_params': params,
            }
            all_records.append(rec)
            print(f"    [{var:12s}] MSE: {m['mse']:.5f} | MAE: {m['mae']:.5f} | Params: {params}")

            if var == 'adaptive':
                weight_distributions[f"Traffic_{h}"] = w

    # 2. Multi-Seed Stability Test on Traffic-96
    print("\n>>> 2. MULTI-SEED STABILITY VALIDATION (Traffic-96)")
    seeds = [2024, 2025, 2026]
    seed_records = []
    cfg96 = build_config(96)
    ckpt96 = find_checkpoint(96)
    model96 = load_model(ckpt96, cfg96, 'adaptive', 'cpu')
    val_in96, val_tgt96 = create_test_windows(traffic_data['val'], traffic_data['full'],
                                             int(len(traffic_data['full']) * 0.7), 96, 96, stride=24)
    test_in96, test_tgt96 = create_test_windows(traffic_data['test'], traffic_data['full'],
                                               int(len(traffic_data['full']) * 0.8), 96, 96, stride=24)
    val_cached96 = extract_multiscale_representations(model96, val_in96[:80], val_tgt96[:80])
    test_cached96 = extract_multiscale_representations(model96, test_in96[:80], test_tgt96[:80])

    for s in seeds:
        m, _, _ = train_and_eval_adaptive(val_cached96, test_cached96, model96, 96, variant='adaptive', seed=s)
        seed_records.append({'seed': s, 'mse': m['mse'], 'mae': m['mae']})
        print(f"    Seed {s}: MSE = {m['mse']:.5f}, MAE = {m['mae']:.5f}")

    # 3. Secondary Validation on Weather (Horizons: 96, 192)
    print("\n>>> 3. SECONDARY BENCHMARK ON WEATHER (Horizons: 96, 192)")
    for wh in [96, 192]:
        w_cfg = build_weather_config(wh)
        w_ckpt = os.path.join(PROJECT_ROOT, 'checkpoints',
                             f"long_term_forecast_Weather_96_{wh}_paper_none_TimeMixer_custom_sl96_pl{wh}_dm16_nh8_el2_dl1_df32_fc{'1' if wh==96 else '3'}_ebtimeF_dtTrue_Exp_0",
                             'checkpoint.pth')
        if os.path.exists(w_ckpt):
            w_model = load_model(w_ckpt, w_cfg, 'adaptive', 'cpu')
            w_val_in, w_val_tgt = create_test_windows(weather_data['val'], weather_data['full'],
                                                      int(len(weather_data['full']) * 0.7), 96, wh, stride=24)
            w_test_in, w_test_tgt = create_test_windows(weather_data['test'], weather_data['full'],
                                                        int(len(weather_data['full']) * 0.8), 96, wh, stride=24)
            w_val_cached = extract_multiscale_representations(w_model, w_val_in[:80], w_val_tgt[:80])
            w_test_cached = extract_multiscale_representations(w_model, w_test_in[:80], w_test_tgt[:80])

            for var in ['baseline', 'adaptive']:
                m, _, p = train_and_eval_adaptive(w_val_cached, w_test_cached, w_model, wh, variant=var)
                all_records.append({
                    'dataset': 'Weather',
                    'horizon': wh,
                    'model': var,
                    'mse': m['mse'],
                    'mae': m['mae'],
                    'rmse': m['rmse'],
                    'overhead_params': p,
                })
                print(f"    Weather-{wh} [{var:10s}] MSE: {m['mse']:.5f} | MAE: {m['mae']:.5f}")

    # Save DataFrame
    df_results = pd.DataFrame(all_records)
    csv_path = os.path.join(results_dir, 'all_experiments_results.csv')
    df_results.to_csv(csv_path, index=False)
    print(f"\nAll results saved to {csv_path}")

    # Generate Publication Plots
    print("\n>>> Generating Publication-Ready Figures in results/plots/...")

    # Plot 1: Baseline vs Adaptive MSE across Horizons (Traffic)
    t_df = df_results[df_results['dataset'] == 'Traffic']
    plt.figure(figsize=(8, 5), dpi=300)
    b_mse = t_df[t_df['model'] == 'baseline']['mse'].values
    a_mse = t_df[t_df['model'] == 'adaptive']['mse'].values
    x_h = np.arange(len(horizons))

    width = 0.35
    plt.bar(x_h - width/2, b_mse, width, label='Baseline TimeMixer (Uniform)', color='#94A3B8')
    plt.bar(x_h + width/2, a_mse, width, label='Adaptive-FMM (Proposed)', color='#2563EB')
    plt.xticks(x_h, [f"H={h}" for h in horizons], fontsize=12)
    plt.ylabel("Test MSE", fontsize=12)
    plt.title("Traffic Forecasting: Baseline vs. Adaptive-FMM", fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'traffic_baseline_vs_adaptive_mse.png'))
    plt.close()

    # Plot 2: Ablation Bar Comparison (Traffic-96 and Traffic-192)
    plt.figure(figsize=(9, 5), dpi=300)
    ab_96 = t_df[t_df['horizon'] == 96].set_index('model').loc[['baseline', 'fixed', 'no_horizon', 'no_context', 'adaptive']]
    models = ['Baseline', 'Fixed Weights', 'No Horizon', 'No Context', 'Adaptive (Full)']
    colors = ['#94A3B8', '#F59E0B', '#10B981', '#8B5CF6', '#2563EB']
    plt.bar(models, ab_96['mse'], color=colors, width=0.55)
    plt.ylabel("Test MSE (Lower is Better)", fontsize=12)
    plt.title("Ablation Study on Traffic (Horizon 96)", fontsize=14, fontweight='bold')
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    for i, v in enumerate(ab_96['mse']):
        plt.text(i, v + 0.00002, f"{v:.5f}", ha='center', fontsize=10, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'traffic_ablation_study.png'))
    plt.close()

    # Plot 3: Learned Scale Weight Attention Dynamics
    fig, axes = plt.subplots(1, 4, figsize=(16, 4), dpi=300, sharey=True)
    scale_names = ['Scale 0 (Fine)', 'Scale 1', 'Scale 2', 'Scale 3 (Coarse)']
    for idx, h in enumerate(horizons):
        key = f"Traffic_{h}"
        if key in weight_distributions:
            w_arr = weight_distributions[key]  # [N, 4]
            means = w_arr.mean(axis=0)
            stds = w_arr.std(axis=0)
            axes[idx].bar(scale_names, means, yerr=stds, capsize=5, color='#3B82F6', alpha=0.85)
            axes[idx].axhline(1.0, color='red', linestyle='--', label='Uniform Baseline (1.0)')
            axes[idx].set_title(f"Horizon {h}", fontsize=12, fontweight='bold')
            axes[idx].tick_params(axis='x', rotation=30)
            axes[idx].grid(axis='y', linestyle=':', alpha=0.6)
            if idx == 0:
                axes[idx].set_ylabel("Learned Scale Weight", fontsize=11)
                axes[idx].legend(loc='upper left', fontsize=9)
    plt.suptitle("Adaptive Scale Attention Across Forecasting Horizons", fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'learned_scale_weights_distribution.png'))
    plt.close()

    # Save summary report
    summary = {
        'timestamp': datetime.now().isoformat(),
        'traffic_results': all_records,
        'multi_seed_96': {
            'seeds': seed_records,
            'mean_mse': float(np.mean([r['mse'] for r in seed_records])),
            'std_mse': float(np.std([r['mse'] for r in seed_records])),
        }
    }
    with open(os.path.join(results_dir, 'comprehensive_results_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 70)
    print("  EXPERIMENT SUITE SUCCESSFULLY COMPLETED!")
    print("=" * 70)


if __name__ == '__main__':
    run_full_suite()

