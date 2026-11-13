# Traffic Dataset — Post-Evaluation Report

**Generated:** 2026-10-05 11:24:51

**Model:** TimeMixer (Baseline)  
**Input Length:** 96  
**Features:** 862 traffic sensors  

## Results Summary

| Horizon | MSE | MAE | RMSE | Paper MSE | Paper MAE | Delta MSE | Delta MAE |
|---------|-----|-----|------|-----------|-----------|-------|-------|
| 96 | 0.4672 | 0.2916 | 0.6835 | 0.462 | 0.285 | [!] +0.0052 | [!] +0.0066 |
| 192 | 0.4775 | 0.2931 | 0.6910 | 0.474 | 0.292 | [!] +0.0035 | [!] +0.0011 |
| 336 | 0.4839 | 0.2997 | 0.6956 | 0.494 | 0.302 | [+] -0.0101 | [+] -0.0023 |
| 720 | 0.5217 | 0.3168 | 0.7223 | 0.529 | 0.322 | [+] -0.0073 | [+] -0.0052 |

## Detailed Metrics

### Horizon 96

- **MSE:** 0.467204
- **MAE:** 0.291591
- **RMSE:** 0.683523
- **MAPE:** 324.3361%
- **sMAPE:** 52.1013%
- **Test Samples:** 342
- **Inference Time:** 71.96s
- **Checkpoint:** `C:\AiProjectNew\TimeMixer_Project\checkpoints\long_term_forecast_Traffic_96_96_paper_none_TimeMixer_custom_sl96_pl96_dm32_nh8_el2_dl1_df64_fc3_ebtimeF_dtTrue_paper_0\checkpoint.pth`

### Horizon 192

- **MSE:** 0.477511
- **MAE:** 0.293128
- **RMSE:** 0.691022
- **MAPE:** 302.7594%
- **sMAPE:** 51.5899%
- **Test Samples:** 332
- **Inference Time:** 72.23s
- **Checkpoint:** `C:\AiProjectNew\TimeMixer_Project\checkpoints\long_term_forecast_Traffic_96_192_paper_none_TimeMixer_custom_sl96_pl192_dm32_nh8_el2_dl1_df64_fc3_ebtimeF_dtTrue_paper_0\checkpoint.pth`

### Horizon 336

- **MSE:** 0.483919
- **MAE:** 0.299711
- **RMSE:** 0.695643
- **MAPE:** 318.3978%
- **sMAPE:** 52.4761%
- **Test Samples:** 318
- **Inference Time:** 89.29s
- **Checkpoint:** `C:\AiProjectNew\TimeMixer_Project\checkpoints\Traffic_96_336_checkpoint.pth`

### Horizon 720

- **MSE:** 0.521743
- **MAE:** 0.316755
- **RMSE:** 0.722318
- **MAPE:** 330.6128%
- **sMAPE:** 55.0321%
- **Test Samples:** 279
- **Inference Time:** 115.37s
- **Checkpoint:** `C:\AiProjectNew\TimeMixer_Project\checkpoints\Traffic_96_720_checkpoint.pth`

## Analysis

### Error Statistics Per Horizon

| Horizon | Mean Sample MSE | Median Sample MSE | Std Sample MSE | Max Sample MSE |
|---------|----------------|-------------------|----------------|----------------|
| 96 | 0.467204 | 0.441856 | 0.143630 | 1.285645 |
| 192 | 0.477511 | 0.460390 | 0.111172 | 1.020927 |
| 336 | 0.483920 | 0.454426 | 0.099795 | 0.897813 |
| 720 | 0.521743 | 0.495020 | 0.090846 | 0.833175 |

### Per-Channel Error Summary

| Horizon | Best Channel MSE | Worst Channel MSE | Mean Channel MSE | Channel Std |
|---------|-----------------|-------------------|------------------|-------------|
| 96 | 0.009987 | 86.866005 | 0.467197 | 2.978277 |
| 192 | 0.011012 | 87.210571 | 0.477493 | 2.989971 |
| 336 | 0.010406 | 78.995552 | 0.483886 | 2.721961 |
| 720 | 0.011409 | 87.548958 | 0.521657 | 3.017925 |

## Plots

- `traffic_horizon_comparison.png` — MSE/MAE across horizons (ours vs paper)
- `traffic_h{H}_forecast_vs_actual.png` — Forecast overlays per horizon
- `traffic_error_distribution.png` — Per-sample error histograms
- `traffic_error_accumulation.png` — Error growth over prediction steps
- `traffic_per_channel_error.png` — Best/worst sensor channels
- `traffic_paper_comparison_bars.png` — Side-by-side bar comparison
