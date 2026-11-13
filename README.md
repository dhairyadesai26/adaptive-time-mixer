# TimeMixer: Adaptive Multiscale Forecast Mixing

> A complete design project: paper reproduction → improvement → experimental proof → prediction system → documentation

## Architecture

```
┌─────────────────────────┐
│    User / Demo          │
│  CSV → Forecast → Plot  │
└────────────┬────────────┘
             │
┌────────────▼────────────┐
│  Inference Pipeline     │
│  preprocessing + model  │
└────────────┬────────────┘
             │
┌────────────▼──────────────────┐
│  Our Improved TimeMixer       │
│  Adaptive Multiscale FMM      │
└────────────┬──────────────────┘
             │
┌────────────▼────────────────┐
│  Original TimeMixer         │
│  Baseline Reproduction      │
└────────────┬────────────────┘
             │
┌────────────▼─────────────────┐
│  Datasets + Training + Eval  │
│  Traffic / ETT / Weather / …│
└──────────────────────────────┘
```

## Project Structure

```
TimeMixer_Project/
├── baseline/               # Frozen upstream TimeMixer (DO NOT MODIFY)
│   ├── layers/             # Autoformer_EncDec, Embed, StandardNorm
│   └── models/             # TimeMixer.py (PDM + FMM)
│
├── extension/              # Our research contribution
│   ├── layers/             # AdaptiveScaleMixer, HorizonEncoder
│   └── models/             # AdaptiveTimeMixer + ablation variants
│
├── configs/                # YAML experiment configurations
│   ├── baseline/           # Traffic, Weather, etc.
│   └── extension/          # Adaptive FMM configs
│
├── checkpoints/            # Trained model weights (.pth)
├── dataset/                # ETT, Traffic, Weather, Solar, PEMS, M4, etc.
├── results/                # Metrics, tables, plots
├── evaluation/             # metrics.py, compare.py, plots.py, generate_tables.py
├── inference/              # preprocessing.py, predict.py
├── app/                    # Streamlit demo application
├── docs/                   # Report, architecture docs
├── environment/            # Environment records
│
├── setup.py                # Environment verification
├── evaluate.py             # Checkpoint evaluation
├── requirements.txt        # Python dependencies
└── README.md               # This file
```

## Quick Start

### 1. Verify Environment

```bash
python setup.py
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Evaluate a Checkpoint

```bash
python evaluate.py \
    --checkpoint checkpoints/long_term_forecast_Traffic_96_96_.../checkpoint.pth \
    --config configs/baseline/traffic_96_96.yaml
```

### 4. Run Inference

```bash
python inference/predict.py \
    --csv dataset/traffic/traffic.csv \
    --checkpoint checkpoints/.../checkpoint.pth \
    --model baseline \
    --horizon 96 \
    --output predictions.csv
```

### 5. Launch Demo App

```bash
streamlit run app/streamlit_app.py
```

### 6. Generate Results

```bash
python evaluation/generate_tables.py
python evaluation/plots.py
```

## Our Contribution: Adaptive Multiscale FMM

**Problem**: Original TimeMixer combines predictions from different scales uniformly.

**Our Solution**: Adaptive input-dependent scale weighting.

```
Original FMM:     forecast = Σ forecast_i           (uniform)
Our Adaptive FMM: forecast = Σ w_i × forecast_i     (learned weights)
```

Where `w_1, ..., w_k = softmax(weight_net(context, horizon))` depends on:
- **Input history** (what patterns are present)
- **Forecast horizon** (short vs long-term)

### Ablation Study

| Model | Description |
|-------|-------------|
| TimeMixer (Baseline) | Original uniform FMM |
| + Adaptive FMM | Full adaptive weighting |
| + No Context | Weights from horizon only |
| + No Horizon | Weights from context only |
| + Fixed Weights | Learned but input-independent |

## Datasets

| Dataset | Variables | Samples | Frequency |
|---------|-----------|---------|-----------|
| Traffic | 862 | 17,544 | Hourly |
| Weather | 21 | 52,696 | 10-min |
| ETTh1/h2 | 7 | 17,420 | Hourly |
| ETTm1/m2 | 7 | 69,680 | 15-min |
| Electricity | 321 | 26,304 | Hourly |
| Solar | 137 | 52,560 | 10-min |
| PEMS (03/04/07/08) | 170-883 | 26,208 | 5-min |
| M4 | 1 | Various | Various |

## Environment

- **Training**: Google Colab / Kaggle (Tesla T4 GPU)
- **Evaluation/Demo**: Local (CPU sufficient)
- **Python**: 3.13+ | **PyTorch**: 2.0+ | **CUDA**: 12.x

## References

- [TimeMixer Paper](https://arxiv.org/abs/2405.14616)
- [Official Repository](https://github.com/kwuking/TimeMixer)
- Upstream commit: `e24610583b36fdd8c76cc17a8df4e65759a5f460`
