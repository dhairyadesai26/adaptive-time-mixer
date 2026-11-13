# TimeMixer Architecture Documentation

## 1. Original TimeMixer Architecture

```mermaid
graph TD
    A["Input: x ∈ ℝ^(B×T×C)"] --> B["Instance Normalization (RevIN)"]
    B --> C["Multiscale Downsampling"]
    C --> D1["Scale 1: T timesteps"]
    C --> D2["Scale 2: T/2 timesteps"]
    C --> D3["Scale 3: T/4 timesteps"]
    C --> D4["Scale 4: T/8 timesteps"]

    D1 --> E["Embedding (TokenConv + TimeFeature)"]
    D2 --> E
    D3 --> E
    D4 --> E

    E --> F["PDM Block ×e_layers"]

    subgraph PDM["Past Decomposable Mixing"]
        F1["Series Decomposition"] --> F2["Season: Bottom-up Mixing"]
        F1 --> F3["Trend: Top-down Mixing"]
        F2 --> F4["Combine + Residual"]
        F3 --> F4
    end

    F --> G["FMM: Future Multipredictor Mixing"]

    subgraph FMM["Future Multipredictor Mixing (Original)"]
        G1["Predictor 1: Linear(T → H)"]
        G2["Predictor 2: Linear(T/2 → H)"]
        G3["Predictor 3: Linear(T/4 → H)"]
        G4["Predictor 4: Linear(T/8 → H)"]
        G1 --> G5["Sum all predictions"]
        G2 --> G5
        G3 --> G5
        G4 --> G5
    end

    G --> H["Inverse Normalization"]
    H --> I["Output: ŷ ∈ ℝ^(B×H×C)"]
```

## 2. Our Extension: Adaptive Multiscale FMM

```mermaid
graph TD
    A["Encoder outputs at 4 scales"] --> B["Per-scale Predictors (same as baseline)"]

    B --> C1["Forecast 1"]
    B --> C2["Forecast 2"]
    B --> C3["Forecast 3"]
    B --> C4["Forecast 4"]

    A --> D["ScaleContextAggregator"]
    D --> E["Global Avg Pool + MLP"]

    F["Horizon h"] --> G["HorizonEncoder MLP"]

    E --> H["Concatenate"]
    G --> H

    H --> I["Weight Network (MLP)"]
    I --> J["Softmax → w₁, w₂, w₃, w₄"]

    C1 --> K["Weighted Sum: Σ wᵢ × forecastᵢ"]
    C2 --> K
    C3 --> K
    C4 --> K
    J --> K

    K --> L["Final Forecast"]
```

## 3. Key Equations

### Original FMM
```
forecast = Σᵢ₌₁ᴷ fᵢ(PDM_output_scaleᵢ)
```

### Adaptive FMM (Ours)
```
context = Agg(enc₁, enc₂, ..., encₖ)     # Scale context aggregation
horizon_emb = HorizonEnc(pred_len)         # Horizon conditioning
weights = softmax(WeightNet(context ⊕ horizon_emb))   # Scale weights
forecast = Σᵢ₌₁ᴷ wᵢ · fᵢ(PDM_output_scaleᵢ)         # Weighted combination
```

### Ablation Variants
| Variant | Weight Computation |
|---------|-------------------|
| Full Adaptive | `w = softmax(MLP(context ⊕ horizon_emb))` |
| No Context | `w = softmax(MLP(horizon_emb))` |
| No Horizon | `w = softmax(MLP(context))` |
| Fixed | `w = softmax(learnable_params)` |

## 4. Parameter Overhead

The adaptive mixer adds:
- `ScaleContextAggregator`: K × d_model² + d_model × (K × d_model)
- `HorizonEncoder`: d_model/2 + d_model²/2
- `WeightNet`: ~3 × d_model² + d_model × K

For Traffic (d_model=32, K=4): **~4,500 additional parameters** (~0.5% overhead)

## 5. Design Decisions

1. **Inherit, don't rewrite**: AdaptiveTimeMixer inherits from baseline Model class
2. **Initialize to uniform**: Weight network initialized to produce equal weights initially
3. **Temperature parameter**: Controls sharpness of weight distribution (softmax temperature)
4. **PDM untouched**: Only the prediction combination step is modified
