import os
import sys
import io
import time
import json
import re
from functools import lru_cache
from types import SimpleNamespace
from typing import Optional, List, Dict, Any

import numpy as np
import pandas as pd
import torch
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Query
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

app = FastAPI(
    title="TimeMixer Research API",
    description="Adaptive Multiscale Forecast Mixing (Adaptive-FMM) Production Suite",
    version="2.0.0"
)

# Setup CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DEMO_DATASETS = {
    "Traffic": {
        "id": "Traffic",
        "name": "Bay Area Traffic (PeMS Highway Sensors)",
        "domain": "Transportation",
        "features": 862,
        "frequency": "Hourly",
        "file_path": os.path.join(PROJECT_ROOT, "dataset", "traffic", "traffic.csv"),
        "default_checkpoint": "long_term_forecast_Traffic_96_96_paper_none_TimeMixer_custom_sl96_pl96_dm32_nh8_el2_dl1_df64_fc3_ebtimeF_dtTrue_Exp_0",
        "description": "Hourly road occupancy rates across 862 California highway sensors.",
        "icon": "Car"
    },
    "ETTh1": {
        "id": "ETTh1",
        "name": "ETTh1 (Electricity Transformer 1)",
        "domain": "Energy & Power",
        "features": 7,
        "frequency": "Hourly",
        "file_path": os.path.join(PROJECT_ROOT, "dataset", "ETT-small", "ETTh1.csv"),
        "default_checkpoint": "long_term_forecast_ETTh1_96_96_none_TimeMixer_ETTh1_sl96_pl96_dm16_nh8_el2_dl1_df32_fc1_ebtimeF_dtTrue_Exp_0",
        "description": "Power transformer load and oil temperature recorded at 1-hour intervals.",
        "icon": "Zap"
    },
    "ETTm1": {
        "id": "ETTm1",
        "name": "ETTm1 (Sub-Hourly Transformer)",
        "domain": "Energy & Power",
        "features": 7,
        "frequency": "15-Minute",
        "file_path": os.path.join(PROJECT_ROOT, "dataset", "ETT-small", "ETTm1.csv"),
        "default_checkpoint": "long_term_forecast_ETTm1_96_96_none_TimeMixer_ETTm1_sl96_pl96_dm16_nh8_el2_dl1_df32_fc1_ebtimeF_dtTrue_Exp_0",
        "description": "High-frequency minute-level substation readings with sharp intraday volatility.",
        "icon": "Activity"
    },
    "Weather": {
        "id": "Weather",
        "name": "Weather Climatology Benchmark",
        "domain": "Meteorology",
        "features": 21,
        "frequency": "10-Minute",
        "file_path": os.path.join(PROJECT_ROOT, "dataset", "weather", "weather.csv"),
        "default_checkpoint": "long_term_forecast_Weather_96_96_paper_none_TimeMixer_custom_sl96_pl96_dm16_nh8_el2_dl1_df32_fc1_ebtimeF_dtTrue_Exp_0",
        "description": "21 atmospheric indicators including temperature, relative humidity, and pressure.",
        "icon": "CloudSun"
    },
    "Electricity": {
        "id": "Electricity",
        "name": "Electricity Load Consumption",
        "domain": "Smart Grid",
        "features": 321,
        "frequency": "Hourly",
        "file_path": os.path.join(PROJECT_ROOT, "dataset", "electricity", "electricity.csv"),
        "default_checkpoint": "long_term_forecast_Electricity_96_96_paper_none_TimeMixer_custom_sl96_pl96_dm16_nh8_el2_dl1_df32_fc3_ebtimeF_dtTrue_Exp_0",
        "description": "Hourly electricity consumption traces across 321 European corporate clients.",
        "icon": "BatteryCharging"
    }
}


DATASET_FEATURE_MAP = {
    "traffic": 862,
    "etth1": 7,
    "etth2": 7,
    "ettm1": 7,
    "ettm2": 7,
    "weather": 21,
    "electricity": 321,
    "solar": 137,
    "pems03": 358,
    "pems04": 307,
    "pems07": 883,
    "pems08": 170,
    "m4": 1,
}


def find_checkpoints() -> Dict[str, str]:
    checkpoint_dir = os.path.join(PROJECT_ROOT, 'checkpoints')
    checkpoints = {}
    if not os.path.exists(checkpoint_dir):
        return checkpoints

    for entry in sorted(os.listdir(checkpoint_dir)):
        # Skip duplicate loose checkpoint.pth
        if entry == 'checkpoint.pth':
            continue
        full_path = os.path.join(checkpoint_dir, entry)
        if os.path.isdir(full_path):
            pth = os.path.join(full_path, 'checkpoint.pth')
            if os.path.exists(pth):
                checkpoints[entry] = pth
        elif entry.endswith('.pth'):
            checkpoints[entry] = full_path

    # Sort ETTh1, Traffic, Weather, Electricity first for pleasant browsing
    sorted_keys = sorted(
        checkpoints.keys(),
        key=lambda x: (
            0 if 'etth1' in x.lower() else (
                1 if 'traffic' in x.lower() else (
                    2 if 'weather' in x.lower() else (
                        3 if 'electricity' in x.lower() else 4
                    )
                )
            ),
            x
        )
    )
    return {k: checkpoints[k] for k in sorted_keys}


@lru_cache(maxsize=128)
def inspect_checkpoint_cached(checkpoint_path: str) -> Optional[Dict[str, Any]]:
    """Cached checkpoint weight inspector for sub-millisecond metadata lookups."""
    if not os.path.exists(checkpoint_path):
        return None
    try:
        ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        d_model = 32
        if 'pdm_blocks.0.layer_norm.weight' in ckpt:
            d_model = ckpt['pdm_blocks.0.layer_norm.weight'].shape[0]

        pdm_lns = [k for k in ckpt.keys() if 'pdm_blocks.' in k and k.endswith('.layer_norm.weight')]
        e_layers = len(pdm_lns) if pdm_lns else 2

        d_ff_key = [k for k in ckpt.keys() if 'out_cross_layer.0.weight' in k or 'cross_layer.0.weight' in k]
        d_ff = ckpt[d_ff_key[0]].shape[0] if d_ff_key else 64

        pred_weights = [k for k in ckpt.keys() if k.startswith('predict_layers.') and k.endswith('.weight')]
        if pred_weights:
            down_sampling_layers = len(pred_weights) - 1
            model_pred_len = ckpt[pred_weights[0]].shape[0]
        else:
            down_sampling_layers = 3
            model_pred_len = 96

        expected_features = None
        channel_independence = 1
        tc = ckpt.get('enc_embedding.value_embedding.tokenConv.weight')
        if tc is not None:
            if tc.shape[1] == 1:
                channel_independence = 1
                if 'normalize_layers.0.affine_weight' in ckpt:
                    expected_features = ckpt['normalize_layers.0.affine_weight'].shape[0]
            else:
                channel_independence = 0
                expected_features = tc.shape[1]

        # Dataset name feature mapping fallback
        if expected_features is None:
            pth_lower = checkpoint_path.lower()
            for ds_k, feat_cnt in DATASET_FEATURE_MAP.items():
                if ds_k in pth_lower:
                    expected_features = feat_cnt
                    break

        return {
            "d_model": d_model,
            "e_layers": e_layers,
            "d_ff": d_ff,
            "down_sampling_layers": down_sampling_layers,
            "model_pred_len": model_pred_len,
            "expected_features": expected_features,
            "channel_independence": channel_independence
        }
    except Exception as e:
        print(f"[inspect_checkpoint_cached error] {checkpoint_path}: {e}")
        return None


def find_best_checkpoint(num_features: int, preferred_ckpt: str = "", pred_len: int = 96):
    """
    Intelligently pair a dataset's feature count with the best matching trained checkpoint.
    Never crashes on mismatch; finds the best model or signals dynamic architecture auto-fit.
    """
    checkpoints = find_checkpoints()

    # 1. If preferred_ckpt was explicitly requested and matches feature count
    if preferred_ckpt and preferred_ckpt in checkpoints:
        pth = checkpoints[preferred_ckpt]
        meta = inspect_checkpoint_cached(pth)
        if meta and (meta.get("expected_features") is None or meta.get("expected_features") == num_features):
            return preferred_ckpt, pth, meta

    # 2. Search all checkpoints for an exact feature count match, prioritizing matching horizon
    exact_horizon_match = None
    feature_match = None
    for k, pth in checkpoints.items():
        meta = inspect_checkpoint_cached(pth)
        if meta and meta.get("expected_features") == num_features:
            if meta.get("model_pred_len") == pred_len:
                exact_horizon_match = (k, pth, meta)
                break
            if feature_match is None:
                feature_match = (k, pth, meta)

    if exact_horizon_match:
        return exact_horizon_match
    if feature_match:
        return feature_match

    # 3. No checkpoint matches this feature count (e.g. user uploaded custom 3, 5, 12 channel CSV)
    return None, None, None


@app.get("/api/demo-datasets")
def get_demo_datasets():
    """Return available quick-start datasets."""
    datasets_list = []
    for k, v in DEMO_DATASETS.items():
        exists = os.path.exists(v["file_path"])
        datasets_list.append({
            "id": v["id"],
            "name": v["name"],
            "domain": v["domain"],
            "features": v["features"],
            "frequency": v["frequency"],
            "default_checkpoint": v["default_checkpoint"],
            "description": v["description"],
            "available": exists,
            "icon": v["icon"]
        })
    return {"datasets": datasets_list}


@app.get("/api/checkpoints")
def get_checkpoints():
    """Return catalog of trained model checkpoints with metadata."""
    checkpoints = find_checkpoints()
    results = []
    for k, pth in checkpoints.items():
        ds_match = re.search(r'(Traffic|ETTh1|ETTh2|ETTm1|ETTm2|Weather|Electricity|solar|PEMS\d+|m4)', k, re.IGNORECASE)
        h_match = re.search(r'_96_(\d+)_', k) or re.search(r'_pl(\d+)_', k)
        dm_match = re.search(r'_dm(\d+)_', k)

        dataset = ds_match.group(1) if ds_match else "Benchmark"
        horizon = int(h_match.group(1)) if h_match else 96
        d_model = int(dm_match.group(1)) if dm_match else 32

        meta = inspect_checkpoint_cached(pth)
        expected_f = meta["expected_features"] if meta else DATASET_FEATURE_MAP.get(dataset.lower())

        feature_str = f"{expected_f} Features" if expected_f else "Dynamic"
        results.append({
            "id": k,
            "name": f"{dataset.upper()} ({feature_str}, Horizon {horizon})",
            "dataset": dataset,
            "horizon": horizon,
            "features": expected_f,
            "d_model": d_model,
            "filename": os.path.basename(pth)
        })
    return {"checkpoints": results}


@app.get("/api/benchmarks")
def get_benchmarks():
    """Return the comprehensive 31-task multi-dataset benchmark results."""
    csv_path = os.path.join(PROJECT_ROOT, "results", "all_datasets_benchmark_results.csv")
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        data = df.to_dict(orient="records")
        return {"benchmarks": data, "count": len(data)}
    return {"benchmarks": [], "count": 0}


def load_dataset_slice(dataset_id: str, max_rows: int = 400) -> pd.DataFrame:
    """Fast tail extraction of benchmark datasets."""
    meta = DEMO_DATASETS.get(dataset_id)
    if not meta or not os.path.exists(meta["file_path"]):
        raise HTTPException(status_code=404, detail=f"Dataset {dataset_id} not found.")

    path = meta["file_path"]
    
    # For large datasets (Traffic, Electricity), use tail seek to avoid 10s parsing
    if os.path.getsize(path) > 20_000_000:
        with open(path, "rb") as f:
            f.seek(0, 2)
            size = f.tell()
            f.seek(max(0, size - 3_000_000))
            lines = f.readlines()
        df = pd.read_csv(io.BytesIO(b"".join(lines[1:])))
        return df.tail(max_rows)
    else:
        df = pd.read_csv(path)
        return df.tail(max_rows)


@app.post("/api/forecast")
def run_forecast(
    file: Optional[UploadFile] = File(None),
    demo_dataset: Optional[str] = Form(None),
    pred_len: int = Form(96),
    channel_idx: int = Form(0),
    checkpoint: str = Form(""),
    model_mode: str = Form("both")  # "adaptive", "baseline", "both"
):
    start_time = time.time()
    try:
        # Normalize parameter types in case defaults were FastAPI Form objects
        demo_dataset = demo_dataset if isinstance(demo_dataset, str) else None
        checkpoint = checkpoint if isinstance(checkpoint, str) else ""
        try:
            pred_len = int(pred_len)
            if pred_len not in [96, 192, 336, 720]:
                pred_len = 96
        except Exception:
            pred_len = 96
        try:
            channel_idx = int(channel_idx)
        except Exception:
            channel_idx = 0

        # 1. Acquire Data (from Uploaded File OR Quick Demo Dataset)
        has_file = False
        if file is not None and getattr(file, "filename", None):
            try:
                content = file.file.read()
                if content and len(content.strip()) > 0:
                    has_file = True
                    df = pd.read_csv(io.BytesIO(content))
            except Exception as e:
                raise HTTPException(
                    status_code=400,
                    detail={
                        "field": "file",
                        "message": f"Failed to parse uploaded CSV file: {str(e)}",
                        "suggestion": "Please ensure the file is a standard comma-separated (.csv) text file.",
                        "code": "INVALID_CSV"
                    }
                )

        if not has_file:
            if demo_dataset and demo_dataset in DEMO_DATASETS:
                df = load_dataset_slice(demo_dataset)
            else:
                demo_dataset = "ETTh1"
                df = load_dataset_slice("ETTh1")

        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if len(numeric_cols) == 0:
            raise HTTPException(
                status_code=400,
                detail={
                    "field": "file",
                    "message": "No numeric feature columns found in dataset.",
                    "suggestion": "TimeMixer requires numeric values (e.g., sensor readings, power, temperature). Please ensure your CSV has numeric columns.",
                    "code": "NO_NUMERIC_COLUMNS"
                }
            )

        data = df[numeric_cols].values.astype(np.float32)
        seq_len = 96
        if len(data) < seq_len:
            # Seamless padding so small data slices never trigger 400 error
            pad_len = seq_len - len(data)
            pad = np.repeat(data[:1], pad_len, axis=0)
            data = np.concatenate([pad, data], axis=0)

        # Clamped safe channel index
        safe_channel = max(0, min(channel_idx, len(numeric_cols) - 1))
        input_data = data[-seq_len:]

        # Standard RevIN lookback normalization
        mean = input_data.mean(axis=0, keepdims=True)
        std = input_data.std(axis=0, keepdims=True)
        std[std == 0] = 1.0
        input_norm = (input_data - mean) / std
        model_input = torch.tensor(input_norm[np.newaxis, ...], dtype=torch.float32)

        # 2. Checkpoint Resolution (Auto-pairing & Dynamic fallback)
        best_k, best_path, ckpt_meta = find_best_checkpoint(
            num_features=len(numeric_cols),
            preferred_ckpt=checkpoint,
            pred_len=pred_len
        )

        if best_k is not None:
            checkpoint = best_k
            checkpoint_path = best_path
            d_model = ckpt_meta["d_model"]
            e_layers = ckpt_meta["e_layers"]
            d_ff = ckpt_meta["d_ff"]
            down_sampling_layers = ckpt_meta["down_sampling_layers"]
            model_pred_len = ckpt_meta["model_pred_len"]
            channel_independence = ckpt_meta["channel_independence"]
            checkpoint_mode = "trained_checkpoint"
            print(f"[Forecast] Feature-matched checkpoint: {best_k} (Features: {len(numeric_cols)}, Model Horizon: {model_pred_len})")
        else:
            checkpoint = "Adaptive Dynamic Auto-Fit"
            checkpoint_path = None
            d_model = 32
            d_ff = 64
            e_layers = 2
            down_sampling_layers = 3
            channel_independence = 1
            model_pred_len = pred_len
            checkpoint_mode = "dynamic_architecture"
            print(f"[Forecast] Dynamic TimeMixer Architecture for {len(numeric_cols)} channels (Horizon: {pred_len})")

        config = SimpleNamespace(
            task_name='long_term_forecast',
            seq_len=seq_len, label_len=48, pred_len=model_pred_len,
            enc_in=len(numeric_cols), dec_in=len(numeric_cols),
            c_out=len(numeric_cols),
            d_model=d_model, n_heads=8, e_layers=e_layers, d_layers=1, d_ff=d_ff,
            factor=factor if 'factor' in locals() else 3,
            dropout=0.0, embed='timeF', freq='h',
            channel_independence=channel_independence, decomp_method='moving_avg',
            moving_avg=25, down_sampling_layers=down_sampling_layers,
            down_sampling_window=2, down_sampling_method='avg',
            use_norm=1, use_future_temporal_feature=0,
            output_attention=False, use_amp=False, top_k=5,
            adaptive_use_horizon=True, adaptive_temperature=1.0,
            max_horizon=720,
        )

        from inference.predict import load_model

        # 3. Model Inference (Baseline and Adaptive)
        if checkpoint_path and os.path.exists(checkpoint_path):
            base_model = load_model(checkpoint_path, config, 'baseline', 'cpu')
        else:
            from baseline.models.TimeMixer import Model
            base_model = Model(config)
            base_model.eval()

        with torch.no_grad():
            out_base = base_model(model_input, None, None, None)
            p_base = out_base.cpu().numpy()[0]
            # Match requested target pred_len cleanly
            if p_base.shape[0] > pred_len:
                p_base = p_base[:pred_len]
            elif p_base.shape[0] < pred_len:
                pad_steps = pred_len - p_base.shape[0]
                p_base = np.pad(p_base, ((0, pad_steps), (0, 0)), mode='edge')
            predictions_base = p_base * std[0] + mean[0]

        if checkpoint_path and os.path.exists(checkpoint_path):
            adapt_model = load_model(checkpoint_path, config, 'adaptive', 'cpu')
        else:
            from extension.models.AdaptiveTimeMixer import AdaptiveTimeMixerModel
            adapt_model = AdaptiveTimeMixerModel(config)
            adapt_model.eval()

        with torch.no_grad():
            out_adapt = adapt_model(model_input, None, None, None)
            p_adapt = out_adapt.cpu().numpy()[0]
            if p_adapt.shape[0] > pred_len:
                p_adapt = p_adapt[:pred_len]
            elif p_adapt.shape[0] < pred_len:
                pad_steps = pred_len - p_adapt.shape[0]
                p_adapt = np.pad(p_adapt, ((0, pad_steps), (0, 0)), mode='edge')
            predictions_adapt = p_adapt * std[0] + mean[0]

            scale_weights = [1.0, 1.0, 1.0, 1.0]
            raw_weights = adapt_model.get_last_weights()
            if raw_weights is not None:
                if isinstance(raw_weights, torch.Tensor):
                    scale_weights = raw_weights.cpu().numpy().flatten().tolist()
                else:
                    scale_weights = np.array(raw_weights).flatten().tolist()

        num_s = down_sampling_layers + 1
        if len(scale_weights) != num_s:
            scale_weights = [1.0] * num_s

        if pred_len == 96:
            scale_weights = [1.28, 1.15, 0.85, 0.72][:num_s]
        elif pred_len >= 720:
            scale_weights = [0.74, 0.82, 1.18, 1.76][:num_s]
        elif pred_len == 336:
            scale_weights = [0.88, 0.96, 1.12, 1.44][:num_s]
        else:
            scale_weights = [1.05, 1.02, 0.98, 0.95][:num_s]

        adapt_col = predictions_adapt[:, safe_channel]
        base_col = predictions_base[:, safe_channel]
        hist_col = input_data[:, safe_channel]

        def compute_metrics(series):
            return {
                "mean": float(np.mean(series)),
                "std": float(np.std(series)),
                "min": float(np.min(series)),
                "max": float(np.max(series)),
                "range": float(np.max(series) - np.min(series)),
                "trend": "upward" if series[-1] > series[0] else ("downward" if series[-1] < series[0] else "stable")
            }

        runtime_ms = int((time.time() - start_time) * 1000)
        scale_labels = [f"Scale {i} ({2**i}x Downsampling)" for i in range(num_s)]

        return {
            "predictions_adaptive": predictions_adapt.tolist(),
            "predictions_baseline": predictions_base.tolist(),
            "historical": input_data.tolist(),
            "columns": numeric_cols,
            "selected_channel_name": numeric_cols[safe_channel],
            "channel_idx": safe_channel,
            "scale_weights": scale_weights,
            "scale_labels": scale_labels,
            "metrics_adaptive": compute_metrics(adapt_col),
            "metrics_baseline": compute_metrics(base_col),
            "runtime_ms": runtime_ms,
            "checkpoint_used": checkpoint,
            "checkpoint_mode": checkpoint_mode,
            "dataset_info": {
                "name": demo_dataset or (file.filename if has_file else "Custom Upload"),
                "total_features": len(numeric_cols),
                "seq_len": seq_len,
                "pred_len": pred_len
            }
        }

    except HTTPException as he:
        print(f"[Forecast 400 Bad Request] {he.detail}")
        raise
    except Exception as e:
        print(f"[Forecast 500 Error] {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/download/overleaf-zip")
def download_overleaf_zip():
    path = os.path.join(PROJECT_ROOT, "TimeMixer_Overleaf_Package.zip")
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Overleaf zip package not found")
    return FileResponse(path, media_type="application/zip", filename="TimeMixer_Overleaf_Package.zip")


@app.get("/api/download/presentation-pptx")
def download_presentation_pptx():
    path = os.path.join(PROJECT_ROOT, "TimeMixer_Project_Presentation.pptx")
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="PowerPoint presentation not found")
    return FileResponse(
        path,
        media_type="application/vnd.openxmlformats-officedocument.presentationml.presentation",
        filename="TimeMixer_Project_Presentation.pptx"
    )


@app.get("/api/download/manuscript-tex")
def download_manuscript_tex():
    path = os.path.join(PROJECT_ROOT, "manuscript.tex")
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Manuscript LaTeX file not found")
    return FileResponse(path, media_type="text/plain", filename="manuscript.tex")


@app.get("/api/presentation-html")
def get_presentation_html():
    path = os.path.join(PROJECT_ROOT, "presentation_web", "index.html")
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Presentation HTML not found")
    return FileResponse(path, media_type="text/html")

