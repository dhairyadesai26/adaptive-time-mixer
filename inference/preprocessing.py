"""
Preprocessing pipeline for inference.
Handles CSV validation, normalization, and input window creation.

Usage:
    from inference.preprocessing import validate_csv, prepare_input

    df = validate_csv("path/to/data.csv")
    model_input, scaler_info = prepare_input(df, seq_len=96)
"""
import numpy as np
import pandas as pd
from typing import Tuple, Optional, Dict, Any


def validate_csv(csv_path: str) -> pd.DataFrame:
    """
    Validate and load a CSV file for forecasting.

    Checks:
        - File exists and is readable
        - Has a parseable date/timestamp column
        - Has at least one numeric column
        - No entirely null columns

    Args:
        csv_path: path to CSV file

    Returns:
        DataFrame with parsed dates and sorted by time

    Raises:
        ValueError: if validation fails
    """
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        raise ValueError(f"Cannot read CSV file: {e}")

    if len(df) == 0:
        raise ValueError("CSV file is empty")

    # Find date column
    date_col = None
    for col in df.columns:
        if col.lower() in ['date', 'timestamp', 'time', 'datetime', 'ds']:
            date_col = col
            break

    if date_col is None:
        # Try first column
        try:
            pd.to_datetime(df.iloc[:, 0])
            date_col = df.columns[0]
        except (ValueError, TypeError):
            date_col = None

    if date_col:
        df[date_col] = pd.to_datetime(df[date_col])
        df = df.sort_values(date_col).reset_index(drop=True)

    # Check numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if len(numeric_cols) == 0:
        raise ValueError("No numeric columns found in CSV")

    # Drop entirely null columns
    null_cols = [c for c in numeric_cols if df[c].isna().all()]
    if null_cols:
        df = df.drop(columns=null_cols)
        numeric_cols = [c for c in numeric_cols if c not in null_cols]

    info = {
        'date_column': date_col,
        'numeric_columns': numeric_cols,
        'n_rows': len(df),
        'n_features': len(numeric_cols),
    }

    print(f"  CSV validated: {info['n_rows']} rows, {info['n_features']} features")
    return df


def handle_missing_values(df: pd.DataFrame, method: str = 'interpolate') -> pd.DataFrame:
    """
    Handle missing values in numeric columns.

    Args:
        df: input DataFrame
        method: 'interpolate', 'ffill', 'mean', or 'drop'

    Returns:
        DataFrame with missing values handled
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns

    if method == 'interpolate':
        df[numeric_cols] = df[numeric_cols].interpolate(method='linear', limit_direction='both')
    elif method == 'ffill':
        df[numeric_cols] = df[numeric_cols].ffill().bfill()
    elif method == 'mean':
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
    elif method == 'drop':
        df = df.dropna(subset=numeric_cols).reset_index(drop=True)

    return df


def normalize(data: np.ndarray) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """
    Standard normalization (zero mean, unit variance).

    Args:
        data: np.ndarray [time, features]

    Returns:
        normalized_data: np.ndarray [time, features]
        scaler_info: dict with 'mean' and 'std' arrays
    """
    mean = data.mean(axis=0)
    std = data.std(axis=0)
    std[std == 0] = 1.0  # Avoid division by zero

    normalized = (data - mean) / std
    scaler_info = {'mean': mean, 'std': std}
    return normalized, scaler_info


def inverse_transform(data: np.ndarray, scaler_info: Dict[str, np.ndarray]) -> np.ndarray:
    """
    Reverse normalization to get predictions in original scale.

    Args:
        data: normalized predictions [samples, pred_len, features] or [pred_len, features]
        scaler_info: dict with 'mean' and 'std' arrays

    Returns:
        denormalized data
    """
    return data * scaler_info['std'] + scaler_info['mean']


def prepare_input(df: pd.DataFrame, seq_len: int = 96,
                  target_cols: Optional[list] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Prepare model input from DataFrame.

    Takes the last `seq_len` timesteps and normalizes them.

    Args:
        df: validated DataFrame
        seq_len: input sequence length
        target_cols: specific columns to use (None = all numeric)

    Returns:
        model_input: np.ndarray [1, seq_len, features] ready for model
        info: dict with scaler_info, column_names, etc.
    """
    # Handle missing values
    df = handle_missing_values(df)

    # Select columns
    if target_cols:
        cols = target_cols
    else:
        cols = df.select_dtypes(include=[np.number]).columns.tolist()

    data = df[cols].values.astype(np.float32)

    if len(data) < seq_len:
        raise ValueError(f"Not enough data: have {len(data)} rows, need {seq_len}")

    # Use the last seq_len points as input
    input_data = data[-seq_len:]

    # Normalize using the full data statistics (more stable)
    normalized, scaler_info = normalize(data)
    input_normalized = normalized[-seq_len:]

    # Add batch dimension: [1, seq_len, features]
    model_input = input_normalized[np.newaxis, ...]

    info = {
        'scaler_info': scaler_info,
        'column_names': cols,
        'n_features': len(cols),
        'seq_len': seq_len,
        'raw_input': input_data,
    }

    return model_input, info


def create_time_features(dates: pd.Series, freq: str = 'h') -> np.ndarray:
    """
    Create time features from datetime series.
    Matches the TimeMixer timeF embedding format.

    Args:
        dates: pandas Series of datetime values
        freq: frequency ('h' for hourly, 'd' for daily, etc.)

    Returns:
        features: np.ndarray [time, n_features]
    """
    features = []

    if freq in ['h', 't', 's', 'ms']:
        features.append(dates.dt.hour / 23.0 - 0.5)
    if freq in ['h', 't', 's', 'ms', 'd', 'b']:
        features.append(dates.dt.dayofweek / 6.0 - 0.5)
    if freq in ['h', 't', 's', 'ms', 'd', 'b', 'w']:
        features.append(dates.dt.day / 30.0 - 0.5)
    features.append(dates.dt.month / 11.0 - 0.5)

    if freq == 't':
        features.append(dates.dt.minute / 59.0 - 0.5)

    return np.stack(features, axis=-1).astype(np.float32)
