"""
Evaluation metrics for time series forecasting.
Computes MSE, MAE, RMSE, MAPE, sMAPE.
"""
import numpy as np


def mse(pred, true):
    """Mean Squared Error"""
    return np.mean((pred - true) ** 2)


def mae(pred, true):
    """Mean Absolute Error"""
    return np.mean(np.abs(pred - true))


def rmse(pred, true):
    """Root Mean Squared Error"""
    return np.sqrt(mse(pred, true))


def mape(pred, true, eps=1e-8):
    """Mean Absolute Percentage Error"""
    return np.mean(np.abs((true - pred) / (np.abs(true) + eps))) * 100


def smape(pred, true, eps=1e-8):
    """Symmetric Mean Absolute Percentage Error"""
    numerator = np.abs(pred - true)
    denominator = (np.abs(pred) + np.abs(true)) / 2 + eps
    return np.mean(numerator / denominator) * 100


def compute_all_metrics(pred, true):
    """
    Compute all forecasting metrics.

    Args:
        pred: np.ndarray of predictions [samples, pred_len, channels]
        true: np.ndarray of ground truth [samples, pred_len, channels]

    Returns:
        dict with keys: mse, mae, rmse, mape, smape
    """
    pred = np.array(pred)
    true = np.array(true)

    return {
        'mse': float(mse(pred, true)),
        'mae': float(mae(pred, true)),
        'rmse': float(rmse(pred, true)),
        'mape': float(mape(pred, true)),
        'smape': float(smape(pred, true)),
    }


def format_metrics(metrics, precision=4):
    """Format metrics dictionary as a readable string."""
    lines = []
    for key, value in metrics.items():
        lines.append(f"  {key.upper():>6s}: {value:.{precision}f}")
    return '\n'.join(lines)


if __name__ == '__main__':
    # Quick test
    np.random.seed(42)
    pred = np.random.randn(100, 96, 7)
    true = np.random.randn(100, 96, 7)
    m = compute_all_metrics(pred, true)
    print("Test metrics:")
    print(format_metrics(m))
