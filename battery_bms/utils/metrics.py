"""
Evaluation metrics for battery state estimation.
"""

import numpy as np
from typing import Dict, Tuple


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Calculate evaluation metrics for predictions.
    
    Parameters
    ----------
    y_true : np.ndarray
        True values
    y_pred : np.ndarray
        Predicted values
        
    Returns
    -------
    dict
        Dictionary of metrics
    """
    # Mean Absolute Error
    mae = np.mean(np.abs(y_true - y_pred))
    
    # Root Mean Squared Error
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    
    # Mean Absolute Percentage Error
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-10))) * 100
    
    # R-squared
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - (ss_res / (ss_tot + 1e-10))
    
    # Max Error
    max_error = np.max(np.abs(y_true - y_pred))
    
    metrics = {
        'mae': mae,
        'rmse': rmse,
        'mape': mape,
        'r2': r2,
        'max_error': max_error
    }
    
    return metrics


def evaluate_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric_name: str = "SOC"
) -> None:
    """
    Evaluate and print prediction metrics.
    
    Parameters
    ----------
    y_true : np.ndarray
        True values
    y_pred : np.ndarray
        Predicted values
    metric_name : str, optional
        Name of the metric being evaluated (default: "SOC")
    """
    metrics = calculate_metrics(y_true, y_pred)
    
    print(f"\n{metric_name} Prediction Metrics")
    print("=" * 50)
    print(f"Mean Absolute Error (MAE):        {metrics['mae']:.4f}")
    print(f"Root Mean Squared Error (RMSE):   {metrics['rmse']:.4f}")
    print(f"Mean Absolute Percentage Error:   {metrics['mape']:.2f}%")
    print(f"R² Score:                         {metrics['r2']:.4f}")
    print(f"Maximum Error:                    {metrics['max_error']:.4f}")
    print("=" * 50)


def calculate_soc_error_statistics(
    soc_true: np.ndarray,
    soc_pred: np.ndarray
) -> Dict[str, float]:
    """
    Calculate SOC-specific error statistics.
    
    Parameters
    ----------
    soc_true : np.ndarray
        True SOC values (0 to 1)
    soc_pred : np.ndarray
        Predicted SOC values (0 to 1)
        
    Returns
    -------
    dict
        SOC error statistics
    """
    error = soc_pred - soc_true
    
    stats = {
        'mean_error': np.mean(error),
        'std_error': np.std(error),
        'mean_absolute_error': np.mean(np.abs(error)),
        'max_positive_error': np.max(error),
        'max_negative_error': np.min(error),
        'percentage_within_5': np.mean(np.abs(error) < 0.05) * 100,
        'percentage_within_10': np.mean(np.abs(error) < 0.10) * 100,
    }
    
    return stats


def calculate_confidence_interval(
    predictions: np.ndarray,
    covariance: np.ndarray,
    confidence: float = 0.95
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate confidence intervals for predictions.
    
    Parameters
    ----------
    predictions : np.ndarray
        Predicted values
    covariance : np.ndarray
        Prediction covariance matrix
    confidence : float, optional
        Confidence level (default: 0.95)
        
    Returns
    -------
    tuple
        (lower_bound, upper_bound)
    """
    from scipy import stats
    
    # Z-score for confidence level
    z = stats.norm.ppf((1 + confidence) / 2)
    
    # Standard deviation from covariance
    std = np.sqrt(np.diag(covariance))
    
    lower_bound = predictions - z * std
    upper_bound = predictions + z * std
    
    return lower_bound, upper_bound


def calculate_rul_accuracy(
    rul_true: np.ndarray,
    rul_pred: np.ndarray,
    tolerance_cycles: int = 50
) -> Dict[str, float]:
    """
    Calculate RUL prediction accuracy.
    
    Parameters
    ----------
    rul_true : np.ndarray
        True RUL values in cycles
    rul_pred : np.ndarray
        Predicted RUL values in cycles
    tolerance_cycles : int, optional
        Acceptable error in cycles (default: 50)
        
    Returns
    -------
    dict
        RUL accuracy metrics
    """
    error = rul_pred - rul_true
    
    accuracy = {
        'mae': np.mean(np.abs(error)),
        'rmse': np.sqrt(np.mean(error ** 2)),
        'percentage_within_tolerance': np.mean(np.abs(error) <= tolerance_cycles) * 100,
        'mean_percentage_error': np.mean(error / (rul_true + 1e-10)) * 100,
    }
    
    return accuracy


def battery_performance_score(
    soc_mae: float,
    soh_mae: float,
    voltage_rmse: float
) -> float:
    """
    Calculate overall battery estimation performance score.
    
    Parameters
    ----------
    soc_mae : float
        SOC mean absolute error
    soh_mae : float
        SOH mean absolute error
    voltage_rmse : float
        Voltage root mean squared error
        
    Returns
    -------
    float
        Performance score (0-100, higher is better)
    """
    # Normalize errors (lower is better)
    soc_score = max(0, 100 - soc_mae * 1000)  # Penalize SOC error heavily
    soh_score = max(0, 100 - soh_mae * 500)
    voltage_score = max(0, 100 - voltage_rmse * 100)
    
    # Weighted average
    overall_score = 0.5 * soc_score + 0.3 * soh_score + 0.2 * voltage_score
    
    return overall_score
