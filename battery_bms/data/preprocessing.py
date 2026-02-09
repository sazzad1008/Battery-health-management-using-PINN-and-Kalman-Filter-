"""
Data preprocessing utilities for battery datasets.
"""

import numpy as np
from typing import Dict, Tuple


def preprocess_battery_data(
    data: Dict,
    normalize: bool = True,
    remove_outliers: bool = True,
    fill_missing: bool = True
) -> Dict:
    """
    Preprocess battery data.
    
    Parameters
    ----------
    data : dict
        Raw battery data
    normalize : bool, optional
        Whether to normalize features (default: True)
    remove_outliers : bool, optional
        Whether to remove outliers (default: True)
    fill_missing : bool, optional
        Whether to fill missing values (default: True)
        
    Returns
    -------
    dict
        Preprocessed data
    """
    processed_data = data.copy()
    
    # Fill missing values
    if fill_missing:
        for key in processed_data:
            if processed_data[key] is not None and isinstance(processed_data[key], np.ndarray):
                if np.any(np.isnan(processed_data[key])):
                    # Forward fill
                    mask = np.isnan(processed_data[key])
                    idx = np.where(~mask, np.arange(len(mask)), 0)
                    np.maximum.accumulate(idx, out=idx)
                    processed_data[key][mask] = processed_data[key][idx[mask]]
    
    # Remove outliers
    if remove_outliers:
        for key in ['voltage', 'current', 'temperature']:
            if key in processed_data and processed_data[key] is not None:
                processed_data[key] = remove_outliers_iqr(processed_data[key])
    
    # Normalize features
    if normalize:
        processed_data['voltage_normalized'] = normalize_feature(
            processed_data['voltage'], min_val=2.0, max_val=4.2
        )
        processed_data['current_normalized'] = normalize_feature(
            processed_data['current'], min_val=-3.0, max_val=3.0
        )
        if 'temperature' in processed_data and processed_data['temperature'] is not None:
            processed_data['temperature_normalized'] = normalize_feature(
                processed_data['temperature'], min_val=-20.0, max_val=60.0
            )
    
    return processed_data


def normalize_feature(data: np.ndarray, min_val: float, max_val: float) -> np.ndarray:
    """
    Normalize feature to [0, 1] range.
    
    Parameters
    ----------
    data : np.ndarray
        Data to normalize
    min_val : float
        Minimum value for normalization
    max_val : float
        Maximum value for normalization
        
    Returns
    -------
    np.ndarray
        Normalized data
    """
    return (data - min_val) / (max_val - min_val)


def denormalize_feature(data: np.ndarray, min_val: float, max_val: float) -> np.ndarray:
    """
    Denormalize feature from [0, 1] range.
    
    Parameters
    ----------
    data : np.ndarray
        Normalized data
    min_val : float
        Minimum value used for normalization
    max_val : float
        Maximum value used for normalization
        
    Returns
    -------
    np.ndarray
        Denormalized data
    """
    return data * (max_val - min_val) + min_val


def remove_outliers_iqr(data: np.ndarray, factor: float = 1.5) -> np.ndarray:
    """
    Remove outliers using Interquartile Range (IQR) method.
    
    Parameters
    ----------
    data : np.ndarray
        Input data
    factor : float, optional
        IQR factor for outlier detection (default: 1.5)
        
    Returns
    -------
    np.ndarray
        Data with outliers clipped
    """
    q1 = np.percentile(data, 25)
    q3 = np.percentile(data, 75)
    iqr = q3 - q1
    
    lower_bound = q1 - factor * iqr
    upper_bound = q3 + factor * iqr
    
    return np.clip(data, lower_bound, upper_bound)


def create_sequences(
    data: Dict,
    sequence_length: int = 10,
    target_key: str = 'soc'
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create sequences for time-series prediction.
    
    Parameters
    ----------
    data : dict
        Battery data
    sequence_length : int, optional
        Length of input sequences (default: 10)
    target_key : str, optional
        Key for target variable (default: 'soc')
        
    Returns
    -------
    tuple
        (X, y) where X is input sequences and y is targets
    """
    features = []
    for key in ['voltage', 'current', 'temperature']:
        if key in data and data[key] is not None:
            features.append(data[key])
    
    X_full = np.column_stack(features)
    y_full = data[target_key]
    
    X_sequences = []
    y_sequences = []
    
    for i in range(len(X_full) - sequence_length):
        X_sequences.append(X_full[i:i + sequence_length])
        y_sequences.append(y_full[i + sequence_length])
    
    return np.array(X_sequences), np.array(y_sequences)


def split_train_test(
    data: Dict,
    train_ratio: float = 0.8
) -> Tuple[Dict, Dict]:
    """
    Split data into training and testing sets.
    
    Parameters
    ----------
    data : dict
        Battery data
    train_ratio : float, optional
        Ratio of training data (default: 0.8)
        
    Returns
    -------
    tuple
        (train_data, test_data)
    """
    n_samples = len(data['time'])
    split_idx = int(n_samples * train_ratio)
    
    train_data = {}
    test_data = {}
    
    for key, value in data.items():
        if value is not None and isinstance(value, np.ndarray):
            train_data[key] = value[:split_idx]
            test_data[key] = value[split_idx:]
    
    return train_data, test_data
