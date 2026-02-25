"""
Data loading utilities for battery datasets.

This module provides functions to load and parse battery data from various formats.
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, Union
from pathlib import Path


class BatteryDataLoader:
    """
    Load and manage battery datasets.
    
    Supports multiple data formats:
    - CSV files with battery measurements
    - HDF5 files for large datasets
    - MAT files from battery research datasets (NASA, CALCE, etc.)
    
    Examples
    --------
    >>> loader = BatteryDataLoader()
    >>> data = loader.load('battery_data.csv')
    >>> data = loader.preprocess(data, normalize=True)
    """
    
    def __init__(self):
        self.data = None
        self.metadata = {}
    
    def load(self, filepath: Union[str, Path]) -> Dict:
        """
        Load battery data from file.
        
        Parameters
        ----------
        filepath : str or Path
            Path to data file
            
        Returns
        -------
        dict
            Loaded data with keys: 'voltage', 'current', 'temperature', 'time', etc.
        """
        filepath = Path(filepath)
        
        if not filepath.exists():
            raise FileNotFoundError(f"Data file not found: {filepath}")
        
        if filepath.suffix == '.csv':
            return self._load_csv(filepath)
        elif filepath.suffix in ['.h5', '.hdf5']:
            return self._load_hdf5(filepath)
        elif filepath.suffix == '.mat':
            return self._load_mat(filepath)
        else:
            raise ValueError(f"Unsupported file format: {filepath.suffix}")
    
    def _load_csv(self, filepath: Path) -> Dict:
        """Load data from CSV file."""
        df = pd.read_csv(filepath)
        
        # Expected columns: time, voltage, current, temperature, SOC, SOH
        data = {
            'time': df['time'].values if 'time' in df else np.arange(len(df)),
            'voltage': df['voltage'].values,
            'current': df['current'].values,
            'temperature': df['temperature'].values if 'temperature' in df else np.ones(len(df)) * 25.0,
            'soc': df['soc'].values if 'soc' in df else None,
            'soh': df['soh'].values if 'soh' in df else None,
        }
        
        self.metadata = {
            'n_samples': len(df),
            'columns': list(df.columns)
        }
        
        return data
    
    def _load_hdf5(self, filepath: Path) -> Dict:
        """Load data from HDF5 file."""
        try:
            import h5py
        except ImportError:
            raise ImportError("h5py is required to load HDF5 files. Install with: pip install h5py")
        
        data = {}
        with h5py.File(filepath, 'r') as f:
            for key in ['time', 'voltage', 'current', 'temperature', 'soc', 'soh']:
                if key in f:
                    data[key] = f[key][:]
        
        return data
    
    def _load_mat(self, filepath: Path) -> Dict:
        """Load data from MATLAB .mat file."""
        try:
            from scipy.io import loadmat
        except ImportError:
            raise ImportError("scipy is required to load .mat files.")
        
        mat_data = loadmat(filepath)
        
        # Extract relevant fields (structure depends on dataset)
        data = {}
        for key in ['time', 'voltage', 'current', 'temperature', 'soc', 'soh']:
            if key in mat_data:
                data[key] = mat_data[key].flatten()
        
        return data
    
    def preprocess(
        self,
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
        from .preprocessing import preprocess_battery_data
        
        return preprocess_battery_data(
            data,
            normalize=normalize,
            remove_outliers=remove_outliers,
            fill_missing=fill_missing
        )
    
    def generate_synthetic_data(
        self,
        n_samples: int = 1000,
        discharge_profile: str = 'constant_current'
    ) -> Dict:
        """
        Generate synthetic battery data for testing.
        
        Parameters
        ----------
        n_samples : int, optional
            Number of samples to generate (default: 1000)
        discharge_profile : str, optional
            Type of discharge profile: 'constant_current', 'dynamic', 'pulse'
            (default: 'constant_current')
            
        Returns
        -------
        dict
            Synthetic battery data
        """
        time = np.arange(n_samples)
        
        if discharge_profile == 'constant_current':
            current = np.ones(n_samples) * 1.0  # 1A constant discharge
        elif discharge_profile == 'dynamic':
            current = 1.0 + 0.5 * np.sin(2 * np.pi * time / 200)
        elif discharge_profile == 'pulse':
            current = np.where((time % 100) < 50, 2.0, 0.5)
        else:
            current = np.ones(n_samples)
        
        # Generate SOC (decreasing with time)
        soc = 1.0 - time / n_samples
        soc = np.clip(soc, 0.0, 1.0)
        
        # Generate voltage (function of SOC)
        voltage = 3.2 + 0.5 * soc - 0.3 * (soc - 0.5) ** 2 - current * 0.05
        
        # Add noise
        voltage += np.random.normal(0, 0.01, n_samples)
        
        # Generate temperature
        temperature = 25.0 + 10.0 * (1 - soc) + np.random.normal(0, 1, n_samples)
        
        data = {
            'time': time,
            'voltage': voltage,
            'current': current,
            'temperature': temperature,
            'soc': soc,
            'soh': np.ones(n_samples) * 0.95,  # Assume 95% health
        }
        
        return data


def load_battery_data(filepath: str) -> Dict:
    """
    Convenience function to load battery data.
    
    Parameters
    ----------
    filepath : str
        Path to data file
        
    Returns
    -------
    dict
        Loaded battery data
    """
    loader = BatteryDataLoader()
    return loader.load(filepath)
