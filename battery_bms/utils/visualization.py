"""
Visualization utilities for battery data and predictions.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Optional, List


def plot_battery_states(predictions: Dict, battery_params: Optional[Dict] = None) -> None:
    """
    Plot battery state predictions.
    
    Parameters
    ----------
    predictions : dict
        Prediction results with keys like 'soc', 'soh', 'voltage', etc.
    battery_params : dict, optional
        Battery parameters for context
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle('Battery State Predictions', fontsize=16, fontweight='bold')
    
    # SOC
    axes[0, 0].bar(['State of Charge'], [predictions.get('soc', 0) * 100], color='green', alpha=0.7)
    axes[0, 0].set_ylim([0, 100])
    axes[0, 0].set_ylabel('SOC (%)')
    axes[0, 0].set_title('State of Charge')
    axes[0, 0].axhline(y=20, color='r', linestyle='--', label='Low SOC Warning')
    axes[0, 0].legend()
    
    # SOH
    axes[0, 1].bar(['State of Health'], [predictions.get('soh', 0) * 100], color='blue', alpha=0.7)
    axes[0, 1].set_ylim([0, 100])
    axes[0, 1].set_ylabel('SOH (%)')
    axes[0, 1].set_title('State of Health')
    axes[0, 1].axhline(y=80, color='orange', linestyle='--', label='Replace Threshold')
    axes[0, 1].legend()
    
    # RUL
    rul = predictions.get('rul', 0)
    axes[1, 0].bar(['Remaining Useful Life'], [rul], color='purple', alpha=0.7)
    axes[1, 0].set_ylabel('Cycles')
    axes[1, 0].set_title('Remaining Useful Life')
    axes[1, 0].text(0, rul + 20, f'{rul:.0f} cycles', ha='center', fontsize=12, fontweight='bold')
    
    # Temperature and Voltage
    temp = predictions.get('temperature', 25)
    voltage = predictions.get('voltage', 3.2)
    
    metrics = ['Temperature\n(°C)', 'Voltage\n(V)', 'Resistance\n(mΩ)']
    values = [temp, voltage, predictions.get('internal_resistance', 0.05) * 1000]
    colors = ['red', 'orange', 'brown']
    
    bars = axes[1, 1].bar(metrics, values, color=colors, alpha=0.7)
    axes[1, 1].set_title('Operating Parameters')
    
    # Add value labels on bars
    for bar, value in zip(bars, values):
        height = bar.get_height()
        axes[1, 1].text(bar.get_x() + bar.get_width()/2., height,
                       f'{value:.2f}', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.show()
    
    print("\n" + "="*50)
    print("Battery State Summary")
    print("="*50)
    print(f"State of Charge (SOC):       {predictions.get('soc', 0)*100:.1f}%")
    print(f"State of Health (SOH):       {predictions.get('soh', 0)*100:.1f}%")
    print(f"Remaining Useful Life:       {predictions.get('rul', 0):.0f} cycles")
    print(f"Temperature:                 {temp:.1f}°C")
    print(f"Voltage:                     {voltage:.2f}V")
    print(f"Internal Resistance:         {predictions.get('internal_resistance', 0.05)*1000:.1f}mΩ")
    print("="*50)


def plot_training_history(history: Dict, title: str = "Training History") -> None:
    """
    Plot training history.
    
    Parameters
    ----------
    history : dict
        Training history with loss values
    title : str, optional
        Plot title
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    epochs = range(1, len(history.get('loss', [])) + 1)
    
    if 'loss' in history:
        ax.plot(epochs, history['loss'], 'b-', label='Total Loss', linewidth=2)
    
    if 'data_loss' in history:
        ax.plot(epochs, history['data_loss'], 'g--', label='Data Loss', linewidth=1.5)
    
    if 'physics_loss' in history:
        ax.plot(epochs, history['physics_loss'], 'r--', label='Physics Loss', linewidth=1.5)
    
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')
    
    plt.tight_layout()
    plt.show()


def plot_time_series(
    data: Dict,
    keys: List[str] = ['voltage', 'current', 'temperature', 'soc'],
    title: str = "Battery Time Series Data"
) -> None:
    """
    Plot time series battery data.
    
    Parameters
    ----------
    data : dict
        Battery data dictionary
    keys : list of str, optional
        Keys to plot
    title : str, optional
        Plot title
    """
    n_plots = len(keys)
    fig, axes = plt.subplots(n_plots, 1, figsize=(12, 3*n_plots), sharex=True)
    
    if n_plots == 1:
        axes = [axes]
    
    time = data.get('time', np.arange(len(data[keys[0]])))
    
    labels = {
        'voltage': 'Voltage (V)',
        'current': 'Current (A)',
        'temperature': 'Temperature (°C)',
        'soc': 'SOC (%)',
        'soh': 'SOH (%)'
    }
    
    for ax, key in zip(axes, keys):
        if key in data and data[key] is not None:
            values = data[key]
            if key in ['soc', 'soh']:
                values = values * 100  # Convert to percentage
            
            ax.plot(time, values, linewidth=1.5)
            ax.set_ylabel(labels.get(key, key), fontsize=11)
            ax.grid(True, alpha=0.3)
    
    axes[-1].set_xlabel('Time (s)', fontsize=11)
    axes[0].set_title(title, fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.show()


def plot_soc_voltage_curve(data: Dict) -> None:
    """
    Plot SOC vs Voltage curve.
    
    Parameters
    ----------
    data : dict
        Battery data with 'soc' and 'voltage' keys
    """
    if 'soc' not in data or 'voltage' not in data:
        print("SOC and voltage data required for this plot")
        return
    
    plt.figure(figsize=(10, 6))
    plt.scatter(data['soc'] * 100, data['voltage'], alpha=0.5, s=10)
    plt.xlabel('State of Charge (%)', fontsize=12)
    plt.ylabel('Voltage (V)', fontsize=12)
    plt.title('SOC vs Voltage Curve', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_comparison(
    true_values: np.ndarray,
    predictions: np.ndarray,
    title: str = "Prediction vs True Values",
    ylabel: str = "Value"
) -> None:
    """
    Plot comparison between true and predicted values.
    
    Parameters
    ----------
    true_values : np.ndarray
        True values
    predictions : np.ndarray
        Predicted values
    title : str, optional
        Plot title
    ylabel : str, optional
        Y-axis label
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Time series comparison
    time = np.arange(len(true_values))
    ax1.plot(time, true_values, 'b-', label='True', linewidth=2, alpha=0.7)
    ax1.plot(time, predictions, 'r--', label='Predicted', linewidth=2, alpha=0.7)
    ax1.set_xlabel('Sample', fontsize=11)
    ax1.set_ylabel(ylabel, fontsize=11)
    ax1.set_title('Time Series Comparison', fontsize=12, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Scatter plot
    ax2.scatter(true_values, predictions, alpha=0.5, s=20)
    
    # Perfect prediction line
    min_val = min(true_values.min(), predictions.min())
    max_val = max(true_values.max(), predictions.max())
    ax2.plot([min_val, max_val], [min_val, max_val], 'k--', label='Perfect Prediction')
    
    ax2.set_xlabel('True Values', fontsize=11)
    ax2.set_ylabel('Predicted Values', fontsize=11)
    ax2.set_title('Prediction Accuracy', fontsize=12, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    fig.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()
