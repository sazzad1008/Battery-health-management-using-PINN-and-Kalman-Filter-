"""
Basic usage example of Battery BMS.

This script demonstrates the basic workflow for battery health monitoring
using the Battery BMS package with PINN and Kalman filtering.
"""

import numpy as np
import sys
sys.path.append('..')

from battery_bms import BatteryBMS
from battery_bms.data import BatteryDataLoader


def main():
    """Main function demonstrating basic BMS usage."""
    
    print("="*60)
    print("Battery Health Management System - Basic Example")
    print("="*60)
    print()
    
    # Step 1: Generate synthetic battery data for demonstration
    print("Step 1: Generating synthetic battery data...")
    loader = BatteryDataLoader()
    data = loader.generate_synthetic_data(
        n_samples=1000,
        discharge_profile='dynamic'
    )
    print(f"  Generated {len(data['time'])} samples")
    print(f"  Features: {list(data.keys())}")
    print()
    
    # Step 2: Initialize Battery BMS
    print("Step 2: Initializing Battery BMS...")
    bms = BatteryBMS(
        use_pinn=True,
        use_kalman=True,
        battery_chemistry='LiFePO4'
    )
    print("  BMS initialized with:")
    print(f"    - PINN: {bms.use_pinn}")
    print(f"    - Kalman Filter: {bms.use_kalman}")
    print(f"    - Battery Chemistry: {bms.battery_chemistry}")
    print()
    
    # Step 3: Train the BMS
    print("Step 3: Training BMS models...")
    history = bms.train(data, epochs=50, physics_weight=0.5)
    print()
    
    # Step 4: Make predictions
    print("Step 4: Making predictions...")
    # Use a subset of data for prediction
    test_data = {
        'voltage': data['voltage'][-100:],
        'current': data['current'][-100:],
        'temperature': data['temperature'][-100:],
        'time': data['time'][-100:]
    }
    
    predictions = bms.predict(test_data)
    print()
    
    # Step 5: Display results
    print("Step 5: Prediction Results")
    print("-" * 60)
    print(f"State of Charge (SOC):       {predictions['soc']*100:.2f}%")
    print(f"State of Health (SOH):       {predictions['soh']*100:.2f}%")
    print(f"Remaining Useful Life (RUL): {predictions['rul']:.0f} cycles")
    print(f"Temperature:                 {predictions['temperature']:.1f}°C")
    print(f"Voltage:                     {predictions['voltage']:.3f}V")
    print(f"Internal Resistance:         {predictions['internal_resistance']*1000:.2f}mΩ")
    print("-" * 60)
    print()
    
    # Step 6: Evaluate performance
    print("Step 6: Evaluating model performance...")
    metrics = bms.evaluate(test_data)
    print("Performance Metrics:")
    print(f"  MAE:  {metrics['mae']:.4f}")
    print(f"  RMSE: {metrics['rmse']:.4f}")
    print(f"  MAPE: {metrics['mape']:.2f}%")
    print(f"  R²:   {metrics['r2']:.4f}")
    print()
    
    # Step 7: Visualize results (optional - requires matplotlib)
    try:
        print("Step 7: Visualizing results...")
        bms.plot_results(predictions)
    except Exception as e:
        print(f"  Visualization skipped: {e}")
    
    # Step 8: Save model
    print("\nStep 8: Saving trained model...")
    model_path = 'trained_bms_model.pkl'
    bms.save(model_path)
    print()
    
    print("="*60)
    print("Example completed successfully!")
    print("="*60)


if __name__ == '__main__':
    main()
