"""Unit tests for Battery BMS models."""

import pytest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from battery_bms.models.pinn import PINN
from battery_bms.models.battery_model import BatteryEquivalentCircuit


class TestPINN:
    """Test suite for Physics-Informed Neural Network."""
    
    def test_initialization(self):
        """Test PINN initialization."""
        pinn = PINN(input_dim=4, hidden_layers=[64, 128, 64], physics_weight=0.5)
        
        assert pinn.input_dim == 4
        assert pinn.hidden_layers == [64, 128, 64]
        assert pinn.physics_weight == 0.5
        assert len(pinn.weights) > 0
        assert len(pinn.biases) > 0
    
    def test_forward_pass(self):
        """Test forward pass through the network."""
        pinn = PINN(input_dim=4, hidden_layers=[32, 32], physics_weight=0.5)
        
        # Create dummy input
        x = np.random.randn(10, 4)
        
        # Forward pass
        output = pinn.forward(x)
        
        assert output.shape == (10, 1)
        assert not np.any(np.isnan(output))
    
    def test_physics_loss(self):
        """Test physics loss computation."""
        pinn = PINN(input_dim=4, hidden_layers=[32], physics_weight=0.5)
        
        # Create dummy data
        x = np.random.randn(100, 4)
        x[:, 1] = 1.0  # Current
        x[:, 3] = 1.0  # Time step
        
        predictions = np.random.rand(100, 1) * 0.5 + 0.5  # SOC between 0.5-1.0
        
        physics_loss = pinn.compute_physics_loss(x, predictions)
        
        assert physics_loss >= 0
        assert not np.isnan(physics_loss)


class TestBatteryEquivalentCircuit:
    """Test suite for Battery Equivalent Circuit Model."""
    
    def test_initialization(self):
        """Test ECM initialization."""
        ecm = BatteryEquivalentCircuit(capacity=2.3)
        
        assert ecm.capacity == 2.3
        assert ecm.soc == 1.0
        assert ecm.v1 == 0.0
    
    def test_ocv_calculation(self):
        """Test OCV calculation."""
        ecm = BatteryEquivalentCircuit(capacity=2.3)
        
        # Test at different SOC values
        ocv_full = ecm.ocv(1.0)
        ocv_half = ecm.ocv(0.5)
        ocv_empty = ecm.ocv(0.0)
        
        assert ocv_full > ocv_half > ocv_empty
        assert 2.0 <= ocv_empty <= 4.2
        assert 2.0 <= ocv_full <= 4.2
    
    def test_voltage_simulation(self):
        """Test voltage simulation."""
        ecm = BatteryEquivalentCircuit(capacity=2.3)
        
        voltage = ecm.simulate_voltage(
            current=1.0,
            soc=0.8,
            temperature=25.0,
            dt=1.0
        )
        
        assert 2.0 <= voltage <= 4.2
        assert not np.isnan(voltage)
    
    def test_soc_update(self):
        """Test SOC update via Coulomb counting."""
        ecm = BatteryEquivalentCircuit(capacity=2.3)
        
        # Discharge for 10 seconds at 1A
        for _ in range(10):
            soc = ecm.update_soc(current=1.0, dt=1.0)
        
        assert soc < 1.0
        assert soc >= 0.0
    
    def test_temperature_dependence(self):
        """Test temperature effects on resistance."""
        ecm = BatteryEquivalentCircuit(capacity=2.3)
        
        r0_25, r1_25 = ecm.update_temperature_dependence(25.0)
        r0_0, r1_0 = ecm.update_temperature_dependence(0.0)
        
        # Resistance should be higher at lower temperature
        assert r0_0 > r0_25
        assert r1_0 > r1_25


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
