"""Unit tests for Kalman filters."""

import pytest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from battery_bms.filters.kalman import KalmanFilter, ExtendedKalmanFilter


class TestKalmanFilter:
    """Test suite for standard Kalman Filter."""
    
    def test_initialization(self):
        """Test KF initialization."""
        A = np.eye(2)
        B = np.array([[0], [1]])
        H = np.array([[1, 0]])
        Q = np.eye(2) * 0.01
        R = np.array([[0.1]])
        initial_state = np.array([1.0, 0.0])
        
        kf = KalmanFilter(A, B, H, Q, R, initial_state)
        
        assert kf.x.shape == (2,)
        assert kf.P.shape == (2, 2)
    
    def test_prediction(self):
        """Test prediction step."""
        A = np.eye(2)
        B = np.array([[0], [1]])
        H = np.array([[1, 0]])
        Q = np.eye(2) * 0.01
        R = np.array([[0.1]])
        initial_state = np.array([1.0, 0.0])
        
        kf = KalmanFilter(A, B, H, Q, R, initial_state)
        
        # Predict
        x_pred = kf.predict(u=np.array([0.0]))
        
        assert x_pred.shape == (2,)
        assert not np.any(np.isnan(x_pred))
    
    def test_update(self):
        """Test update step."""
        A = np.eye(2)
        B = np.array([[0], [1]])
        H = np.array([[1, 0]])
        Q = np.eye(2) * 0.01
        R = np.array([[0.1]])
        initial_state = np.array([1.0, 0.0])
        
        kf = KalmanFilter(A, B, H, Q, R, initial_state)
        
        # Update with measurement
        z = np.array([0.95])
        x_updated = kf.update(z, u=np.array([0.0]))
        
        assert x_updated.shape == (2,)
        assert not np.any(np.isnan(x_updated))


class TestExtendedKalmanFilter:
    """Test suite for Extended Kalman Filter."""
    
    def test_initialization(self):
        """Test EKF initialization."""
        ekf = ExtendedKalmanFilter(
            initial_state={'soc': 1.0, 'soh': 0.95},
            process_noise=0.01,
            measurement_noise=0.05
        )
        
        assert ekf.x.shape == (3,)  # [SOC, SOH, R0]
        assert ekf.P.shape == (3, 3)
        assert ekf.x[0] == 1.0  # SOC
        assert ekf.x[1] == 0.95  # SOH
    
    def test_state_transition(self):
        """Test state transition function."""
        ekf = ExtendedKalmanFilter(
            initial_state={'soc': 1.0, 'soh': 0.95},
            process_noise=0.01,
            measurement_noise=0.05
        )
        
        x = np.array([0.8, 0.95, 0.05])
        u = np.array([1.0, 25.0])  # Current, temperature
        
        x_next = ekf.state_transition(x, u)
        
        assert x_next.shape == (3,)
        assert 0.0 <= x_next[0] <= 1.0  # SOC bounds
        assert 0.5 <= x_next[1] <= 1.0  # SOH bounds
        assert x_next[2] > 0  # Positive resistance
    
    def test_measurement_function(self):
        """Test measurement function."""
        ekf = ExtendedKalmanFilter(
            initial_state={'soc': 1.0, 'soh': 0.95},
            process_noise=0.01,
            measurement_noise=0.05
        )
        
        x = np.array([0.8, 0.95, 0.05])
        u = np.array([1.0, 25.0])
        
        voltage = ekf.measurement_function(x, u)
        
        assert voltage.shape == (1,)
        assert 2.0 <= voltage[0] <= 4.5
    
    def test_update(self):
        """Test EKF update with measurement."""
        ekf = ExtendedKalmanFilter(
            initial_state={'soc': 1.0, 'soh': 0.95},
            process_noise=0.01,
            measurement_noise=0.05
        )
        
        measurement = {
            'voltage': 3.3,
            'current': 1.0,
            'temperature': 25.0
        }
        
        state = ekf.update(measurement)
        
        assert 'soc' in state
        assert 'soh' in state
        assert 'resistance' in state
        assert 0.0 <= state['soc'] <= 1.0
        assert 0.5 <= state['soh'] <= 1.0
        assert state['resistance'] > 0
    
    def test_get_state(self):
        """Test state retrieval."""
        ekf = ExtendedKalmanFilter(
            initial_state={'soc': 0.8, 'soh': 0.92},
            process_noise=0.01,
            measurement_noise=0.05
        )
        
        state = ekf.get_state()
        
        assert isinstance(state, dict)
        assert 'soc' in state
        assert 'soh' in state
        assert 'resistance' in state


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
