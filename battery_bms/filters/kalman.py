"""
Kalman Filter implementations for battery state estimation.

This module provides Kalman Filter (KF) and Extended Kalman Filter (EKF)
for optimal state estimation of battery parameters.
"""

import numpy as np
from typing import Dict, Optional, Tuple


class KalmanFilter:
    """
    Linear Kalman Filter for state estimation.
    
    Implements the standard Kalman filter for linear systems:
    - Prediction step: x(k+1) = A*x(k) + B*u(k) + w
    - Update step: Correct prediction using measurements
    
    Parameters
    ----------
    A : np.ndarray
        State transition matrix
    B : np.ndarray
        Control input matrix
    H : np.ndarray
        Measurement matrix
    Q : np.ndarray
        Process noise covariance
    R : np.ndarray
        Measurement noise covariance
    initial_state : np.ndarray
        Initial state estimate
    initial_P : np.ndarray, optional
        Initial state covariance (default: identity matrix)
    
    Examples
    --------
    >>> A = np.array([[1, 0.1], [0, 1]])
    >>> B = np.array([[0], [0.1]])
    >>> H = np.array([[1, 0]])
    >>> Q = np.eye(2) * 0.01
    >>> R = np.array([[0.1]])
    >>> kf = KalmanFilter(A, B, H, Q, R, initial_state=np.array([1.0, 0.0]))
    >>> state = kf.update(measurement=0.95, control_input=0.0)
    """
    
    def __init__(
        self,
        A: np.ndarray,
        B: np.ndarray,
        H: np.ndarray,
        Q: np.ndarray,
        R: np.ndarray,
        initial_state: np.ndarray,
        initial_P: Optional[np.ndarray] = None
    ):
        self.A = A  # State transition matrix
        self.B = B  # Control matrix
        self.H = H  # Measurement matrix
        self.Q = Q  # Process noise covariance
        self.R = R  # Measurement noise covariance
        
        self.x = initial_state  # State estimate
        
        if initial_P is None:
            self.P = np.eye(len(initial_state)) * 0.1
        else:
            self.P = initial_P  # State covariance
    
    def predict(self, u: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Prediction step of Kalman filter.
        
        Parameters
        ----------
        u : np.ndarray, optional
            Control input
            
        Returns
        -------
        np.ndarray
            Predicted state
        """
        # Predict state
        if u is not None:
            self.x = self.A @ self.x + self.B @ u
        else:
            self.x = self.A @ self.x
        
        # Predict covariance
        self.P = self.A @ self.P @ self.A.T + self.Q
        
        return self.x
    
    def update(self, z: np.ndarray, u: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Update step of Kalman filter.
        
        Parameters
        ----------
        z : np.ndarray
            Measurement
        u : np.ndarray, optional
            Control input
            
        Returns
        -------
        np.ndarray
            Updated state estimate
        """
        # Prediction
        self.predict(u)
        
        # Innovation (measurement residual)
        y = z - self.H @ self.x
        
        # Innovation covariance
        S = self.H @ self.P @ self.H.T + self.R
        
        # Kalman gain
        K = self.P @ self.H.T @ np.linalg.inv(S)
        
        # Update state estimate
        self.x = self.x + K @ y
        
        # Update covariance
        I = np.eye(len(self.x))
        self.P = (I - K @ self.H) @ self.P
        
        return self.x
    
    def get_state(self) -> np.ndarray:
        """Get current state estimate."""
        return self.x
    
    def get_covariance(self) -> np.ndarray:
        """Get current state covariance."""
        return self.P


class ExtendedKalmanFilter:
    """
    Extended Kalman Filter (EKF) for nonlinear battery state estimation.
    
    The EKF handles nonlinear battery dynamics by linearizing the system
    around the current state estimate. It estimates:
    - State of Charge (SOC)
    - State of Health (SOH)
    - Internal resistance
    
    Parameters
    ----------
    initial_state : dict
        Initial state estimates {'soc': value, 'soh': value, ...}
    process_noise : float, optional
        Process noise level (default: 0.01)
    measurement_noise : float, optional
        Measurement noise level (default: 0.05)
    
    Examples
    --------
    >>> ekf = ExtendedKalmanFilter(
    ...     initial_state={'soc': 1.0, 'soh': 1.0},
    ...     process_noise=0.01,
    ...     measurement_noise=0.05
    ... )
    >>> measurement = {'voltage': 3.3, 'current': 1.0, 'temperature': 25.0}
    >>> state = ekf.update(measurement)
    >>> print(f"SOC: {state['soc']:.2%}")
    """
    
    def __init__(
        self,
        initial_state: Dict[str, float],
        process_noise: float = 0.01,
        measurement_noise: float = 0.05
    ):
        # State vector: [SOC, SOH, R0]
        self.x = np.array([
            initial_state.get('soc', 1.0),
            initial_state.get('soh', 1.0),
            initial_state.get('resistance', 0.05)
        ])
        
        # State covariance matrix
        self.P = np.eye(3) * 0.1
        
        # Process noise covariance
        self.Q = np.eye(3) * process_noise
        
        # Measurement noise covariance
        self.R = np.array([[measurement_noise]])
        
        # Battery parameters
        self.capacity = 2.3  # Ah
        self.dt = 1.0  # Time step in seconds
        
    def state_transition(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        """
        Nonlinear state transition function.
        
        Parameters
        ----------
        x : np.ndarray
            Current state [SOC, SOH, R0]
        u : np.ndarray
            Control input [current, temperature]
            
        Returns
        -------
        np.ndarray
            Next state
        """
        soc, soh, r0 = x
        current, temperature = u
        
        # SOC dynamics (Coulomb counting)
        delta_soc = -current * self.dt / (3600.0 * self.capacity * soh)
        soc_next = np.clip(soc + delta_soc, 0.0, 1.0)
        
        # SOH dynamics (slow degradation)
        # Simplified: SOH decreases with cycling and temperature
        aging_rate = 1e-6 * (1 + 0.01 * (temperature - 25))
        soh_next = max(soh - aging_rate * self.dt, 0.5)
        
        # Internal resistance (increases with aging)
        r0_next = r0 * (1 + 0.001 * (1 - soh))
        
        return np.array([soc_next, soh_next, r0_next])
    
    def measurement_function(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        """
        Nonlinear measurement function (voltage model).
        
        Parameters
        ----------
        x : np.ndarray
            State [SOC, SOH, R0]
        u : np.ndarray
            Input [current, temperature]
            
        Returns
        -------
        np.ndarray
            Predicted measurement (voltage)
        """
        soc, soh, r0 = x
        current, _ = u
        
        # OCV as function of SOC (simplified)
        ocv = 3.2 + 0.5 * soc - 0.3 * (soc - 0.5) ** 2
        
        # Terminal voltage
        voltage = ocv - current * r0
        
        return np.array([voltage])
    
    def jacobian_F(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        """
        Compute Jacobian of state transition function.
        
        Parameters
        ----------
        x : np.ndarray
            State vector
        u : np.ndarray
            Control input
            
        Returns
        -------
        np.ndarray
            Jacobian matrix F
        """
        soc, soh, r0 = x
        current, temperature = u
        
        # Numerical Jacobian (simplified)
        F = np.array([
            [1.0, -current * self.dt / (3600 * self.capacity * soh**2), 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.001 * r0, 1.0]
        ])
        
        return F
    
    def jacobian_H(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        """
        Compute Jacobian of measurement function.
        
        Parameters
        ----------
        x : np.ndarray
            State vector
        u : np.ndarray
            Control input
            
        Returns
        -------
        np.ndarray
            Jacobian matrix H
        """
        soc, soh, r0 = x
        current, _ = u
        
        # dV/dSOC
        dV_dsoc = 0.5 - 0.6 * (soc - 0.5)
        
        # dV/dSOH (approximately 0 for voltage measurement)
        dV_dsoh = 0.0
        
        # dV/dR0
        dV_dr0 = -current
        
        H = np.array([[dV_dsoc, dV_dsoh, dV_dr0]])
        
        return H
    
    def predict(self, u: np.ndarray) -> np.ndarray:
        """
        EKF prediction step.
        
        Parameters
        ----------
        u : np.ndarray
            Control input [current, temperature]
            
        Returns
        -------
        np.ndarray
            Predicted state
        """
        # Predict state
        self.x = self.state_transition(self.x, u)
        
        # Linearize around current state
        F = self.jacobian_F(self.x, u)
        
        # Predict covariance
        self.P = F @ self.P @ F.T + self.Q
        
        return self.x
    
    def update(self, measurement: Dict[str, float]) -> Dict[str, float]:
        """
        EKF update step with measurement.
        
        Parameters
        ----------
        measurement : dict
            Measurement data {'voltage': v, 'current': i, 'temperature': t}
            
        Returns
        -------
        dict
            Updated state estimates
        """
        # Extract measurement data
        z = np.array([measurement['voltage']])
        u = np.array([measurement['current'], measurement['temperature']])
        
        # Prediction step
        self.predict(u)
        
        # Measurement prediction
        z_pred = self.measurement_function(self.x, u)
        
        # Innovation
        y = z - z_pred
        
        # Linearize measurement function
        H = self.jacobian_H(self.x, u)
        
        # Innovation covariance
        S = H @ self.P @ H.T + self.R
        
        # Kalman gain
        K = self.P @ H.T @ np.linalg.inv(S)
        
        # Update state
        self.x = self.x + (K @ y).flatten()
        
        # Ensure physical constraints
        self.x[0] = np.clip(self.x[0], 0.0, 1.0)  # SOC
        self.x[1] = np.clip(self.x[1], 0.5, 1.0)  # SOH
        self.x[2] = max(self.x[2], 0.01)  # R0
        
        # Update covariance
        I = np.eye(len(self.x))
        self.P = (I - K @ H) @ self.P
        
        # Return state as dictionary
        return {
            'soc': self.x[0],
            'soh': self.x[1],
            'resistance': self.x[2],
            'covariance': self.P
        }
    
    def get_state(self) -> Dict[str, float]:
        """Get current state estimate."""
        return {
            'soc': self.x[0],
            'soh': self.x[1],
            'resistance': self.x[2]
        }
