"""
Battery BMS - Battery Management System using PINN and Kalman Filter

A comprehensive battery health management system that combines:
- Physics-Informed Neural Networks (PINN)
- Kalman Filtering
- Graph Neural Networks for battery pack modeling
"""

__version__ = "0.1.0"
__author__ = "Battery BMS Team"

from .core import BatteryBMS

__all__ = ['BatteryBMS']
