"""
Battery Equivalent Circuit Models.

This module implements various equivalent circuit models (ECM) for battery simulation,
including RC networks and thermal models.
"""

import numpy as np
from typing import Dict, Tuple


class BatteryEquivalentCircuit:
    """
    Battery Equivalent Circuit Model (ECM).
    
    Implements a first-order RC equivalent circuit model for battery dynamics.
    The model includes:
    - Open circuit voltage (OCV) as function of SOC
    - Internal resistance (R0)
    - RC pair for transient response (R1, C1)
    
    Parameters
    ----------
    capacity : float
        Battery capacity in Ah
    r0 : float, optional
        Series resistance in Ohms (default: 0.05)
    r1 : float, optional
        Polarization resistance in Ohms (default: 0.03)
    c1 : float, optional
        Polarization capacitance in F (default: 1000)
    
    Examples
    --------
    >>> ecm = BatteryEquivalentCircuit(capacity=2.3)
    >>> voltage = ecm.simulate_voltage(current=1.0, soc=0.8, temperature=25.0)
    """
    
    def __init__(
        self,
        capacity: float,
        r0: float = 0.05,
        r1: float = 0.03,
        c1: float = 1000.0
    ):
        self.capacity = capacity  # Ah
        self.r0 = r0  # Series resistance
        self.r1 = r1  # Polarization resistance
        self.c1 = c1  # Polarization capacitance
        
        # State variables
        self.soc = 1.0  # State of Charge
        self.v1 = 0.0   # Voltage across RC pair
        
    def ocv(self, soc: float) -> float:
        """
        Calculate Open Circuit Voltage (OCV) as a function of SOC.
        
        Uses a polynomial approximation of the OCV-SOC curve for LiFePO4.
        
        Parameters
        ----------
        soc : float
            State of Charge (0 to 1)
            
        Returns
        -------
        float
            Open circuit voltage in V
        """
        # LiFePO4 OCV-SOC curve (simplified polynomial)
        # Real implementations use lookup tables or better fitting
        soc = np.clip(soc, 0.0, 1.0)
        
        # Polynomial coefficients for LiFePO4
        ocv_voltage = (3.2 + 
                      0.5 * soc + 
                      -0.3 * (soc - 0.5) ** 2 + 
                      0.2 * (soc - 0.5) ** 3)
        
        return ocv_voltage
    
    def update_temperature_dependence(self, temperature: float) -> Tuple[float, float]:
        """
        Adjust resistances based on temperature.
        
        Parameters
        ----------
        temperature : float
            Temperature in Celsius
            
        Returns
        -------
        tuple
            (adjusted_r0, adjusted_r1)
        """
        # Temperature coefficient (simplified)
        # Resistance typically increases at lower temperatures
        T_ref = 25.0  # Reference temperature
        alpha = 0.01  # Temperature coefficient per °C
        
        temp_factor = 1.0 + alpha * (T_ref - temperature)
        
        r0_adj = self.r0 * temp_factor
        r1_adj = self.r1 * temp_factor
        
        return r0_adj, r1_adj
    
    def simulate_voltage(
        self,
        current: float,
        soc: float,
        temperature: float = 25.0,
        dt: float = 1.0
    ) -> float:
        """
        Simulate battery terminal voltage.
        
        Parameters
        ----------
        current : float
            Current in A (positive for discharge, negative for charge)
        soc : float
            State of Charge (0 to 1)
        temperature : float, optional
            Temperature in Celsius (default: 25)
        dt : float, optional
            Time step in seconds (default: 1.0)
            
        Returns
        -------
        float
            Terminal voltage in V
        """
        # Get temperature-adjusted resistances
        r0, r1 = self.update_temperature_dependence(temperature)
        
        # Update RC voltage (first-order dynamics)
        tau = r1 * self.c1  # Time constant
        self.v1 = self.v1 * np.exp(-dt / tau) + r1 * current * (1 - np.exp(-dt / tau))
        
        # Calculate terminal voltage
        ocv_voltage = self.ocv(soc)
        terminal_voltage = ocv_voltage - current * r0 - self.v1
        
        return terminal_voltage
    
    def update_soc(self, current: float, dt: float = 1.0) -> float:
        """
        Update State of Charge using Coulomb counting.
        
        Parameters
        ----------
        current : float
            Current in A (positive for discharge)
        dt : float, optional
            Time step in seconds (default: 1.0)
            
        Returns
        -------
        float
            Updated SOC (0 to 1)
        """
        # Coulomb counting: dSOC = -I * dt / (3600 * Q)
        delta_soc = -current * dt / (3600.0 * self.capacity)
        self.soc = np.clip(self.soc + delta_soc, 0.0, 1.0)
        
        return self.soc
    
    def reset(self, initial_soc: float = 1.0) -> None:
        """
        Reset the battery state.
        
        Parameters
        ----------
        initial_soc : float, optional
            Initial State of Charge (default: 1.0)
        """
        self.soc = initial_soc
        self.v1 = 0.0


class ThermalModel:
    """
    Simple thermal model for battery temperature dynamics.
    
    Models battery heating due to:
    - Joule heating (I²R losses)
    - Entropic heating
    - Heat transfer to ambient
    
    Parameters
    ----------
    mass : float
        Battery mass in kg
    specific_heat : float
        Specific heat capacity in J/(kg·K)
    heat_transfer_coeff : float
        Heat transfer coefficient in W/(m²·K)
    surface_area : float
        Surface area in m²
    """
    
    def __init__(
        self,
        mass: float = 0.05,
        specific_heat: float = 800.0,
        heat_transfer_coeff: float = 10.0,
        surface_area: float = 0.01
    ):
        self.mass = mass
        self.specific_heat = specific_heat
        self.h = heat_transfer_coeff
        self.A = surface_area
        
        self.temperature = 25.0  # Initial temperature in Celsius
    
    def update_temperature(
        self,
        current: float,
        voltage: float,
        ocv: float,
        ambient_temp: float = 25.0,
        dt: float = 1.0
    ) -> float:
        """
        Update battery temperature.
        
        Parameters
        ----------
        current : float
            Current in A
        voltage : float
            Terminal voltage in V
        ocv : float
            Open circuit voltage in V
        ambient_temp : float, optional
            Ambient temperature in Celsius (default: 25)
        dt : float, optional
            Time step in seconds (default: 1.0)
            
        Returns
        -------
        float
            Updated temperature in Celsius
        """
        # Heat generation (Joule heating)
        P_loss = current * (ocv - voltage)  # Power loss in W
        Q_gen = P_loss * dt  # Heat generated in J
        
        # Heat dissipation to ambient
        Q_loss = self.h * self.A * (self.temperature - ambient_temp) * dt
        
        # Temperature change
        dT = (Q_gen - Q_loss) / (self.mass * self.specific_heat)
        self.temperature += dT
        
        return self.temperature
