# Battery Health Management using PINN and Kalman Filter

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A data and physics-driven Battery Management System (BMS) that combines Physics-Informed Neural Networks (PINN) with Kalman Filtering for accurate battery health monitoring and state estimation.

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Architecture](#architecture)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Usage](#usage)
- [Theory and Background](#theory-and-background)
- [Project Structure](#project-structure)
- [Contributing](#contributing)
- [License](#license)

## 🔋 Overview

This project implements a sophisticated Battery Management System that leverages the power of:

- **Physics-Informed Neural Networks (PINN)**: Deep learning models that incorporate physical laws and constraints into the learning process
- **Kalman Filtering**: Optimal state estimation for real-time battery monitoring
- **Graph Neural Networks**: For modeling battery pack interactions and thermal dynamics

The system provides accurate estimates of:
- State of Charge (SOC)
- State of Health (SOH)
- Remaining Useful Life (RUL)
- Battery temperature distribution
- Internal resistance

## ✨ Features

- 🎯 **High Accuracy**: Physics-informed approach ensures predictions respect fundamental battery dynamics
- ⚡ **Real-time Monitoring**: Kalman filter enables efficient online state estimation
- 🔬 **Data-Driven**: Neural networks learn patterns from historical battery data
- 🌡️ **Thermal Management**: Graph-based modeling of temperature distribution across battery packs
- 📊 **Visualization**: Comprehensive plotting and analysis tools
- 🔄 **Modular Design**: Easy to extend and customize for different battery chemistries

## 🏗️ Architecture

```
┌─────────────────┐
│  Battery Data   │
│  (Voltage, I,   │
│   Temperature)  │
└────────┬────────┘
         │
         ▼
┌─────────────────────────────────────┐
│   Physics-Informed Neural Network   │
│  - Learns battery dynamics          │
│  - Respects physical constraints    │
│  - Estimates SOC/SOH/RUL            │
└────────┬────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────┐
│       Kalman Filter                 │
│  - Real-time state estimation       │
│  - Noise filtering                  │
│  - Sensor fusion                    │
└────────┬────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────┐
│     Output & Visualization          │
│  - SOC, SOH, RUL predictions        │
│  - Confidence intervals             │
│  - Performance metrics              │
└─────────────────────────────────────┘
```

## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Setup

1. Clone the repository:
```bash
git clone https://github.com/sazzad1008/Battery-health-management-using-PINN-and-Kalman-Filter-.git
cd Battery-health-management-using-PINN-and-Kalman-Filter-
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## 🎯 Quick Start

Here's a simple example to get you started:

```python
from battery_bms import BatteryBMS
from battery_bms.data import load_battery_data

# Load battery data
data = load_battery_data('path/to/battery_data.csv')

# Initialize BMS with PINN and Kalman Filter
bms = BatteryBMS(
    use_pinn=True,
    use_kalman=True,
    battery_chemistry='LiFePO4'
)

# Train the model
bms.train(data['train'], epochs=100)

# Predict battery states
predictions = bms.predict(data['test'])

# Display results
print(f"SOC: {predictions['soc']:.2%}")
print(f"SOH: {predictions['soh']:.2%}")
print(f"RUL: {predictions['rul']:.0f} cycles")

# Visualize results
bms.plot_results(predictions)
```

## 📚 Usage

### 1. Data Preparation

```python
from battery_bms.data import BatteryDataLoader

# Load and preprocess battery data
loader = BatteryDataLoader()
data = loader.load('battery_data.csv')
data = loader.preprocess(data, normalize=True)
```

### 2. Training PINN Model

```python
from battery_bms.models import PINN

# Initialize PINN with physics constraints
pinn = PINN(
    input_dim=4,  # Voltage, Current, Temperature, Time
    hidden_layers=[64, 128, 64],
    physics_weight=0.5  # Balance between data and physics
)

# Train with physics-informed loss
pinn.train(
    data=data,
    epochs=200,
    learning_rate=0.001
)
```

### 3. State Estimation with Kalman Filter

```python
from battery_bms.filters import ExtendedKalmanFilter

# Initialize Kalman filter
ekf = ExtendedKalmanFilter(
    initial_state={'soc': 1.0, 'soh': 1.0},
    process_noise=0.01,
    measurement_noise=0.05
)

# Real-time state update
for measurement in battery_measurements:
    state = ekf.update(measurement)
    print(f"Estimated SOC: {state['soc']:.2%}")
```

### 4. Complete Pipeline

```python
from battery_bms import BatteryBMS

# Create complete BMS pipeline
bms = BatteryBMS()

# Load and train
bms.load_data('battery_data.csv')
bms.train(epochs=100)

# Evaluate
metrics = bms.evaluate()
print(f"MAE: {metrics['mae']:.4f}")
print(f"RMSE: {metrics['rmse']:.4f}")

# Save model
bms.save('trained_bms_model.pkl')
```

## 📖 Theory and Background

### Physics-Informed Neural Networks (PINN)

PINNs integrate physical laws directly into the neural network training process. For battery systems, we incorporate:

1. **Electrochemical Constraints**: 
   - Butler-Volmer equation for reaction kinetics
   - Fick's law for diffusion
   - Ohm's law for electrical resistance

2. **Energy Conservation**:
   - ∂E/∂t = I·V - P_loss - P_heat

3. **Thermal Dynamics**:
   - dT/dt = (Q_gen - Q_loss) / (m·c_p)

### Kalman Filtering

The Extended Kalman Filter (EKF) provides optimal state estimation by:

1. **Prediction Step**: Project current state forward using battery model
2. **Update Step**: Correct prediction using new measurements
3. **Covariance Tracking**: Maintain uncertainty estimates

State-space model:
```
x(k+1) = f(x(k), u(k)) + w(k)  # State equation
y(k) = h(x(k)) + v(k)          # Measurement equation
```

Where:
- x: State vector [SOC, SOH, internal resistance]
- u: Input vector [current, temperature]
- y: Measurements [voltage, temperature]
- w, v: Process and measurement noise

### Battery Models

1. **Equivalent Circuit Model (ECM)**: RC networks representing battery dynamics
2. **Electrochemical Model**: P2D (Pseudo-2D) model for detailed physics
3. **Data-Driven Model**: Neural networks learning from battery data

## 📁 Project Structure

```
Battery-health-management-using-PINN-and-Kalman-Filter-/
├── README.md                 # This file
├── requirements.txt          # Python dependencies
├── setup.py                  # Package setup
├── .gitignore               # Git ignore file
│
├── battery_bms/             # Main package
│   ├── __init__.py
│   ├── core.py              # Core BMS class
│   ├── models/              # Model implementations
│   │   ├── __init__.py
│   │   ├── pinn.py          # Physics-Informed Neural Network
│   │   ├── battery_model.py # Battery equivalent circuit models
│   │   └── graph_model.py   # Graph neural network for pack modeling
│   ├── filters/             # State estimation filters
│   │   ├── __init__.py
│   │   ├── kalman.py        # Kalman filter implementations
│   │   └── particle.py      # Particle filter (alternative)
│   ├── data/                # Data processing utilities
│   │   ├── __init__.py
│   │   ├── loader.py        # Data loading
│   │   └── preprocessing.py # Data preprocessing
│   └── utils/               # Utility functions
│       ├── __init__.py
│       ├── visualization.py # Plotting functions
│       └── metrics.py       # Evaluation metrics
│
├── examples/                # Example scripts and notebooks
│   ├── basic_usage.py
│   ├── advanced_training.py
│   └── tutorial.ipynb       # Jupyter notebook tutorial
│
├── tests/                   # Unit tests
│   ├── __init__.py
│   ├── test_models.py
│   ├── test_filters.py
│   └── test_integration.py
│
└── data/                    # Sample data (not included in repo)
    └── README.md            # Data format specification
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📧 Contact

For questions or feedback, please open an issue on GitHub.

## 🙏 Acknowledgments

- Battery dataset providers (NASA, CALCE, etc.)
- Physics-informed machine learning research community
- Open-source contributors

## 📚 References

1. Raissi, M., Perdikaris, P., & Karniadakis, G. E. (2019). Physics-informed neural networks. Journal of Computational Physics.
2. Plett, G. L. (2004). Extended Kalman filtering for battery management systems of LiPB-based HEV battery packs.
3. Roman, D., et al. (2021). Machine learning pipeline for battery state-of-health estimation.
