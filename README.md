# Battery-health-management-using-PINN-and-Kalman-Filter-
Data and physicsdriven Battery Management System that combines physics-informed neural networks with graph learning for battery-pack modeling.

## Project layout
- `src/charge_discharge.py`: Charge/discharge PINN training script (from `CHarge_discharge.ipynb`).
- `src/current_integrated_online_learning.py`: Online learning PINN training script (from `Current Integrted_online_learning (another copy).ipynb`).
- `run.sh`: Convenience runner for the training scripts.
- `CHarge_discharge.ipynb`, `Current Integrted_online_learning (another copy).ipynb`: Original notebooks.

## Setup
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run
```bash
./run.sh charge_discharge
./run.sh online_learning
```
