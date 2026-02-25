# Quick Start Guide

Welcome to the Battery Health Management System! This guide will help you get started in 5 minutes.

## 📦 Installation

### Step 1: Clone the Repository

```bash
git clone https://github.com/sazzad1008/Battery-health-management-using-PINN-and-Kalman-Filter-.git
cd Battery-health-management-using-PINN-and-Kalman-Filter-
```

### Step 2: Install Dependencies

```bash
# Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install the package
pip install -e .
```

## 🚀 Run Your First Example

### Option 1: Basic Usage

Run the basic example to see the BMS in action:

```bash
python examples/basic_usage.py
```

**What it does:**
- Generates synthetic battery data
- Trains a PINN model with physics constraints
- Initializes Kalman filter for state estimation
- Predicts SOC, SOH, and RUL
- Shows performance metrics

### Option 2: Interactive Python

Try it interactively:

```python
from battery_bms import BatteryBMS
from battery_bms.data import BatteryDataLoader

# Generate test data
loader = BatteryDataLoader()
data = loader.generate_synthetic_data(n_samples=1000)

# Initialize and train BMS
bms = BatteryBMS(use_pinn=True, use_kalman=True)
bms.train(data, epochs=50)

# Make predictions
predictions = bms.predict(data)
print(f"SOC: {predictions['soc']:.2%}")
print(f"SOH: {predictions['soh']:.2%}")
print(f"RUL: {predictions['rul']:.0f} cycles")
```

## 🔍 Understanding the Output

When you run the basic example, you'll see:

```
State of Charge (SOC):       85.00%  ← Current battery charge level
State of Health (SOH):       92.00%  ← Battery health/capacity retention
Remaining Useful Life (RUL): 450 cycles  ← Estimated cycles until EOL
Temperature:                 25.0°C  ← Battery temperature
Voltage:                     3.250V  ← Terminal voltage
Internal Resistance:         55.00mΩ ← Battery internal resistance
```

## 📊 Using Your Own Data

### Data Format

Create a CSV file with these columns:

```csv
time,voltage,current,temperature,soc,soh
0.0,3.65,0.0,25.0,1.0,0.95
1.0,3.62,1.0,25.5,0.998,0.95
...
```

### Load and Use Your Data

```python
from battery_bms import BatteryBMS

# Load your data
bms = BatteryBMS()
bms.load_data('path/to/your/battery_data.csv')

# Train on your data
bms.train(epochs=100)

# Evaluate
metrics = bms.evaluate()
print(f"Accuracy (R²): {metrics['r2']:.4f}")
```

## 🧪 Run Tests

Verify everything is working:

```bash
pip install pytest
pytest tests/ -v
```

You should see: ✅ **16 passed**

## 📚 Next Steps

1. **Explore Advanced Features**: Check `examples/advanced_training.py`
2. **Read Full Documentation**: See the main README.md
3. **Customize Models**: Modify `battery_bms/models/pinn.py`
4. **Try Different Chemistries**: Use `battery_chemistry='NMC'` or `'LCO'`

## ❓ Common Issues

### Import Error
```
ModuleNotFoundError: No module named 'battery_bms'
```
**Solution**: Make sure you installed the package with `pip install -e .`

### NumPy/SciPy Not Found
```
ModuleNotFoundError: No module named 'numpy'
```
**Solution**: Install dependencies with `pip install -r requirements.txt`

## 💡 Tips

- Start with synthetic data to understand the system
- Use smaller epochs (20-50) for quick testing
- Increase `physics_weight` for more physically consistent predictions
- Check the data/README.md for dataset recommendations

## 🎯 What Makes This Special?

This BMS combines:
- **Physics**: Incorporates electrochemical laws into learning
- **Data**: Learns patterns from real battery behavior
- **Real-time**: Kalman filter provides online state estimation

Result: More accurate and reliable battery monitoring! 🔋

---

Need help? Check the [main README](README.md) or open an issue on GitHub!
