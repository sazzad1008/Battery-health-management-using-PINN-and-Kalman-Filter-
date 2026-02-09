# Contributing to Battery Health Management System

Thank you for your interest in contributing to this project! We welcome contributions from the community.

## 🌟 Ways to Contribute

- **Report bugs** by opening an issue
- **Suggest enhancements** for new features
- **Improve documentation** to help others understand the code
- **Submit pull requests** with bug fixes or new features
- **Share your use cases** and results

## 🚀 Getting Started

### 1. Fork and Clone

```bash
# Fork the repository on GitHub, then clone your fork
git clone https://github.com/YOUR_USERNAME/Battery-health-management-using-PINN-and-Kalman-Filter-.git
cd Battery-health-management-using-PINN-and-Kalman-Filter-
```

### 2. Set Up Development Environment

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in development mode with dev dependencies
pip install -e ".[dev]"
```

### 3. Create a Branch

```bash
git checkout -b feature/your-feature-name
# or
git checkout -b fix/your-bug-fix
```

## 📝 Coding Guidelines

### Code Style

- Follow PEP 8 style guide
- Use meaningful variable and function names
- Write docstrings for all functions and classes
- Keep functions focused and concise

### Documentation

All functions should have NumPy-style docstrings:

```python
def calculate_soc(voltage: float, current: float) -> float:
    """
    Calculate State of Charge from voltage and current.
    
    Parameters
    ----------
    voltage : float
        Terminal voltage in V
    current : float
        Current in A
        
    Returns
    -------
    float
        State of Charge (0 to 1)
        
    Examples
    --------
    >>> soc = calculate_soc(3.3, 1.0)
    >>> print(f"SOC: {soc:.2%}")
    SOC: 85.00%
    """
    # Implementation here
    pass
```

### Testing

- Write unit tests for new features
- Ensure all tests pass before submitting PR
- Aim for >80% code coverage

```bash
# Run tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=battery_bms --cov-report=html
```

## 🔧 Development Workflow

### 1. Make Your Changes

- Write clean, well-documented code
- Add tests for new functionality
- Update documentation as needed

### 2. Test Your Changes

```bash
# Run unit tests
pytest tests/ -v

# Run example scripts
python examples/basic_usage.py
python examples/advanced_training.py

# Check code style (optional)
flake8 battery_bms/
pylint battery_bms/
```

### 3. Commit Your Changes

Write clear, descriptive commit messages:

```bash
git add .
git commit -m "Add feature: SOC estimation with UKF

- Implement Unscented Kalman Filter
- Add tests for UKF
- Update documentation
"
```

### 4. Push and Create Pull Request

```bash
git push origin feature/your-feature-name
```

Then create a Pull Request on GitHub with:
- Clear title and description
- Reference to related issues
- Screenshots (if applicable)
- Test results

## 📋 Pull Request Checklist

Before submitting your PR, ensure:

- [ ] Code follows project style guidelines
- [ ] All tests pass (`pytest tests/ -v`)
- [ ] New features have tests
- [ ] Documentation is updated
- [ ] Docstrings are complete
- [ ] Examples run without errors
- [ ] Commit messages are clear
- [ ] Branch is up-to-date with main

## 🐛 Reporting Bugs

When reporting bugs, please include:

1. **Description**: Clear description of the issue
2. **Steps to Reproduce**: Minimal code example
3. **Expected Behavior**: What you expected to happen
4. **Actual Behavior**: What actually happened
5. **Environment**: Python version, OS, package versions
6. **Error Messages**: Full error traceback

### Example Bug Report

```markdown
**Description**: PINN training fails with NaN loss

**Steps to Reproduce**:
```python
from battery_bms.models import PINN
pinn = PINN(input_dim=4, hidden_layers=[1000, 1000])
# ... training code
```

**Expected**: Training completes successfully
**Actual**: Loss becomes NaN after 10 epochs

**Environment**:
- Python 3.9
- NumPy 1.21.0
- Ubuntu 20.04

**Error**:
```
RuntimeWarning: invalid value encountered in add
```
```

## 💡 Suggesting Enhancements

For feature requests, please provide:

1. **Use Case**: Why is this feature needed?
2. **Proposed Solution**: How should it work?
3. **Alternatives**: Other solutions you've considered
4. **Example**: Code example showing desired usage

## 🏗️ Project Structure

Understanding the codebase:

```
battery_bms/
├── __init__.py          # Package initialization
├── core.py              # Main BatteryBMS class
├── models/              # Battery models
│   ├── pinn.py          # Physics-Informed NN
│   └── battery_model.py # Equivalent circuit models
├── filters/             # State estimation
│   └── kalman.py        # Kalman filters
├── data/                # Data utilities
│   ├── loader.py        # Data loading
│   └── preprocessing.py # Data preprocessing
└── utils/               # Helper functions
    ├── visualization.py # Plotting
    └── metrics.py       # Evaluation metrics
```

## 🎯 Priority Areas

We especially welcome contributions in:

1. **Additional Battery Models**: Different chemistries (NCA, LTO, etc.)
2. **Advanced Filters**: Particle filters, H-infinity filters
3. **Deep Learning**: Integration with PyTorch/TensorFlow
4. **Real-time Systems**: Optimization for embedded systems
5. **Datasets**: Integration with public battery datasets
6. **Visualization**: Interactive dashboards, real-time plots

## 📞 Getting Help

- **Questions**: Open a GitHub Discussion
- **Bugs**: Open a GitHub Issue
- **Security**: Email maintainers directly

## 📄 License

By contributing, you agree that your contributions will be licensed under the MIT License.

## 🙏 Acknowledgments

Thank you for making this project better! Your contributions help advance battery management technology.

---

**Happy Coding!** 🚀🔋
