# Battery Data Format

This directory should contain battery datasets for training and testing the BMS models.

## Data Format

The expected data format is CSV with the following columns:

| Column      | Description                          | Unit | Range      |
|-------------|--------------------------------------|------|------------|
| time        | Timestamp or time index              | s    | 0+         |
| voltage     | Terminal voltage                     | V    | 2.0-4.2    |
| current     | Current (positive=discharge)         | A    | -5 to +5   |
| temperature | Battery temperature                  | °C   | -20 to 60  |
| soc         | State of Charge (optional)           | -    | 0-1        |
| soh         | State of Health (optional)           | -    | 0-1        |

### Example CSV Format

```csv
time,voltage,current,temperature,soc,soh
0.0,3.65,0.0,25.0,1.0,0.95
1.0,3.62,1.0,25.5,0.998,0.95
2.0,3.60,1.0,26.0,0.996,0.95
...
```

## Available Datasets

### Public Battery Datasets

1. **NASA Battery Dataset**
   - Source: https://ti.arc.nasa.gov/tech/dash/groups/pcoe/prognostic-data-repository/
   - Description: Li-ion battery aging data
   - Format: MAT files

2. **CALCE Battery Dataset**
   - Source: https://calce.umd.edu/battery-data
   - Description: Various battery chemistries under different conditions
   - Format: MAT, CSV

3. **Oxford Battery Degradation Dataset**
   - Source: https://ora.ox.ac.uk/objects/uuid:03ba4b01-cfed-46d3-9b1a-7d4a7bdf6fac
   - Description: Commercial Li-ion cells under various fast charging conditions

## Generating Synthetic Data

If you don't have real battery data, you can generate synthetic data using:

```python
from battery_bms.data import BatteryDataLoader

loader = BatteryDataLoader()
data = loader.generate_synthetic_data(
    n_samples=1000,
    discharge_profile='dynamic'
)
```

## Data Preprocessing

The BMS package includes preprocessing utilities:

```python
from battery_bms.data import preprocess_battery_data

processed_data = preprocess_battery_data(
    data,
    normalize=True,
    remove_outliers=True,
    fill_missing=True
)
```

## Notes

- Keep large data files (.csv, .h5, .mat) in this directory
- This directory is excluded from version control (see .gitignore)
- For sharing, provide download links rather than committing large files
