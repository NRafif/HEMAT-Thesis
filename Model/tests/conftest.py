"""
Pytest configuration and fixtures for testing
"""
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta


@pytest.fixture
def sample_uci_data():
    """Generate sample UCI-format data for testing"""
    dates = pd.date_range(start='2023-01-01', periods=100, freq='h')
    data = {
        'Global_active_power': np.random.uniform(0.5, 3.0, 100),
        'Global_reactive_power': np.random.uniform(0.1, 0.5, 100),
        'Voltage': np.random.uniform(235, 245, 100),
        'Global_intensity': np.random.uniform(2, 12, 100),
        'Sub_metering_1': np.random.uniform(0, 1, 100),
        'Sub_metering_2': np.random.uniform(0, 1, 100),
        'Sub_metering_3': np.random.uniform(0, 2, 100),
    }
    df = pd.DataFrame(data, index=dates)
    df.index.name = 'Datetime'
    return df


@pytest.fixture
def sample_PELITA_data():
    """Generate sample PELITA-format data for testing"""
    dates = pd.date_range(start='2025-11-27', periods=100, freq='h')
    data = {
        'Global_active_power': np.random.uniform(0.3, 2.0, 100),
        'Global_reactive_power': np.random.uniform(0.1, 0.4, 100),
        'Voltage': np.random.uniform(210, 230, 100),
        'Global_intensity': np.random.uniform(1, 10, 100),
        'Sub_metering_1': np.random.uniform(0, 0.5, 100),
        'Sub_metering_2': np.random.uniform(0, 0.5, 100),
        'Sub_metering_3': np.random.uniform(0, 1.5, 100),
        'Sub_metering_4': np.random.uniform(0, 0.3, 100),
        'Hour': [d.hour for d in dates],
        'Dayofweek': [d.dayofweek for d in dates],
        'Month': [d.month for d in dates],
        'Season': [1 if d.month in [5,6,7,8,9,10] else 2 for d in dates],
        'Emisi_CO2': np.random.uniform(0.2, 1.8, 100),
        'MCB_Tripped': np.random.choice([0, 1], 100, p=[0.95, 0.05]),
        'Lag_1': np.random.uniform(0.3, 2.0, 100),
        'Lag_24': np.random.uniform(0.3, 2.0, 100),
        'Lag_168': np.random.uniform(0.3, 2.0, 100),
        'Rolling_mean_3': np.random.uniform(0.3, 2.0, 100),
        'Rolling_mean_24': np.random.uniform(0.3, 2.0, 100),
        'Rolling_std_24': np.random.uniform(0.1, 0.5, 100),
        'House_ID': np.random.randint(1, 11, 100),
        'VA': np.random.choice([900, 1300, 2200], 100),
        'House_Type': np.random.choice(['working_class', 'retired', 'mixed'], 100),
    }
    df = pd.DataFrame(data, index=dates)
    df.index.name = 'Datetime'
    return df


@pytest.fixture
def temp_model_path(tmp_path):
    """Provide temporary path for model saving/loading tests"""
    return tmp_path / "test_model.joblib"
