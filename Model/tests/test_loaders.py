"""
Unit tests for data loaders
============================

Tests for UCILoader, PELITALoader, and UniversalLoader
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import os

from modules.data_loaders import (
    DatasetLoader,
    UCILoader,
    PELITALoader,
    UniversalLoader
)


class TestDatasetLoader:
    """Test abstract base class"""
    
    def test_abstract_methods(self):
        """Test that DatasetLoader cannot be instantiated"""
        with pytest.raises(TypeError):
            loader = DatasetLoader()


class TestUCILoader:
    """Test UCI dataset loader"""
    
    def test_initialization(self):
        """Test UCI loader can be initialized"""
        loader = UCILoader(verbose=False)
        assert loader is not None
        assert isinstance(loader, DatasetLoader)
    
    def test_required_columns(self):
        """Test that required columns are defined"""
        assert len(UCILoader.REQUIRED_COLUMNS) == 7
        assert 'Global_active_power' in UCILoader.REQUIRED_COLUMNS
        assert 'Sub_metering_1' in UCILoader.REQUIRED_COLUMNS
    
    def test_validate_with_valid_data(self, sample_uci_data):
        """Test validation passes with valid UCI data"""
        loader = UCILoader(verbose=False)
        assert loader.validate(sample_uci_data) is True
    
    def test_validate_missing_columns(self):
        """Test validation fails with missing columns"""
        loader = UCILoader(verbose=False)
        df = pd.DataFrame({'wrong_column': [1, 2, 3]})
        df.index = pd.date_range('2023-01-01', periods=3, freq='h')
        
        with pytest.raises(KeyError, match="Missing required columns"):
            loader.validate(df)
    
    def test_validate_non_datetime_index(self, sample_uci_data):
        """Test validation fails without DatetimeIndex"""
        loader = UCILoader(verbose=False)
        df = sample_uci_data.reset_index()
        
        with pytest.raises(ValueError, match="Index must be DatetimeIndex"):
            loader.validate(df)
    
    def test_validate_non_monotonic_index(self, sample_uci_data):
        """Test validation fails with non-monotonic index"""
        loader = UCILoader(verbose=False)
        df = sample_uci_data.copy()
        # Reverse the index
        df.index = df.index[::-1]
        
        with pytest.raises(ValueError, match="monotonically increasing"):
            loader.validate(df)
    
    def test_clip_outliers_iqr(self, sample_uci_data):
        """Test IQR outlier clipping"""
        loader = UCILoader(verbose=False)
        
        # Add extreme outlier
        df = sample_uci_data.copy()
        df.iloc[0, 0] = 1000  # Extreme value
        
        df_clipped = loader._clip_outliers_iqr(df)
        
        # Check that outlier was clipped
        assert df_clipped.iloc[0, 0] < 1000
        assert df_clipped.iloc[0, 0] > 0
    
    def test_resample_hourly(self, sample_uci_data):
        """Test hourly resampling"""
        loader = UCILoader(verbose=False)
        
        # Create minute-level data
        df_minute = sample_uci_data.copy()
        df_minute.index = pd.date_range('2023-01-01', periods=len(df_minute), freq='min')
        
        df_hourly = loader._resample_hourly(df_minute)
        
        # Check that data is hourly
        assert len(df_hourly) < len(df_minute)
        # Check that index frequency is hourly
        assert df_hourly.index.freq is not None or len(df_hourly) > 1


class TestPELITALoader:
    """Test PELITA dataset loader"""
    
    def test_initialization(self):
        """Test PELITA loader can be initialized"""
        loader = PELITALoader(verbose=False)
        assert loader is not None
        assert isinstance(loader, DatasetLoader)
    
    def test_required_columns(self):
        """Test that required columns include PELITA-specific ones"""
        assert len(PELITALoader.REQUIRED_COLUMNS) >= 11
        assert 'MCB_Tripped' in PELITALoader.REQUIRED_COLUMNS
        assert 'VA' in PELITALoader.REQUIRED_COLUMNS
        assert 'House_Type' in PELITALoader.REQUIRED_COLUMNS
        assert 'Sub_metering_4' in PELITALoader.REQUIRED_COLUMNS
    
    def test_PELITA_specific_columns(self):
        """Test PELITA-specific column list"""
        assert 'MCB_Tripped' in PELITALoader.PELITA_SPECIFIC
        assert 'VA' in PELITALoader.PELITA_SPECIFIC
        assert 'Lag_1' in PELITALoader.PELITA_SPECIFIC
        assert 'Rolling_mean_3' in PELITALoader.PELITA_SPECIFIC
    
    def test_validate_with_valid_data(self, sample_PELITA_data):
        """Test validation passes with valid PELITA data"""
        loader = PELITALoader(verbose=False)
        assert loader.validate(sample_PELITA_data) is True
    
    def test_validate_invalid_va_values(self, sample_PELITA_data):
        """Test validation fails with invalid VA values"""
        loader = PELITALoader(verbose=False)
        df = sample_PELITA_data.copy()
        df['VA'] = 1500  # Invalid VA
        
        with pytest.raises(ValueError, match="Invalid VA values"):
            loader.validate(df)
    
    def test_validate_invalid_mcb_values(self, sample_PELITA_data):
        """Test validation fails with invalid MCB_Tripped values"""
        loader = PELITALoader(verbose=False)
        df = sample_PELITA_data.copy()
        df['MCB_Tripped'] = 2  # Invalid (should be 0 or 1)
        
        with pytest.raises(ValueError, match="Invalid MCB_Tripped values"):
            loader.validate(df)
    
    def test_validate_invalid_house_type(self, sample_PELITA_data):
        """Test validation fails with invalid House_Type"""
        loader = PELITALoader(verbose=False)
        df = sample_PELITA_data.copy()
        df['House_Type'] = 'invalid_type'
        
        with pytest.raises(ValueError, match="Invalid House_Type values"):
            loader.validate(df)
    
    def test_conditional_outlier_handling(self, sample_PELITA_data):
        """Test that MCB trip zeros are preserved"""
        loader = PELITALoader(verbose=False)
        df = sample_PELITA_data.copy()
        
        # Set some rows to MCB tripped with power=0
        df.loc[df.index[0], 'MCB_Tripped'] = 1
        df.loc[df.index[0], 'Global_active_power'] = 0
        
        df_processed = loader._conditional_outlier_handling(df)
        
        # Check that MCB trip zero is preserved
        assert df_processed.loc[df.index[0], 'Global_active_power'] == 0
        assert df_processed.loc[df.index[0], 'MCB_Tripped'] == 1
    
    def test_get_va_statistics(self, sample_PELITA_data):
        """Test VA statistics calculation"""
        loader = PELITALoader(verbose=False)
        stats = loader.get_va_statistics(sample_PELITA_data)
        
        assert isinstance(stats, dict)
        # Should have stats for VA classes present in data
        for key in stats.keys():
            assert 'VA' in key
            assert 'mean_power' in stats[key]
            assert 'mcb_trip_rate' in stats[key]
            assert 'utilization' in stats[key]


class TestUniversalLoader:
    """Test Universal loader with auto-detection"""
    
    def test_initialization(self):
        """Test Universal loader can be initialized"""
        loader = UniversalLoader(verbose=False)
        assert loader is not None
        assert isinstance(loader, DatasetLoader)
        assert loader.detected_type is None
    
    def test_detect_PELITA_type(self, tmp_path):
        """Test detection of PELITA dataset"""
        loader = UniversalLoader(verbose=False)
        
        # Create temporary PELITA-like CSV
        df = pd.DataFrame({
            'Datetime': pd.date_range('2025-01-01', periods=10, freq='h'),
            'Global_active_power': np.random.rand(10),
            'MCB_Tripped': [0] * 10,
            'VA': [1300] * 10,
            'House_Type': ['working_class'] * 10
        })
        
        temp_file = tmp_path / "PELITA_test.csv"
        df.to_csv(temp_file, index=False)
        
        detected = loader.detect_type(str(temp_file))
        assert detected == 'PELITA'
    
    def test_detect_uci_type(self, tmp_path):
        """Test detection of UCI dataset"""
        loader = UniversalLoader(verbose=False)
        
        # Create temporary UCI-like CSV
        df = pd.DataFrame({
            'Date': ['01/01/2023'] * 10,
            'Time': ['00:00:00'] * 10,
            'Global_active_power': np.random.rand(10),
            'Voltage': [240] * 10,
            'Sub_metering_1': np.random.rand(10)
        })
        
        temp_file = tmp_path / "uci_test.csv"
        df.to_csv(temp_file, index=False, sep=';')
        
        detected = loader.detect_type(str(temp_file))
        assert detected == 'uci'
    
    def test_get_loader_type_before_load(self):
        """Test get_loader_type returns None before loading"""
        loader = UniversalLoader(verbose=False)
        assert loader.get_loader_type() is None
    
    def test_validate_before_load_raises_error(self, sample_uci_data):
        """Test that validate raises error if called before load"""
        loader = UniversalLoader(verbose=False)
        
        with pytest.raises(RuntimeError, match="Must call load"):
            loader.validate(sample_uci_data)


@pytest.mark.integration
class TestLoaderIntegration:
    """Integration tests for loaders"""
    
    def test_uci_loader_full_pipeline(self, tmp_path):
        """Test full UCI loading pipeline"""
        # Create temporary UCI-format file
        df = pd.DataFrame({
            'Date': ['01/01/2023'] * 100,
            'Time': [f'{i:02d}:00:00' for i in range(100)],
            'Global_active_power': np.random.uniform(0.5, 3.0, 100),
            'Global_reactive_power': np.random.uniform(0.1, 0.5, 100),
            'Voltage': np.random.uniform(235, 245, 100),
            'Global_intensity': np.random.uniform(2, 12, 100),
            'Sub_metering_1': np.random.uniform(0, 1, 100),
            'Sub_metering_2': np.random.uniform(0, 1, 100),
            'Sub_metering_3': np.random.uniform(0, 2, 100),
        })
        
        temp_file = tmp_path / "uci_test.txt"
        df.to_csv(temp_file, index=False, sep=';')
        
        # Load
        loader = UCILoader(verbose=False)
        df_loaded = loader.load(str(temp_file))
        
        # Verify
        assert isinstance(df_loaded, pd.DataFrame)
        assert isinstance(df_loaded.index, pd.DatetimeIndex)
        assert len(df_loaded) > 0
        assert all(col in df_loaded.columns for col in UCILoader.REQUIRED_COLUMNS)
    
    def test_PELITA_loader_full_pipeline(self, tmp_path):
        """Test full PELITA loading pipeline"""
        # Create temporary PELITA-format file
        dates = pd.date_range('2025-01-01', periods=100, freq='h')
        df = pd.DataFrame({
            'Datetime': dates,
            'Global_active_power': np.random.uniform(0.3, 2.0, 100),
            'Global_reactive_power': np.random.uniform(0.1, 0.4, 100),
            'Voltage': np.random.uniform(210, 230, 100),
            'Global_intensity': np.random.uniform(1, 10, 100),
            'Sub_metering_1': np.random.uniform(0, 0.5, 100),
            'Sub_metering_2': np.random.uniform(0, 0.5, 100),
            'Sub_metering_3': np.random.uniform(0, 1.5, 100),
            'Sub_metering_4': np.random.uniform(0, 0.3, 100),
            'MCB_Tripped': np.random.choice([0, 1], 100, p=[0.95, 0.05]),
            'VA': np.random.choice([900, 1300, 2200], 100),
            'House_Type': np.random.choice(['working_class', 'retired', 'mixed'], 100),
            'House_ID': 1,
        })
        
        temp_file = tmp_path / "PELITA_test.csv"
        df.to_csv(temp_file, index=False)
        
        # Load
        loader = PELITALoader(verbose=False)
        df_loaded = loader.load(str(temp_file))
        
        # Verify
        assert isinstance(df_loaded, pd.DataFrame)
        assert isinstance(df_loaded.index, pd.DatetimeIndex)
        assert len(df_loaded) > 0
        assert all(col in df_loaded.columns for col in PELITALoader.REQUIRED_COLUMNS)
        
        # Check info
        info = loader.get_info()
        assert info['type'] == 'PELITA'
        assert 'mcb_trip_rate' in info
