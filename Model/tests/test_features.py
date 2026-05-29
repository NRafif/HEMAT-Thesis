"""
Unit tests for feature engineering
===================================

Tests for FeatureEngineer, UCIFeatureEngineer, PELITAFeatureEngineer,
and UniversalFeatureEngineer
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from modules.feature_engineering import (
    FeatureEngineer,
    UCIFeatureEngineer,
    PELITAFeatureEngineer,
    UniversalFeatureEngineer
)


class TestFeatureEngineer:
    """Test base FeatureEngineer class"""
    
    def test_abstract_methods(self):
        """Test that FeatureEngineer cannot be instantiated"""
        with pytest.raises(TypeError):
            engineer = FeatureEngineer()
    
    def test_create_temporal_features(self, sample_uci_data):
        """Test temporal feature creation"""
        engineer = UCIFeatureEngineer(verbose=False)
        df = engineer.create_temporal_features(sample_uci_data)
        
        # Check that temporal features were created
        assert 'hour' in df.columns
        assert 'dayofweek' in df.columns
        assert 'day' in df.columns
        assert 'month' in df.columns
        assert 'season' in df.columns
        
        # Check value ranges
        assert df['hour'].min() >= 0
        assert df['hour'].max() <= 23
        assert df['dayofweek'].min() >= 0
        assert df['dayofweek'].max() <= 6
        assert df['month'].min() >= 1
        assert df['month'].max() <= 12
        assert df['season'].min() >= 1
        assert df['season'].max() <= 4
    
    def test_create_temporal_features_non_datetime_index(self):
        """Test that temporal features fail without DatetimeIndex"""
        engineer = UCIFeatureEngineer(verbose=False)
        df = pd.DataFrame({'value': [1, 2, 3]})
        
        with pytest.raises(ValueError, match="DatetimeIndex"):
            engineer.create_temporal_features(df)
    
    def test_create_lag_features(self, sample_uci_data):
        """Test lag feature creation"""
        engineer = UCIFeatureEngineer(verbose=False)
        df = engineer.create_lag_features(
            sample_uci_data,
            target_col='Global_active_power',
            lags=[1, 24]
        )
        
        # Check that lag features were created
        assert 'lag_1' in df.columns
        assert 'lag_24' in df.columns
        
        # Check that lag_1 is shifted correctly
        # lag_1 at index i should equal Global_active_power at index i-1
        for i in range(1, min(10, len(df))):
            expected = sample_uci_data.iloc[i-1]['Global_active_power']
            actual = df.iloc[i]['lag_1']
            assert np.isclose(expected, actual, rtol=1e-5)
    
    def test_create_lag_features_no_data_leakage(self, sample_uci_data):
        """Test that lag features don't use future data"""
        engineer = UCIFeatureEngineer(verbose=False)
        df = engineer.create_lag_features(
            sample_uci_data,
            target_col='Global_active_power',
            lags=[1]
        )
        
        # First row should have NaN for lag_1 (no past data)
        assert pd.isna(df.iloc[0]['lag_1'])
        
        # Second row lag_1 should equal first row power
        assert np.isclose(
            df.iloc[1]['lag_1'],
            sample_uci_data.iloc[0]['Global_active_power'],
            rtol=1e-5
        )
    
    def test_create_lag_features_missing_column(self, sample_uci_data):
        """Test that lag features fail with missing column"""
        engineer = UCIFeatureEngineer(verbose=False)
        
        with pytest.raises(KeyError, match="not found"):
            engineer.create_lag_features(
                sample_uci_data,
                target_col='nonexistent_column',
                lags=[1]
            )
    
    def test_create_rolling_features(self, sample_uci_data):
        """Test rolling feature creation"""
        engineer = UCIFeatureEngineer(verbose=False)
        df = engineer.create_rolling_features(
            sample_uci_data,
            target_col='Global_active_power',
            windows=[3, 24]
        )
        
        # Check that rolling features were created
        assert 'rolling_mean_3' in df.columns
        assert 'rolling_mean_24' in df.columns
        assert 'rolling_std_24' in df.columns
        
        # Check that rolling_mean_3 is calculated correctly
        # Should be mean of previous 3 values (due to shift(1))
        if len(df) >= 4:
            # At index 3, rolling_mean_3 should be mean of indices 0, 1, 2
            expected_mean = sample_uci_data.iloc[0:3]['Global_active_power'].mean()
            actual_mean = df.iloc[3]['rolling_mean_3']
            assert np.isclose(expected_mean, actual_mean, rtol=1e-5)
    
    def test_create_rolling_features_no_data_leakage(self, sample_uci_data):
        """Test that rolling features don't use future data"""
        engineer = UCIFeatureEngineer(verbose=False)
        df = engineer.create_rolling_features(
            sample_uci_data,
            target_col='Global_active_power',
            windows=[3]
        )
        
        # At any index i, rolling_mean_3 should NOT include value at index i
        # It should only use values before index i
        if len(df) >= 5:
            idx = 4
            # rolling_mean_3 at idx should be mean of [idx-3, idx-2, idx-1]
            expected = sample_uci_data.iloc[1:4]['Global_active_power'].mean()
            actual = df.iloc[idx]['rolling_mean_3']
            
            # Should NOT equal mean including current value
            wrong = sample_uci_data.iloc[2:5]['Global_active_power'].mean()
            
            assert np.isclose(expected, actual, rtol=1e-5)
            assert not np.isclose(wrong, actual, rtol=1e-5)
    
    def test_fill_missing_values_forward(self, sample_uci_data):
        """Test forward fill strategy"""
        engineer = UCIFeatureEngineer(verbose=False)
        
        # Create data with NaN
        df = sample_uci_data.copy()
        df.iloc[5, 0] = np.nan
        
        df_filled = engineer.fill_missing_values(df, strategy='forward')
        
        # Check that NaN was filled with previous value
        assert not pd.isna(df_filled.iloc[5, 0])
        assert df_filled.iloc[5, 0] == df.iloc[4, 0]
    
    def test_fill_missing_values_mean(self, sample_uci_data):
        """Test mean fill strategy"""
        engineer = UCIFeatureEngineer(verbose=False)
        
        # Create data with NaN
        df = sample_uci_data.copy()
        col_name = df.columns[0]
        original_mean = df[col_name].mean()
        df.iloc[5, 0] = np.nan
        
        df_filled = engineer.fill_missing_values(df, strategy='mean')
        
        # Check that NaN was filled
        assert not pd.isna(df_filled.iloc[5, 0])
    
    def test_fill_missing_values_zero(self, sample_uci_data):
        """Test zero fill strategy"""
        engineer = UCIFeatureEngineer(verbose=False)
        
        # Create data with NaN
        df = sample_uci_data.copy()
        df.iloc[5, 0] = np.nan
        
        df_filled = engineer.fill_missing_values(df, strategy='zero')
        
        # Check that NaN was filled with zero
        assert df_filled.iloc[5, 0] == 0


class TestUCIFeatureEngineer:
    """Test UCI feature engineer"""
    
    def test_initialization(self):
        """Test UCI engineer can be initialized"""
        engineer = UCIFeatureEngineer(verbose=False)
        assert engineer is not None
        assert isinstance(engineer, FeatureEngineer)
    
    def test_engineer_full_pipeline(self, sample_uci_data):
        """Test full UCI feature engineering pipeline"""
        engineer = UCIFeatureEngineer(verbose=False)
        df = engineer.engineer(sample_uci_data)
        
        # Check that all expected features were created
        expected_features = [
            'hour', 'dayofweek', 'day', 'month', 'season',
            'lag_1', 'lag_24', 'lag_168',
            'rolling_mean_3', 'rolling_mean_24', 'rolling_std_24'
        ]
        
        for feature in expected_features:
            assert feature in df.columns, f"Missing feature: {feature}"
        
        # Check that original columns are preserved
        assert 'Global_active_power' in df.columns
        assert 'Voltage' in df.columns
        
        # Check that no NaN values remain
        assert df.isnull().sum().sum() == 0
    
    def test_engineer_feature_count(self, sample_uci_data):
        """Test that correct number of features are created"""
        engineer = UCIFeatureEngineer(verbose=False)
        df = engineer.engineer(sample_uci_data)
        
        feature_names = engineer.get_feature_names()
        assert len(feature_names) == 11  # 5 temporal + 3 lag + 3 rolling


class TestPELITAFeatureEngineer:
    """Test PELITA feature engineer"""
    
    def test_initialization(self):
        """Test PELITA engineer can be initialized"""
        engineer = PELITAFeatureEngineer(verbose=False)
        assert engineer is not None
        assert isinstance(engineer, FeatureEngineer)
    
    def test_engineer_full_pipeline(self, sample_PELITA_data):
        """Test full PELITA feature engineering pipeline"""
        engineer = PELITAFeatureEngineer(verbose=False)
        df = engineer.engineer(sample_PELITA_data)
        
        # Check that features were renamed to lowercase
        assert 'lag_1' in df.columns
        assert 'lag_24' in df.columns
        assert 'rolling_mean_3' in df.columns
        
        # Check that PELITA-specific features were added
        assert 'VA_normalized' in df.columns
        
        # Check that House_Type was encoded
        ht_cols = [col for col in df.columns if col.startswith('HT_')]
        assert len(ht_cols) > 0
        
        # Check that original PELITA columns are preserved
        assert 'MCB_Tripped' in df.columns
        assert 'VA' in df.columns
    
    def test_rename_PELITA_features(self, sample_PELITA_data):
        """Test renaming of PELITA features"""
        engineer = PELITAFeatureEngineer(verbose=False)
        df = engineer._rename_PELITA_features(sample_PELITA_data)
        
        # Check that features were renamed
        assert 'lag_1' in df.columns
        assert 'Lag_1' not in df.columns
        assert 'rolling_mean_3' in df.columns
        assert 'Rolling_mean_3' not in df.columns
    
    def test_normalize_va(self, sample_PELITA_data):
        """Test VA normalization"""
        engineer = PELITAFeatureEngineer(verbose=False)
        df = engineer.normalize_va(sample_PELITA_data)
        
        # Check that VA_normalized was created
        assert 'VA_normalized' in df.columns
        
        # Check normalization values
        # 900VA → 0.409, 1300VA → 0.591, 2200VA → 1.0
        for va_value in df['VA'].unique():
            normalized = df[df['VA'] == va_value]['VA_normalized'].iloc[0]
            expected = va_value / 2200.0
            assert np.isclose(normalized, expected, rtol=1e-5)
        
        # Check that all values are in [0, 1]
        assert df['VA_normalized'].min() >= 0
        assert df['VA_normalized'].max() <= 1
    
    def test_encode_house_type(self, sample_PELITA_data):
        """Test House_Type encoding"""
        engineer = PELITAFeatureEngineer(verbose=False)
        df = engineer.encode_house_type(sample_PELITA_data)
        
        # Check that one-hot encoded columns were created
        ht_cols = [col for col in df.columns if col.startswith('HT_')]
        assert len(ht_cols) >= 1
        
        # Check that original House_Type column was removed
        assert 'House_Type' not in df.columns
        
        # Check that one-hot encoding is correct (sum should be 1 per row)
        ht_sum = df[ht_cols].sum(axis=1)
        assert all(ht_sum == 1)
    
    def test_engineer_preserves_mcb_data(self, sample_PELITA_data):
        """Test that MCB trip data is preserved"""
        engineer = PELITAFeatureEngineer(verbose=False)
        
        # Set some rows to MCB tripped with power=0
        df = sample_PELITA_data.copy()
        df.loc[df.index[0], 'MCB_Tripped'] = 1
        df.loc[df.index[0], 'Global_active_power'] = 0
        
        df_engineered = engineer.engineer(df)
        
        # Check that MCB trip data is preserved
        assert df_engineered.loc[df.index[0], 'MCB_Tripped'] == 1
        assert df_engineered.loc[df.index[0], 'Global_active_power'] == 0


class TestUniversalFeatureEngineer:
    """Test Universal feature engineer"""
    
    def test_initialization(self):
        """Test Universal engineer can be initialized"""
        engineer = UniversalFeatureEngineer(verbose=False)
        assert engineer is not None
        assert isinstance(engineer, FeatureEngineer)
    
    def test_detect_PELITA_type(self, sample_PELITA_data):
        """Test detection of PELITA dataset"""
        engineer = UniversalFeatureEngineer(verbose=False)
        detected = engineer.detect_type(sample_PELITA_data)
        assert detected == 'PELITA'
    
    def test_detect_uci_type(self, sample_uci_data):
        """Test detection of UCI dataset"""
        engineer = UniversalFeatureEngineer(verbose=False)
        detected = engineer.detect_type(sample_uci_data)
        assert detected == 'uci'
    
    def test_engineer_with_uci_data(self, sample_uci_data):
        """Test engineering with UCI data"""
        engineer = UniversalFeatureEngineer(verbose=False)
        df = engineer.engineer(sample_uci_data)
        
        # Should have UCI features
        assert 'lag_1' in df.columns
        assert 'rolling_mean_3' in df.columns
        
        # Should NOT have PELITA-specific features
        assert 'VA_normalized' not in df.columns
        assert 'MCB_Tripped' not in df.columns
    
    def test_engineer_with_PELITA_data(self, sample_PELITA_data):
        """Test engineering with PELITA data"""
        engineer = UniversalFeatureEngineer(verbose=False)
        df = engineer.engineer(sample_PELITA_data)
        
        # Should have PELITA features
        assert 'VA_normalized' in df.columns
        assert 'MCB_Tripped' in df.columns
        
        # Should have renamed features
        assert 'lag_1' in df.columns
        assert 'Lag_1' not in df.columns
    
    def test_engineer_with_explicit_type(self, sample_uci_data):
        """Test engineering with explicitly specified type"""
        engineer = UniversalFeatureEngineer(dataset_type='uci', verbose=False)
        df = engineer.engineer(sample_uci_data)
        
        # Should use UCI engineer
        assert engineer.dataset_type == 'uci'
        assert 'lag_1' in df.columns


@pytest.mark.integration
class TestFeatureEngineeringIntegration:
    """Integration tests for feature engineering"""
    
    def test_uci_full_pipeline(self, sample_uci_data):
        """Test full UCI pipeline from raw data to features"""
        engineer = UCIFeatureEngineer(verbose=False)
        
        # Engineer features
        df = engineer.engineer(sample_uci_data)
        
        # Verify data integrity
        assert len(df) > 0
        assert df.isnull().sum().sum() == 0
        
        # Verify feature count
        feature_names = engineer.get_feature_names()
        assert len(feature_names) > 0
        
        # Verify no data leakage in lag features
        for i in range(1, min(10, len(df))):
            assert np.isclose(
                df.iloc[i]['lag_1'],
                sample_uci_data.iloc[i-1]['Global_active_power'],
                rtol=1e-5
            )
    
    def test_PELITA_full_pipeline(self, sample_PELITA_data):
        """Test full PELITA pipeline from raw data to features"""
        engineer = PELITAFeatureEngineer(verbose=False)
        
        # Engineer features
        df = engineer.engineer(sample_PELITA_data)
        
        # Verify data integrity
        assert len(df) > 0
        assert df.isnull().sum().sum() == 0
        
        # Verify PELITA-specific features
        assert 'VA_normalized' in df.columns
        assert 'MCB_Tripped' in df.columns
        
        # Verify VA normalization
        assert df['VA_normalized'].min() >= 0
        assert df['VA_normalized'].max() <= 1
    
    def test_universal_pipeline_both_datasets(self, sample_uci_data, sample_PELITA_data):
        """Test universal engineer with both datasets"""
        engineer_uci = UniversalFeatureEngineer(verbose=False)
        df_uci = engineer_uci.engineer(sample_uci_data)
        
        engineer_PELITA = UniversalFeatureEngineer(verbose=False)
        df_PELITA = engineer_PELITA.engineer(sample_PELITA_data)
        
        # Both should succeed
        assert len(df_uci) > 0
        assert len(df_PELITA) > 0
        
        # Should have different features
        assert 'VA_normalized' not in df_uci.columns
        assert 'VA_normalized' in df_PELITA.columns
    
    def test_feature_consistency_across_runs(self, sample_uci_data):
        """Test that feature engineering is deterministic"""
        engineer1 = UCIFeatureEngineer(verbose=False)
        df1 = engineer1.engineer(sample_uci_data.copy())
        
        engineer2 = UCIFeatureEngineer(verbose=False)
        df2 = engineer2.engineer(sample_uci_data.copy())
        
        # Results should be identical
        pd.testing.assert_frame_equal(df1, df2)
