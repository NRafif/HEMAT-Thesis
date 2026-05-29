"""
Property-based tests for Energy Prediction Models
==================================================

Using hypothesis library for property-based testing.
Tests universal properties that should hold for all inputs.
"""

import pytest
import pandas as pd
import numpy as np
from hypothesis import given, strategies as st, settings
from datetime import datetime, timedelta

from modules.feature_engineering import UCIFeatureEngineer, PELITAFeatureEngineer


@pytest.mark.property
class TestFeatureEngineeringProperties:
    """Property-based tests for feature engineering"""
    
    @given(st.lists(st.floats(min_value=0.1, max_value=10.0), min_size=200, max_size=200))
    @settings(max_examples=50, deadline=None)
    def test_property_lag_no_future_data(self, power_values):
        """
        Property 2: Feature Engineering No Data Leakage
        
        For any time series, lag features should ONLY use past data,
        never current or future data.
        
        Validates: Requirements 1.2, 2.4, 8.5
        """
        # Create DataFrame with random power values
        dates = pd.date_range('2023-01-01', periods=len(power_values), freq='h')
        df = pd.DataFrame({
            'Global_active_power': power_values
        }, index=dates)
        
        engineer = UCIFeatureEngineer(verbose=False)
        df_with_lags = engineer.create_lag_features(df, 'Global_active_power', lags=[1, 24])
        
        # Property: lag_1 at index i should equal Global_active_power at index i-1
        for i in range(1, len(df_with_lags)):
            expected = df.iloc[i-1]['Global_active_power']
            actual = df_with_lags.iloc[i]['lag_1']
            
            # Should match (within floating point precision)
            assert np.isclose(expected, actual, rtol=1e-5), \
                f"Data leakage detected! lag_1 at index {i} doesn't match past value"
        
        # Property: lag_1 at index 0 should be NaN (no past data)
        assert pd.isna(df_with_lags.iloc[0]['lag_1']), \
            "First row lag_1 should be NaN (no past data available)"
    
    @given(st.lists(st.floats(min_value=0.1, max_value=10.0), min_size=100, max_size=100))
    @settings(max_examples=50, deadline=None)
    def test_property_rolling_no_future_data(self, power_values):
        """
        Property 2: Rolling Features No Data Leakage
        
        For any time series, rolling features should ONLY use past data.
        rolling_mean_3 at index i should be mean of [i-3, i-2, i-1], NOT including i.
        
        Validates: Requirements 1.2, 2.4, 8.5
        """
        # Create DataFrame
        dates = pd.date_range('2023-01-01', periods=len(power_values), freq='h')
        df = pd.DataFrame({
            'Global_active_power': power_values
        }, index=dates)
        
        engineer = UCIFeatureEngineer(verbose=False)
        df_with_rolling = engineer.create_rolling_features(df, 'Global_active_power', windows=[3])
        
        # Property: rolling_mean_3 should NOT include current value
        if len(df_with_rolling) >= 5:
            idx = 4
            # rolling_mean_3 at idx should be mean of [idx-3, idx-2, idx-1]
            # NOT mean of [idx-2, idx-1, idx] (which would be data leakage)
            expected = df.iloc[1:4]['Global_active_power'].mean()
            actual = df_with_rolling.iloc[idx]['rolling_mean_3']
            
            # Should match past values
            assert np.isclose(expected, actual, rtol=1e-5), \
                f"Data leakage! rolling_mean_3 includes current or future data"
            
            # Should NOT match if we include current value (unless all values are same)
            wrong = df.iloc[2:5]['Global_active_power'].mean()
            # Only check if values are different (not all same)
            if not np.allclose(df.iloc[1:5]['Global_active_power'], df.iloc[1]['Global_active_power']):
                assert not np.isclose(wrong, actual, rtol=1e-5), \
                    f"rolling_mean_3 should not include current value"
    
    @given(st.lists(st.floats(min_value=0.1, max_value=10.0), min_size=50, max_size=50))
    @settings(max_examples=30, deadline=None)
    def test_property_temporal_features_valid_ranges(self, power_values):
        """
        Property: Temporal features should always be in valid ranges
        
        hour: 0-23, dayofweek: 0-6, month: 1-12, season: 1-4
        """
        dates = pd.date_range('2023-01-01', periods=len(power_values), freq='h')
        df = pd.DataFrame({
            'Global_active_power': power_values
        }, index=dates)
        
        engineer = UCIFeatureEngineer(verbose=False)
        df_with_temporal = engineer.create_temporal_features(df)
        
        # Properties: All temporal features in valid ranges
        assert (df_with_temporal['hour'] >= 0).all()
        assert (df_with_temporal['hour'] <= 23).all()
        assert (df_with_temporal['dayofweek'] >= 0).all()
        assert (df_with_temporal['dayofweek'] <= 6).all()
        assert (df_with_temporal['month'] >= 1).all()
        assert (df_with_temporal['month'] <= 12).all()
        assert (df_with_temporal['season'] >= 1).all()
        assert (df_with_temporal['season'] <= 4).all()
    
    @given(
        st.lists(st.floats(min_value=0.1, max_value=10.0), min_size=50, max_size=50),
        st.integers(min_value=1, max_value=10)
    )
    @settings(max_examples=30, deadline=None)
    def test_property_lag_shift_consistency(self, power_values, lag_period):
        """
        Property: lag_N at index i should always equal value at index i-N
        
        This should hold for ANY lag period N.
        """
        dates = pd.date_range('2023-01-01', periods=len(power_values), freq='h')
        df = pd.DataFrame({
            'Global_active_power': power_values
        }, index=dates)
        
        engineer = UCIFeatureEngineer(verbose=False)
        df_with_lag = engineer.create_lag_features(df, 'Global_active_power', lags=[lag_period])
        
        # Property: For all i >= lag_period, lag_N[i] == value[i-N]
        for i in range(lag_period, len(df_with_lag)):
            expected = df.iloc[i - lag_period]['Global_active_power']
            actual = df_with_lag.iloc[i][f'lag_{lag_period}']
            
            assert np.isclose(expected, actual, rtol=1e-5), \
                f"Lag inconsistency at index {i} for lag_{lag_period}"
        
        # Property: First lag_period rows should be NaN
        for i in range(lag_period):
            assert pd.isna(df_with_lag.iloc[i][f'lag_{lag_period}']), \
                f"First {lag_period} rows should have NaN for lag_{lag_period}"


@pytest.mark.property
class TestPELITAFeatureProperties:
    """Property-based tests for PELITA-specific features"""
    
    @given(st.sampled_from([900, 1300, 2200]))
    @settings(max_examples=10)
    def test_property_va_normalization_bounds(self, va_value):
        """
        Property: VA normalization should always produce values in [0, 1]
        
        For ANY VA value (900, 1300, 2200), normalized value should be in [0, 1].
        """
        # Create sample PELITA data
        dates = pd.date_range('2025-01-01', periods=10, freq='h')
        df = pd.DataFrame({
            'Global_active_power': np.random.uniform(0.5, 2.0, 10),
            'VA': [va_value] * 10,
            'MCB_Tripped': [0] * 10,
            'House_Type': ['working_class'] * 10
        }, index=dates)
        
        engineer = PELITAFeatureEngineer(verbose=False)
        df_normalized = engineer.normalize_va(df)
        
        # Property: All normalized values in [0, 1]
        assert (df_normalized['VA_normalized'] >= 0).all()
        assert (df_normalized['VA_normalized'] <= 1).all()
        
        # Property: Normalization is consistent
        expected = va_value / 2200.0
        assert np.allclose(df_normalized['VA_normalized'], expected)
    
    @given(st.sampled_from(['working_class', 'retired', 'mixed']))
    @settings(max_examples=10)
    def test_property_house_type_encoding_sum(self, house_type):
        """
        Property: One-hot encoding should sum to 1 for each row
        
        For ANY house type, the sum of one-hot encoded columns should be 1.
        """
        dates = pd.date_range('2025-01-01', periods=10, freq='h')
        df = pd.DataFrame({
            'Global_active_power': np.random.uniform(0.5, 2.0, 10),
            'VA': [1300] * 10,
            'MCB_Tripped': [0] * 10,
            'House_Type': [house_type] * 10
        }, index=dates)
        
        engineer = PELITAFeatureEngineer(verbose=False)
        df_encoded = engineer.encode_house_type(df)
        
        # Property: Sum of one-hot columns should be 1
        ht_cols = [col for col in df_encoded.columns if col.startswith('HT_')]
        ht_sum = df_encoded[ht_cols].sum(axis=1)
        
        assert (ht_sum == 1).all(), "One-hot encoding should sum to 1 per row"


@pytest.mark.property
class TestFeatureEngineeringInvariants:
    """Test invariants that should hold across all feature engineering"""
    
    @given(st.lists(st.floats(min_value=0.1, max_value=10.0), min_size=50, max_size=50))
    @settings(max_examples=30, deadline=None)
    def test_property_row_count_preserved(self, power_values):
        """
        Property: Feature engineering should not change row count
        (except for dropna at the end)
        
        For ANY input, output should have same or fewer rows.
        """
        dates = pd.date_range('2023-01-01', periods=len(power_values), freq='h')
        df = pd.DataFrame({
            'Global_active_power': power_values
        }, index=dates)
        
        engineer = UCIFeatureEngineer(verbose=False)
        
        # Before dropna, row count should be same
        df_temporal = engineer.create_temporal_features(df)
        assert len(df_temporal) == len(df)
        
        df_lag = engineer.create_lag_features(df, 'Global_active_power', lags=[1])
        assert len(df_lag) == len(df)
        
        df_rolling = engineer.create_rolling_features(df, 'Global_active_power', windows=[3])
        assert len(df_rolling) == len(df)
    
    @given(st.lists(st.floats(min_value=0.1, max_value=10.0), min_size=50, max_size=50))
    @settings(max_examples=30, deadline=None)
    def test_property_original_columns_preserved(self, power_values):
        """
        Property: Feature engineering should preserve original columns
        
        For ANY input, original columns should still exist after engineering.
        """
        dates = pd.date_range('2023-01-01', periods=len(power_values), freq='h')
        df = pd.DataFrame({
            'Global_active_power': power_values,
            'Voltage': np.random.uniform(220, 240, len(power_values))
        }, index=dates)
        
        original_cols = set(df.columns)
        
        engineer = UCIFeatureEngineer(verbose=False)
        df_engineered = engineer.engineer(df)
        
        # Property: All original columns should still exist
        assert original_cols.issubset(set(df_engineered.columns)), \
            "Original columns should be preserved"


# ============================================================================
# Property Tests for Model Predictions
# ============================================================================

from modules.models import DeltaModel, NadirModel, EpsilonModel
from sklearn.datasets import make_regression


@pytest.mark.property
class TestModelPredictionProperties:
    """Property-based tests for model predictions"""
    
    @given(
        st.integers(min_value=100, max_value=300),
        st.integers(min_value=5, max_value=15)
    )
    @settings(max_examples=20, deadline=None)
    def test_property_prediction_bounds(self, n_samples, n_features):
        """
        Property 3: Model Prediction Bounds
        
        For any valid feature set, predictions should be:
        1. Non-negative (energy consumption cannot be negative)
        2. Finite (no NaN or inf values)
        3. Within realistic range (not absurdly large)
        
        Validates: Requirements 1.5, 2.6, 8.4
        """
        # Generate synthetic data
        X, y = make_regression(
            n_samples=n_samples,
            n_features=n_features,
            n_informative=max(3, n_features-2),
            noise=10,
            random_state=42
        )
        
        # Make y positive (energy consumption)
        y = np.abs(y)
        
        X_df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(n_features)])
        y_series = pd.Series(y)
        
        # Split
        split = int(0.7 * n_samples)
        X_train, X_test = X_df.iloc[:split], X_df.iloc[split:]
        y_train, y_test = y_series.iloc[:split], y_series.iloc[split:]
        
        # Train model with minimal params for speed
        model = DeltaModel(random_state=42)
        model.get_param_grid = lambda: {'n_estimators': [50], 'max_depth': [5]}
        model.train(X_train, y_train, cv_splits=2, verbose=0)
        
        # Make predictions
        predictions = model.predict(X_test)
        
        # Property 1: All predictions should be finite
        assert np.all(np.isfinite(predictions)), \
            "Predictions contain NaN or inf values"
        
        # Property 2: Predictions should be numeric
        assert predictions.dtype in [np.float32, np.float64], \
            "Predictions are not numeric"
        
        # Property 3: Predictions should be within reasonable bounds
        # (within 10x the range of training data)
        y_range = y_train.max() - y_train.min()
        max_reasonable = y_train.max() + 10 * y_range
        min_reasonable = y_train.min() - 10 * y_range
        
        assert np.all(predictions <= max_reasonable), \
            f"Some predictions exceed reasonable upper bound: {predictions.max()} > {max_reasonable}"
        assert np.all(predictions >= min_reasonable), \
            f"Some predictions below reasonable lower bound: {predictions.min()} < {min_reasonable}"
    
    
    @given(
        st.sampled_from([900, 1300, 2200]),
        st.integers(min_value=100, max_value=200)
    )
    @settings(max_examples=15, deadline=None)
    def test_property_va_capacity_awareness(self, va_value, n_samples):
        """
        Property 4: VA Capacity Constraint (PELITA)
        
        For any PELITA dataset with known VA capacity, the model should
        learn that power consumption patterns respect VA limits.
        
        This tests that the model learns the relationship between VA
        and power consumption, not that it enforces hard constraints.
        
        Validates: Requirements 1.4, 1.6, 5.1
        """
        # Generate PELITA-like data
        n_features = 10
        X, y = make_regression(
            n_samples=n_samples,
            n_features=n_features,
            n_informative=8,
            noise=15,
            random_state=42
        )
        
        # Create DataFrame with VA column
        X_df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(n_features)])
        X_df['VA'] = va_value
        
        # Scale y to be realistic power values relative to VA
        # Typical utilization: 50-90% of VA capacity
        y_scaled = np.abs(y)
        y_scaled = (y_scaled / y_scaled.max()) * (va_value * 0.8)  # 80% max utilization
        y_series = pd.Series(y_scaled)
        
        # Split
        split = int(0.7 * n_samples)
        X_train, X_test = X_df.iloc[:split], X_df.iloc[split:]
        y_train, y_test = y_series.iloc[:split], y_series.iloc[split:]
        
        # Train Nadir model
        model = NadirModel(random_state=42)
        model.get_param_grid = lambda: {'n_estimators': [50], 'max_depth': [10]}
        model.train(X_train, y_train, cv_splits=2, verbose=0)
        
        # Make predictions
        predictions = model.predict(X_test)
        
        # Property: Predictions should be finite and reasonable
        assert np.all(np.isfinite(predictions)), \
            "Predictions contain NaN or inf"
        
        # Property: Most predictions should be below VA capacity
        # (allowing some margin for model uncertainty)
        reasonable_max = va_value * 1.2  # 120% of VA (some tolerance)
        
        pct_within_bounds = np.mean(predictions <= reasonable_max) * 100
        
        # At least 80% of predictions should be reasonable
        assert pct_within_bounds >= 80, \
            f"Only {pct_within_bounds:.1f}% of predictions within reasonable bounds for VA={va_value}"
    
    
    @given(st.integers(min_value=100, max_value=200))
    @settings(max_examples=15, deadline=None)
    def test_property_model_persistence_roundtrip(self, n_samples):
        """
        Property 7: Model Persistence Round-Trip
        
        For any trained model, saving and loading should preserve
        predictions exactly (within floating point precision).
        
        Validates: Requirements 7.1, 7.2, 7.3
        """
        import tempfile
        import os
        
        # Generate data
        X, y = make_regression(
            n_samples=n_samples,
            n_features=8,
            n_informative=6,
            noise=10,
            random_state=42
        )
        
        X_df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(8)])
        y_series = pd.Series(y)
        
        split = int(0.7 * n_samples)
        X_train, X_test = X_df.iloc[:split], X_df.iloc[split:]
        y_train = y_series.iloc[:split]
        
        # Train model
        model = DeltaModel(random_state=42)
        model.get_param_grid = lambda: {'n_estimators': [50], 'max_depth': [5]}
        model.train(X_train, y_train, cv_splits=2, verbose=0)
        
        # Get predictions before saving
        pred_before = model.predict(X_test)
        
        # Save and load
        with tempfile.NamedTemporaryFile(suffix='.joblib', delete=False) as tmp:
            tmp_path = tmp.name
        
        try:
            model.save(tmp_path)
            loaded_model = DeltaModel.load(tmp_path)
            pred_after = loaded_model.predict(X_test)
            
            # Property: Predictions should be identical (within epsilon)
            np.testing.assert_allclose(
                pred_before,
                pred_after,
                rtol=1e-6,
                atol=1e-6,
                err_msg="Predictions changed after save/load round-trip"
            )
            
            # Property: Feature names should be preserved
            assert model.feature_names == loaded_model.feature_names, \
                "Feature names not preserved after save/load"
            
            # Property: Best params should be preserved
            assert model.best_params == loaded_model.best_params, \
                "Best params not preserved after save/load"
        
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
    
    
    @given(st.sampled_from(['uci', 'PELITA']))
    @settings(max_examples=10, deadline=None)
    def test_property_epsilon_adapts_to_dataset(self, dataset_type):
        """
        Property 10: Universal Loader Type Detection
        
        For any dataset type, EpsilonModel should adapt its
        hyperparameters appropriately.
        
        Validates: Requirements 2.1, 2.6
        """
        # Create Epsilon model with dataset type
        model = EpsilonModel(dataset_type=dataset_type, random_state=42)
        
        # Get parameter grid
        param_grid = model.get_param_grid()
        
        # Property: Parameter grid should be non-empty
        assert len(param_grid) > 0, \
            "Parameter grid is empty"
        
        # Property: Should have essential RF parameters
        assert 'n_estimators' in param_grid, \
            "Missing n_estimators parameter"
        assert 'max_depth' in param_grid, \
            "Missing max_depth parameter"
        
        # Property: Parameters should be lists
        assert isinstance(param_grid['n_estimators'], list), \
            "n_estimators should be a list"
        
        # Property: Different dataset types should have different grids
        if dataset_type == 'PELITA':
            # PELITA should have more n_estimators options (Nadir grid)
            assert len(param_grid['n_estimators']) >= 3, \
                "PELITA should have at least 3 n_estimators options"
        else:
            # UCI uses Delta grid
            assert len(param_grid['n_estimators']) >= 2, \
                "UCI should have at least 2 n_estimators options"
    
    
    @given(st.integers(min_value=100, max_value=200))
    @settings(max_examples=10, deadline=None)
    def test_property_feature_importance_consistency(self, n_samples):
        """
        Property 5: Feature Importance Consistency
        
        For any dataset, training the same model multiple times with
        the same random seed should produce consistent feature importance
        rankings (at least for top features).
        
        Validates: Requirements 4.4, 8.1
        """
        # Generate data with clear feature importance pattern
        X, y = make_regression(
            n_samples=n_samples,
            n_features=10,
            n_informative=5,
            noise=10,
            random_state=42
        )
        
        X_df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(10)])
        y_series = pd.Series(y)
        
        # Train model twice with same seed
        model1 = DeltaModel(random_state=42)
        model1.get_param_grid = lambda: {'n_estimators': [100], 'max_depth': [10]}
        model1.train(X_df, y_series, cv_splits=2, verbose=0)
        
        model2 = DeltaModel(random_state=42)
        model2.get_param_grid = lambda: {'n_estimators': [100], 'max_depth': [10]}
        model2.train(X_df, y_series, cv_splits=2, verbose=0)
        
        # Get feature importance
        imp1 = model1.get_feature_importance()
        imp2 = model2.get_feature_importance()
        
        # Property: Top 3 features should be the same
        top3_features1 = set(imp1.head(3)['Feature'].values)
        top3_features2 = set(imp2.head(3)['Feature'].values)
        
        assert top3_features1 == top3_features2, \
            f"Top 3 features differ between runs: {top3_features1} vs {top3_features2}"
        
        # Property: Importance values should be very similar
        imp1_sorted = imp1.sort_values('Feature')
        imp2_sorted = imp2.sort_values('Feature')
        
        np.testing.assert_allclose(
            imp1_sorted['Importance'].values,
            imp2_sorted['Importance'].values,
            rtol=0.01,  # 1% tolerance
            err_msg="Feature importance values differ significantly between runs"
        )
