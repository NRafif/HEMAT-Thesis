"""
Unit tests for model training module.

Tests cover:
- Model initialization
- Training with synthetic data
- Prediction output validation
- Feature importance extraction
- Model persistence (save/load)
"""

import pytest
import numpy as np
import pandas as pd
import tempfile
import os
from sklearn.datasets import make_regression

from modules.models import (
    EnergyPredictionModel,
    DeltaModel,
    NadirModel,
    EpsilonModel
)


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def synthetic_data():
    """Generate synthetic regression data for testing."""
    X, y = make_regression(
        n_samples=200,
        n_features=10,
        n_informative=8,
        noise=10,
        random_state=42
    )
    
    feature_names = [f'feature_{i}' for i in range(10)]
    X_df = pd.DataFrame(X, columns=feature_names)
    y_series = pd.Series(y, name='target')
    
    # Split into train/test
    split_idx = 150
    X_train = X_df.iloc[:split_idx]
    X_test = X_df.iloc[split_idx:]
    y_train = y_series.iloc[:split_idx]
    y_test = y_series.iloc[split_idx:]
    
    return X_train, X_test, y_train, y_test


@pytest.fixture
def PELITA_like_data():
    """Generate PELITA-like data with VA column."""
    X, y = make_regression(
        n_samples=200,
        n_features=12,
        n_informative=10,
        noise=15,
        random_state=42
    )
    
    feature_names = [f'feature_{i}' for i in range(11)] + ['VA']
    X_df = pd.DataFrame(X, columns=feature_names)
    
    # Add realistic VA values
    X_df['VA'] = np.random.choice([900, 1300, 2200], size=len(X_df))
    
    y_series = pd.Series(y, name='power')
    
    # Split
    split_idx = 150
    X_train = X_df.iloc[:split_idx]
    X_test = X_df.iloc[split_idx:]
    y_train = y_series.iloc[:split_idx]
    y_test = y_series.iloc[split_idx:]
    
    return X_train, X_test, y_train, y_test


# ============================================================================
# Test Model Initialization
# ============================================================================

def test_delta_model_initialization():
    """Test DeltaModel initialization."""
    model = DeltaModel(random_state=42)
    
    assert model.model_type == 'rf'
    assert model.random_state == 42
    assert model.model is None
    assert model.best_params is None


def test_nadir_model_initialization():
    """Test NadirModel initialization."""
    model = NadirModel(random_state=123)
    
    assert model.model_type == 'rf'
    assert model.random_state == 123
    assert model.model is None


def test_epsilon_model_initialization():
    """Test EpsilonModel initialization with dataset type."""
    model_uci = EpsilonModel(dataset_type='uci', random_state=42)
    model_PELITA = EpsilonModel(dataset_type='PELITA', random_state=42)
    
    assert model_uci.dataset_type == 'uci'
    assert model_PELITA.dataset_type == 'PELITA'


def test_invalid_model_type():
    """Test that invalid model type raises error during training."""
    model = DeltaModel(model_type='invalid')
    X_train = pd.DataFrame(np.random.rand(50, 5))
    y_train = pd.Series(np.random.rand(50))
    
    with pytest.raises(ValueError, match="Unsupported model_type"):
        model.train(X_train, y_train, cv_splits=2)


# ============================================================================
# Test Parameter Grids
# ============================================================================

def test_delta_param_grid():
    """Test DeltaModel parameter grid structure."""
    model = DeltaModel()
    param_grid = model.get_param_grid()
    
    assert 'n_estimators' in param_grid
    assert 'max_depth' in param_grid
    assert 'min_samples_split' in param_grid
    assert isinstance(param_grid['n_estimators'], list)


def test_nadir_param_grid():
    """Test NadirModel has different parameters than Delta."""
    delta = DeltaModel()
    nadir = NadirModel()
    
    delta_grid = delta.get_param_grid()
    nadir_grid = nadir.get_param_grid()
    
    # Nadir should have more n_estimators options
    assert len(nadir_grid['n_estimators']) >= len(delta_grid['n_estimators'])


def test_epsilon_adaptive_param_grid():
    """Test EpsilonModel adapts parameter grid based on dataset type."""
    epsilon_uci = EpsilonModel(dataset_type='uci')
    epsilon_PELITA = EpsilonModel(dataset_type='PELITA')
    
    delta = DeltaModel()
    nadir = NadirModel()
    
    # Epsilon should use Delta's grid for UCI
    assert epsilon_uci.get_param_grid() == delta.get_param_grid()
    
    # Epsilon should use Nadir's grid for PELITA
    assert epsilon_PELITA.get_param_grid() == nadir.get_param_grid()


# ============================================================================
# Test Model Training
# ============================================================================

def test_delta_model_training(synthetic_data):
    """Test DeltaModel can train on synthetic data."""
    X_train, X_test, y_train, y_test = synthetic_data
    
    model = DeltaModel(random_state=42)
    
    # Use small param grid for faster testing
    original_get_param_grid = model.get_param_grid
    model.get_param_grid = lambda: {'n_estimators': [50], 'max_depth': [5]}
    
    model.train(X_train, y_train, cv_splits=2, verbose=0)
    
    assert model.model is not None
    assert model.best_params is not None
    assert model.feature_names is not None
    assert len(model.feature_names) == X_train.shape[1]


def test_nadir_model_training(PELITA_like_data):
    """Test NadirModel can train on PELITA-like data."""
    X_train, X_test, y_train, y_test = PELITA_like_data
    
    model = NadirModel(random_state=42)
    
    # Small param grid for speed
    model.get_param_grid = lambda: {'n_estimators': [50], 'max_depth': [10]}
    
    model.train(X_train, y_train, cv_splits=2, verbose=0)
    
    assert model.model is not None
    assert 'VA' in model.feature_names


def test_epsilon_model_training(synthetic_data):
    """Test EpsilonModel can train with adaptive parameters."""
    X_train, X_test, y_train, y_test = synthetic_data
    
    model = EpsilonModel(dataset_type='uci', random_state=42)
    model.get_param_grid = lambda: {'n_estimators': [50], 'max_depth': [5]}
    
    model.train(X_train, y_train, cv_splits=2, verbose=0)
    
    assert model.model is not None


def test_training_with_numpy_array(synthetic_data):
    """Test training works with numpy arrays (not just DataFrames)."""
    X_train, X_test, y_train, y_test = synthetic_data
    
    model = DeltaModel(random_state=42)
    model.get_param_grid = lambda: {'n_estimators': [50], 'max_depth': [5]}
    
    # Convert to numpy
    X_train_np = X_train.values
    y_train_np = y_train.values
    
    model.train(X_train_np, y_train_np, cv_splits=2, verbose=0)
    
    assert model.model is not None
    # Should generate default feature names
    assert all('feature_' in name for name in model.feature_names)


# ============================================================================
# Test Predictions
# ============================================================================

def test_prediction_output_shape(synthetic_data):
    """Test prediction output has correct shape."""
    X_train, X_test, y_train, y_test = synthetic_data
    
    model = DeltaModel(random_state=42)
    model.get_param_grid = lambda: {'n_estimators': [50], 'max_depth': [5]}
    model.train(X_train, y_train, cv_splits=2, verbose=0)
    
    predictions = model.predict(X_test)
    
    assert len(predictions) == len(X_test)
    assert isinstance(predictions, np.ndarray)


def test_prediction_without_training():
    """Test that prediction fails if model not trained."""
    model = DeltaModel()
    X_test = pd.DataFrame(np.random.rand(10, 5))
    
    with pytest.raises(ValueError, match="Model must be trained"):
        model.predict(X_test)


def test_predictions_are_numeric(synthetic_data):
    """Test that predictions are numeric values."""
    X_train, X_test, y_train, y_test = synthetic_data
    
    model = DeltaModel(random_state=42)
    model.get_param_grid = lambda: {'n_estimators': [50], 'max_depth': [5]}
    model.train(X_train, y_train, cv_splits=2, verbose=0)
    
    predictions = model.predict(X_test)
    
    assert np.all(np.isfinite(predictions))
    assert predictions.dtype in [np.float32, np.float64]


# ============================================================================
# Test Feature Importance
# ============================================================================

def test_feature_importance_extraction(synthetic_data):
    """Test feature importance can be extracted."""
    X_train, X_test, y_train, y_test = synthetic_data
    
    model = DeltaModel(random_state=42)
    model.get_param_grid = lambda: {'n_estimators': [50], 'max_depth': [5]}
    model.train(X_train, y_train, cv_splits=2, verbose=0)
    
    importance = model.get_feature_importance()
    
    assert isinstance(importance, pd.DataFrame)
    assert 'Feature' in importance.columns
    assert 'Importance' in importance.columns
    assert len(importance) == X_train.shape[1]


def test_feature_importance_sorted(synthetic_data):
    """Test feature importance is sorted in descending order."""
    X_train, X_test, y_train, y_test = synthetic_data
    
    model = DeltaModel(random_state=42)
    model.get_param_grid = lambda: {'n_estimators': [50], 'max_depth': [5]}
    model.train(X_train, y_train, cv_splits=2, verbose=0)
    
    importance = model.get_feature_importance()
    
    # Check descending order
    assert all(importance['Importance'].iloc[i] >= importance['Importance'].iloc[i+1]
               for i in range(len(importance)-1))


def test_feature_importance_top_n(synthetic_data):
    """Test getting top N features."""
    X_train, X_test, y_train, y_test = synthetic_data
    
    model = DeltaModel(random_state=42)
    model.get_param_grid = lambda: {'n_estimators': [50], 'max_depth': [5]}
    model.train(X_train, y_train, cv_splits=2, verbose=0)
    
    top_5 = model.get_feature_importance(top_n=5)
    
    assert len(top_5) == 5


def test_feature_importance_without_training():
    """Test feature importance fails if model not trained."""
    model = DeltaModel()
    
    with pytest.raises(ValueError, match="Model must be trained"):
        model.get_feature_importance()


# ============================================================================
# Test Model Persistence
# ============================================================================

def test_model_save_and_load(synthetic_data):
    """Test model can be saved and loaded."""
    X_train, X_test, y_train, y_test = synthetic_data
    
    # Train model
    model = DeltaModel(random_state=42)
    model.get_param_grid = lambda: {'n_estimators': [50], 'max_depth': [5]}
    model.train(X_train, y_train, cv_splits=2, verbose=0)
    
    # Get predictions before saving
    pred_before = model.predict(X_test)
    
    # Save to temp file
    with tempfile.NamedTemporaryFile(suffix='.joblib', delete=False) as tmp:
        tmp_path = tmp.name
    
    try:
        model.save(tmp_path)
        
        # Load model
        loaded_model = DeltaModel.load(tmp_path)
        
        # Get predictions after loading
        pred_after = loaded_model.predict(X_test)
        
        # Predictions should be identical
        np.testing.assert_array_almost_equal(pred_before, pred_after)
        
        # Metadata should be preserved
        assert loaded_model.model_type == model.model_type
        assert loaded_model.feature_names == model.feature_names
        assert loaded_model.best_params == model.best_params
        
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def test_save_without_training():
    """Test that saving fails if model not trained."""
    model = DeltaModel()
    
    with tempfile.NamedTemporaryFile(suffix='.joblib') as tmp:
        with pytest.raises(ValueError, match="Model must be trained"):
            model.save(tmp.name)


def test_load_preserves_model_class():
    """Test that loaded model preserves class-specific behavior."""
    X_train = pd.DataFrame(np.random.rand(100, 5))
    y_train = pd.Series(np.random.rand(100))
    
    # Train Nadir model
    model = NadirModel(random_state=42)
    model.get_param_grid = lambda: {'n_estimators': [50], 'max_depth': [5]}
    model.train(X_train, y_train, cv_splits=2, verbose=0)
    
    with tempfile.NamedTemporaryFile(suffix='.joblib', delete=False) as tmp:
        tmp_path = tmp.name
    
    try:
        model.save(tmp_path)
        loaded = NadirModel.load(tmp_path)
        
        # Should still be able to use Nadir-specific methods
        assert hasattr(loaded, 'train_stratified')
        
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


# ============================================================================
# Test Nadir-Specific Features
# ============================================================================

def test_nadir_stratified_training(PELITA_like_data):
    """Test NadirModel stratified training method."""
    X_train, X_test, y_train, y_test = PELITA_like_data
    
    model = NadirModel(random_state=42)
    model.get_param_grid = lambda: {'n_estimators': [50], 'max_depth': [10]}
    
    # Train with stratification
    model.train_stratified(X_train, y_train, va_column='VA', cv_splits=2)
    
    assert model.model is not None


def test_nadir_stratified_fallback_without_va():
    """Test stratified training falls back to regular training without VA."""
    X_train = pd.DataFrame(np.random.rand(100, 5))
    y_train = pd.Series(np.random.rand(100))
    
    model = NadirModel(random_state=42)
    model.get_param_grid = lambda: {'n_estimators': [50], 'max_depth': [5]}
    
    # Should not fail even without VA column
    model.train_stratified(X_train, y_train, va_column='VA', cv_splits=2)
    
    assert model.model is not None


# ============================================================================
# Test Epsilon-Specific Features
# ============================================================================

def test_epsilon_set_dataset_type():
    """Test EpsilonModel can change dataset type."""
    model = EpsilonModel(dataset_type='uci')
    
    assert model.dataset_type == 'uci'
    
    model.set_dataset_type('PELITA')
    
    assert model.dataset_type == 'PELITA'


def test_epsilon_param_grid_changes_with_dataset_type():
    """Test parameter grid changes when dataset type is changed."""
    model = EpsilonModel(dataset_type='uci')
    
    grid_uci = model.get_param_grid()
    
    model.set_dataset_type('PELITA')
    
    grid_PELITA = model.get_param_grid()
    
    # Grids should be different
    assert grid_uci != grid_PELITA
