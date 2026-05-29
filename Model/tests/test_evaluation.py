"""
Unit tests for evaluation module.

Tests cover:
- ModelEvaluator metrics calculation
- Plotting functions
- ComparativeAnalyzer comparison logic
- Report generation
"""

import pytest
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression

from modules.evaluation import ModelEvaluator, ComparativeAnalyzer


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def sample_predictions():
    """Generate sample predictions for testing."""
    np.random.seed(42)
    y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y_pred = np.array([1.1, 1.9, 3.2, 3.8, 5.1])
    return y_true, y_pred


@pytest.fixture
def larger_predictions():
    """Generate larger dataset for testing."""
    np.random.seed(42)
    X, y = make_regression(n_samples=100, n_features=5, noise=10, random_state=42)
    y_pred = y + np.random.normal(0, 5, size=len(y))
    return y, y_pred


# ============================================================================
# Test ModelEvaluator
# ============================================================================

def test_evaluator_initialization():
    """Test ModelEvaluator initialization."""
    evaluator = ModelEvaluator(model_name='TestModel')
    
    assert evaluator.model_name == 'TestModel'
    assert evaluator.metrics == {}
    assert evaluator.predictions is None
    assert evaluator.actuals is None


def test_evaluate_metrics(sample_predictions):
    """Test metric calculation."""
    y_true, y_pred = sample_predictions
    
    evaluator = ModelEvaluator()
    metrics = evaluator.evaluate(y_true, y_pred)
    
    # Check all metrics are present
    assert 'MAE' in metrics
    assert 'RMSE' in metrics
    assert 'R2' in metrics
    
    # Check metrics are reasonable
    assert metrics['MAE'] > 0
    assert metrics['RMSE'] > 0
    assert 0 <= metrics['R2'] <= 1


def test_evaluate_stores_predictions(sample_predictions):
    """Test that evaluate stores predictions."""
    y_true, y_pred = sample_predictions
    
    evaluator = ModelEvaluator()
    evaluator.evaluate(y_true, y_pred)
    
    assert evaluator.actuals is not None
    assert evaluator.predictions is not None
    np.testing.assert_array_equal(evaluator.actuals, y_true)
    np.testing.assert_array_equal(evaluator.predictions, y_pred)


def test_evaluate_perfect_predictions():
    """Test metrics with perfect predictions."""
    y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y_pred = y_true.copy()
    
    evaluator = ModelEvaluator()
    metrics = evaluator.evaluate(y_true, y_pred)
    
    # Perfect predictions should have MAE=0, RMSE=0, R²=1
    assert metrics['MAE'] == pytest.approx(0.0, abs=1e-10)
    assert metrics['RMSE'] == pytest.approx(0.0, abs=1e-10)
    assert metrics['R2'] == pytest.approx(1.0, abs=1e-10)


def test_plot_predictions_creates_figure(sample_predictions):
    """Test that plot_predictions creates a figure."""
    y_true, y_pred = sample_predictions
    
    evaluator = ModelEvaluator()
    evaluator.evaluate(y_true, y_pred)
    
    fig = evaluator.plot_predictions(show=False)
    
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_plot_predictions_without_evaluate():
    """Test plot_predictions fails without evaluate."""
    evaluator = ModelEvaluator()
    
    with pytest.raises(ValueError, match="No predictions available"):
        evaluator.plot_predictions()


def test_plot_predictions_with_explicit_data(sample_predictions):
    """Test plot_predictions with explicit data."""
    y_true, y_pred = sample_predictions
    
    evaluator = ModelEvaluator()
    fig = evaluator.plot_predictions(y_true, y_pred, show=False)
    
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_plot_residuals_creates_figure(sample_predictions):
    """Test that plot_residuals creates a figure."""
    y_true, y_pred = sample_predictions
    
    evaluator = ModelEvaluator()
    evaluator.evaluate(y_true, y_pred)
    
    fig = evaluator.plot_residuals(show=False)
    
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_analyze_residuals(sample_predictions):
    """Test residual analysis."""
    y_true, y_pred = sample_predictions
    
    evaluator = ModelEvaluator()
    evaluator.evaluate(y_true, y_pred)
    
    stats = evaluator.analyze_residuals()
    
    # Check all stats are present
    assert 'mean' in stats
    assert 'std' in stats
    assert 'min' in stats
    assert 'max' in stats
    assert 'median' in stats
    assert 'q25' in stats
    assert 'q75' in stats
    
    # Check stats are reasonable
    assert isinstance(stats['mean'], (int, float))
    assert stats['std'] >= 0


# ============================================================================
# Test ComparativeAnalyzer
# ============================================================================

def test_comparative_analyzer_initialization():
    """Test ComparativeAnalyzer initialization."""
    analyzer = ComparativeAnalyzer()
    
    assert analyzer.results == {}


def test_add_model_results():
    """Test adding model results."""
    analyzer = ComparativeAnalyzer()
    
    metrics = {'MAE': 0.5, 'RMSE': 0.7, 'R2': 0.85}
    analyzer.add_model_results('Model1', metrics)
    
    assert 'Model1' in analyzer.results
    assert analyzer.results['Model1']['metrics'] == metrics


def test_add_multiple_models():
    """Test adding multiple model results."""
    analyzer = ComparativeAnalyzer()
    
    analyzer.add_model_results('Model1', {'MAE': 0.5, 'RMSE': 0.7, 'R2': 0.85})
    analyzer.add_model_results('Model2', {'MAE': 0.4, 'RMSE': 0.6, 'R2': 0.90})
    
    assert len(analyzer.results) == 2
    assert 'Model1' in analyzer.results
    assert 'Model2' in analyzer.results


def test_generate_comparison_table():
    """Test comparison table generation."""
    analyzer = ComparativeAnalyzer()
    
    analyzer.add_model_results('Model1', {'MAE': 0.5, 'RMSE': 0.7, 'R2': 0.85})
    analyzer.add_model_results('Model2', {'MAE': 0.4, 'RMSE': 0.6, 'R2': 0.90})
    
    df = analyzer.generate_comparison_table()
    
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 2
    assert 'Model' in df.columns
    assert 'MAE' in df.columns
    assert 'RMSE' in df.columns
    assert 'R²' in df.columns


def test_comparison_table_sorted_by_r2():
    """Test that comparison table is sorted by R² descending."""
    analyzer = ComparativeAnalyzer()
    
    analyzer.add_model_results('Model1', {'MAE': 0.5, 'RMSE': 0.7, 'R2': 0.85})
    analyzer.add_model_results('Model2', {'MAE': 0.4, 'RMSE': 0.6, 'R2': 0.90})
    analyzer.add_model_results('Model3', {'MAE': 0.6, 'RMSE': 0.8, 'R2': 0.80})
    
    df = analyzer.generate_comparison_table()
    
    # Should be sorted by R² descending
    assert df.iloc[0]['Model'] == 'Model2'  # Highest R²
    assert df.iloc[1]['Model'] == 'Model1'
    assert df.iloc[2]['Model'] == 'Model3'  # Lowest R²


def test_generate_comparison_table_without_results():
    """Test comparison table fails without results."""
    analyzer = ComparativeAnalyzer()
    
    with pytest.raises(ValueError, match="No results added"):
        analyzer.generate_comparison_table()


def test_plot_metric_comparison_creates_figure():
    """Test metric comparison plot."""
    analyzer = ComparativeAnalyzer()
    
    analyzer.add_model_results('Model1', {'MAE': 0.5, 'RMSE': 0.7, 'R2': 0.85})
    analyzer.add_model_results('Model2', {'MAE': 0.4, 'RMSE': 0.6, 'R2': 0.90})
    
    fig = analyzer.plot_metric_comparison(show=False)
    
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_plot_feature_importance_comparison():
    """Test feature importance comparison plot."""
    analyzer = ComparativeAnalyzer()
    
    fi1 = pd.DataFrame({
        'Feature': ['f1', 'f2', 'f3'],
        'Importance': [0.5, 0.3, 0.2]
    })
    
    fi2 = pd.DataFrame({
        'Feature': ['f1', 'f2', 'f3'],
        'Importance': [0.4, 0.4, 0.2]
    })
    
    analyzer.add_model_results('Model1', {'MAE': 0.5, 'RMSE': 0.7, 'R2': 0.85}, feature_importance=fi1)
    analyzer.add_model_results('Model2', {'MAE': 0.4, 'RMSE': 0.6, 'R2': 0.90}, feature_importance=fi2)
    
    fig = analyzer.plot_feature_importance_comparison(top_n=3, show=False)
    
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_plot_feature_importance_without_data():
    """Test feature importance plot fails without data."""
    analyzer = ComparativeAnalyzer()
    
    analyzer.add_model_results('Model1', {'MAE': 0.5, 'RMSE': 0.7, 'R2': 0.85})
    
    with pytest.raises(ValueError, match="No feature importance data"):
        analyzer.plot_feature_importance_comparison()


def test_get_best_model_by_r2():
    """Test getting best model by R²."""
    analyzer = ComparativeAnalyzer()
    
    analyzer.add_model_results('Model1', {'MAE': 0.5, 'RMSE': 0.7, 'R2': 0.85})
    analyzer.add_model_results('Model2', {'MAE': 0.4, 'RMSE': 0.6, 'R2': 0.90})
    analyzer.add_model_results('Model3', {'MAE': 0.6, 'RMSE': 0.8, 'R2': 0.80})
    
    best = analyzer.get_best_model('R²')
    
    assert best == 'Model2'


def test_get_best_model_by_mae():
    """Test getting best model by MAE (lowest is best)."""
    analyzer = ComparativeAnalyzer()
    
    analyzer.add_model_results('Model1', {'MAE': 0.5, 'RMSE': 0.7, 'R2': 0.85})
    analyzer.add_model_results('Model2', {'MAE': 0.4, 'RMSE': 0.6, 'R2': 0.90})
    analyzer.add_model_results('Model3', {'MAE': 0.6, 'RMSE': 0.8, 'R2': 0.80})
    
    best = analyzer.get_best_model('MAE')
    
    assert best == 'Model2'


def test_get_best_model_by_rmse():
    """Test getting best model by RMSE (lowest is best)."""
    analyzer = ComparativeAnalyzer()
    
    analyzer.add_model_results('Model1', {'MAE': 0.5, 'RMSE': 0.7, 'R2': 0.85})
    analyzer.add_model_results('Model2', {'MAE': 0.4, 'RMSE': 0.6, 'R2': 0.90})
    analyzer.add_model_results('Model3', {'MAE': 0.6, 'RMSE': 0.8, 'R2': 0.80})
    
    best = analyzer.get_best_model('RMSE')
    
    assert best == 'Model2'


def test_generate_summary_report():
    """Test summary report generation."""
    analyzer = ComparativeAnalyzer()
    
    analyzer.add_model_results('Model1', {'MAE': 0.5, 'RMSE': 0.7, 'R2': 0.85})
    analyzer.add_model_results('Model2', {'MAE': 0.4, 'RMSE': 0.6, 'R2': 0.90})
    
    report = analyzer.generate_summary_report()
    
    assert isinstance(report, str)
    assert 'MODEL COMPARISON SUMMARY' in report
    assert 'Model1' in report
    assert 'Model2' in report
    assert 'BEST MODELS' in report


def test_add_model_with_training_time():
    """Test adding model with training time."""
    analyzer = ComparativeAnalyzer()
    
    analyzer.add_model_results(
        'Model1',
        {'MAE': 0.5, 'RMSE': 0.7, 'R2': 0.85},
        training_time=120.5
    )
    
    df = analyzer.generate_comparison_table()
    
    assert 'Training Time (s)' in df.columns
    assert df.iloc[0]['Training Time (s)'] == 120.5


def test_add_model_with_predictions():
    """Test adding model with predictions."""
    analyzer = ComparativeAnalyzer()
    
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([1.1, 1.9, 3.1])
    
    analyzer.add_model_results(
        'Model1',
        {'MAE': 0.5, 'RMSE': 0.7, 'R2': 0.85},
        predictions=(y_true, y_pred)
    )
    
    assert analyzer.results['Model1']['predictions'] is not None
    assert len(analyzer.results['Model1']['predictions']) == 2
