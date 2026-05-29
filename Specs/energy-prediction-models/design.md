# Design Document: Energy Prediction Models Enhancement

## Overview

Sistem prediksi konsumsi energi rumah tangga dengan 3 pendekatan model berbeda menggunakan Random Forest Regressor. Arsitektur dirancang untuk mendukung:
1. **Model Delta**: Baseline untuk UCI dataset (existing, enhanced)
2. **Model Nadir**: PELITA-optimized dengan fitur Indonesia-specific
3. **Model Epsilon**: Universal model dengan auto-detection dataset type

Sistem ini menggunakan time series forecasting dengan feature engineering yang sophisticated dan comparative analysis framework untuk evaluasi komprehensif.

## Architecture

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Data Input Layer                          │
│  ┌──────────────┐              ┌──────────────┐            │
│  │ UCI Dataset  │              │PELITA Dataset│            │
│  │ (France)     │              │ (Indonesia)  │            │
│  └──────┬───────┘              └──────┬───────┘            │
└─────────┼──────────────────────────────┼───────────────────┘
          │                              │
          ▼                              ▼
┌─────────────────────────────────────────────────────────────┐
│              Data Processing Layer                           │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  Dataset Type Detector (Epsilon)                     │  │
│  │  - Column structure analysis                         │  │
│  │  - Auto-format detection                             │  │
│  └──────────────────────────────────────────────────────┘  │
│                                                              │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐    │
│  │ UCI Loader   │  │PELITA Loader │  │Universal     │    │
│  │ (Delta)      │  │ (Nadir)      │  │Loader        │    │
│  │              │  │              │  │(Epsilon)     │    │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘    │
└─────────┼──────────────────┼──────────────────┼───────────┘
          │                  │                  │
          ▼                  ▼                  ▼
┌─────────────────────────────────────────────────────────────┐
│           Feature Engineering Layer                          │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐    │
│  │ UCI Features │  │PELITA        │  │Universal     │    │
│  │ - Temporal   │  │Features      │  │Features      │    │
│  │ - Lag/Roll   │  │ - VA         │  │ - Adaptive   │    │
│  │ - Sub-meter  │  │ - MCB        │  │ - Optional   │    │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘    │
└─────────┼──────────────────┼──────────────────┼───────────┘
          │                  │                  │
          ▼                  ▼                  ▼
┌─────────────────────────────────────────────────────────────┐
│              Model Training Layer                            │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐    │
│  │ RF Delta     │  │ RF Nadir     │  │ RF Epsilon   │    │
│  │ + XGBoost    │  │ + XGBoost    │  │ + XGBoost    │    │
│  │ GridSearchCV │  │ GridSearchCV │  │ GridSearchCV │    │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘    │
└─────────┼──────────────────┼──────────────────┼───────────┘
          │                  │                  │
          ▼                  ▼                  ▼
┌─────────────────────────────────────────────────────────────┐
│           Evaluation & Comparison Layer                      │
│  ┌────────────────────────────────────────────────────┐    │
│  │  Comparative Analysis Engine                       │    │
│  │  - Performance metrics (MAE, RMSE, R², MAPE)      │    │
│  │  - Feature importance comparison                   │    │
│  │  - Training time analysis                          │    │
│  │  - Cross-dataset validation                        │    │
│  └────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────┘
          │
          ▼
┌─────────────────────────────────────────────────────────────┐
│         Recommendation & Output Layer                        │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐    │
│  │ UCI Recomm.  │  │PELITA Recomm.│  │Adaptive      │    │
│  │ - Time-based │  │ - MCB prevent│  │Recomm.       │    │
│  │ - Sub-meter  │  │ - VA-aware   │  │              │    │
│  └──────────────┘  └──────────────┘  └──────────────┘    │
└─────────────────────────────────────────────────────────────┘
```

## Components and Interfaces

### 1. Data Loader Module

**Purpose**: Load dan preprocess dataset dengan format yang berbeda

**Classes**:


```python
class DatasetLoader:
    """Base class untuk data loading"""
    def load(self, path: str) -> pd.DataFrame:
        raise NotImplementedError
    
    def validate(self, df: pd.DataFrame) -> bool:
        raise NotImplementedError

class UCILoader(DatasetLoader):
    """Loader untuk UCI dataset"""
    def load(self, path: str) -> pd.DataFrame:
        # Parse semicolon-separated, Date+Time merge
        # Resample to hourly
        # IQR outlier clipping
        pass

class PELITALoader(DatasetLoader):
    """Loader untuk PELITA dataset"""
    def load(self, path: str) -> pd.DataFrame:
        # Parse comma-separated
        # Use existing Datetime
        # NO resampling, conditional outlier handling
        pass

class UniversalLoader(DatasetLoader):
    """Auto-detect dan load kedua format"""
    def detect_type(self, df: pd.DataFrame) -> str:
        # Check for PELITA-specific columns
        if 'MCB_Tripped' in df.columns and 'VA' in df.columns:
            return 'PELITA'
        return 'uci'
    
    def load(self, path: str) -> pd.DataFrame:
        # Auto-detect and delegate to appropriate loader
        pass
```

### 2. Feature Engineering Module

**Purpose**: Generate features yang sesuai dengan setiap dataset

**Classes**:

```python
class FeatureEngineer:
    """Base class untuk feature engineering"""
    def create_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        # hour, dayofweek, month, season
        pass
    
    def create_lag_features(self, df: pd.DataFrame, target: str) -> pd.DataFrame:
        # lag_1, lag_24, lag_168
        pass
    
    def create_rolling_features(self, df: pd.DataFrame, target: str) -> pd.DataFrame:
        # rolling_mean_3, rolling_mean_24, rolling_std_24
        pass

class UCIFeatureEngineer(FeatureEngineer):
    """Feature engineering untuk UCI"""
    def engineer(self, df: pd.DataFrame) -> pd.DataFrame:
        # Generate all features from scratch
        # Sub-metering features
        pass

class PELITAFeatureEngineer(FeatureEngineer):
    """Feature engineering untuk PELITA"""
    def engineer(self, df: pd.DataFrame) -> pd.DataFrame:
        # Use existing Lag_*, Rolling_* (rename to lowercase)
        # Add VA, MCB_Tripped, House_Type encoding
        # Add Emisi_CO2 if needed
        pass
    
    def encode_house_type(self, df: pd.DataFrame) -> pd.DataFrame:
        # One-hot encoding for House_Type
        return pd.get_dummies(df, columns=['House_Type'], prefix='HT')
    
    def normalize_va(self, df: pd.DataFrame) -> pd.DataFrame:
        # Normalize VA to 0-1 range or categorical
        df['VA_normalized'] = df['VA'] / 2200
        return df

class UniversalFeatureEngineer(FeatureEngineer):
    """Adaptive feature engineering"""
    def engineer(self, df: pd.DataFrame, dataset_type: str) -> pd.DataFrame:
        # Create common features
        # Add optional features if available
        pass
```

### 3. Model Training Module

**Purpose**: Train Random Forest dan XGBoost dengan hyperparameter tuning

**Classes**:

```python
class EnergyPredictionModel:
    """Base model class"""
    def __init__(self, model_type='rf', random_state=42):
        self.model_type = model_type
        self.random_state = random_state
        self.model = None
        self.best_params = None
    
    def train(self, X_train, y_train, cv_strategy):
        raise NotImplementedError
    
    def predict(self, X_test):
        return self.model.predict(X_test)
    
    def get_feature_importance(self, feature_names):
        return pd.DataFrame({
            'Feature': feature_names,
            'Importance': self.model.feature_importances_
        }).sort_values('Importance', ascending=False)

class DeltaModel(EnergyPredictionModel):
    """Model Delta - UCI optimized"""
    def get_param_grid(self):
        return {
            'n_estimators': [100, 200],
            'max_depth': [10, 20, None],
            'min_samples_split': [2, 5],
            'min_samples_leaf': [1, 2],
            'max_features': ['sqrt', 'log2']
        }

class NadirModel(EnergyPredictionModel):
    """Model Nadir - PELITA optimized"""
    def get_param_grid(self):
        # Tuned for PELITA characteristics
        return {
            'n_estimators': [150, 250, 300],
            'max_depth': [15, 25, None],
            'min_samples_split': [2, 3, 5],
            'min_samples_leaf': [1, 2],
            'max_features': ['sqrt', 'log2', None],
            'min_samples_leaf': [1, 2, 3]  # More conservative for MCB events
        }
    
    def train_stratified(self, X_train, y_train, va_column):
        # Stratified training by VA class
        pass

class EpsilonModel(EnergyPredictionModel):
    """Model Epsilon - Universal"""
    def get_param_grid(self, dataset_type):
        # Adaptive param grid based on dataset
        if dataset_type == 'PELITA':
            return NadirModel().get_param_grid()
        return DeltaModel().get_param_grid()
```

### 4. Evaluation Module

**Purpose**: Comprehensive evaluation dan comparison

**Classes**:

```python
class ModelEvaluator:
    """Evaluate single model"""
    def evaluate(self, model, X_test, y_test):
        y_pred = model.predict(X_test)
        return {
            'MAE': mean_absolute_error(y_test, y_pred),
            'RMSE': np.sqrt(mean_squared_error(y_test, y_pred)),
            'R2': r2_score(y_test, y_pred),
            'MAPE': np.mean(np.abs((y_test - y_pred) / y_test)) * 100
        }
    
    def plot_predictions(self, y_test, y_pred, model_name):
        # Visualization: actual vs predicted
        pass
    
    def analyze_residuals(self, y_test, y_pred):
        # Residual analysis for diagnostics
        pass

class ComparativeAnalyzer:
    """Compare multiple models"""
    def __init__(self):
        self.results = {}
    
    def add_model_results(self, model_name, metrics, feature_importance, training_time):
        self.results[model_name] = {
            'metrics': metrics,
            'importance': feature_importance,
            'time': training_time
        }
    
    def generate_comparison_table(self):
        # Create DataFrame with all metrics
        pass
    
    def plot_metric_comparison(self):
        # Bar chart: MAE, RMSE, R² comparison
        pass
    
    def plot_feature_importance_comparison(self):
        # Side-by-side feature importance
        pass
    
    def cross_dataset_validation(self, models, datasets):
        # Train on UCI, test on PELITA and vice versa
        pass
    
    def generate_report(self, output_path):
        # Comprehensive markdown/PDF report
        pass
```

### 5. Recommendation Module

**Purpose**: Generate actionable energy saving recommendations

**Classes**:

```python
class RecommendationEngine:
    """Base recommendation engine"""
    def generate(self, feature_importance, data_stats):
        raise NotImplementedError

class UCIRecommendationEngine(RecommendationEngine):
    """Recommendations for UCI dataset"""
    def generate(self, feature_importance, data_stats):
        recommendations = []
        # Time-based recommendations
        # Sub-metering optimization
        # Seasonal adjustments
        return recommendations

class PELITARecommendationEngine(RecommendationEngine):
    """Recommendations for PELITA dataset"""
    def generate(self, feature_importance, data_stats, va_capacity):
        recommendations = []
        # MCB trip prevention
        # VA-aware load management
        # Rupiah savings estimation
        # Peak/off-peak optimization
        return recommendations
    
    def predict_mcb_risk(self, predicted_power, va_capacity):
        # Calculate risk of MCB trip
        utilization = (predicted_power * 1000) / va_capacity
        if utilization > 0.9:
            return 'HIGH', 'Immediate action required'
        elif utilization > 0.8:
            return 'MEDIUM', 'Consider load shedding'
        return 'LOW', 'Normal operation'
    
    def suggest_load_shedding(self, current_loads):
        # Priority: AC > Laundry > Electronics
        suggestions = []
        if current_loads['Sub_metering_3'] > 0:
            suggestions.append('Reduce AC usage (highest priority)')
        if current_loads['Sub_metering_2'] > 0:
            suggestions.append('Postpone laundry')
        return suggestions
    
    def estimate_savings_idr(self, reduction_kwh, tariff_per_kwh=1444.70):
        # PLN tariff 1300VA: Rp 1,444.70/kWh
        return reduction_kwh * tariff_per_kwh
```

## Data Models

### UCI Dataset Schema
```python
{
    'Datetime': datetime,
    'Global_active_power': float,  # kW
    'Global_reactive_power': float,  # kVAR
    'Voltage': float,  # V
    'Global_intensity': float,  # A
    'Sub_metering_1': float,  # kWh (kitchen)
    'Sub_metering_2': float,  # kWh (laundry)
    'Sub_metering_3': float,  # kWh (AC/heater)
    # Engineered features
    'hour': int,
    'dayofweek': int,
    'month': int,
    'season': int,
    'lag_1': float,
    'lag_24': float,
    'lag_168': float,
    'rolling_mean_3': float,
    'rolling_mean_24': float,
    'rolling_std_24': float
}
```

### PELITA Dataset Schema
```python
{
    'Datetime': datetime,
    'Global_active_power': float,  # kW
    'Global_reactive_power': float,  # kVAR
    'Voltage': float,  # V
    'Global_intensity': float,  # A
    'Sub_metering_1': float,  # kWh
    'Sub_metering_2': float,  # kWh
    'Sub_metering_3': float,  # kWh
    'Sub_metering_4': float,  # kWh (residual)
    'Hour': int,
    'Dayofweek': int,
    'Month': int,
    'Season': int,
    'Emisi_CO2': float,  # kg
    'MCB_Tripped': int,  # 0=normal, 1=tripped
    'Lag_1': float,  # Pre-generated
    'Lag_24': float,
    'Lag_168': float,
    'Rolling_mean_3': float,
    'Rolling_mean_24': float,
    'Rolling_std_24': float,
    'House_ID': int,
    'VA': int,  # 900, 1300, 2200
    'House_Type': str,  # working_class, retired, mixed
    # Engineered features
    'VA_normalized': float,
    'HT_working_class': int,  # One-hot encoded
    'HT_retired': int,
    'HT_mixed': int
}
```

## Correctness Properties

*A property is a characteristic or behavior that should hold true across all valid executions of a system-essentially, a formal statement about what the system should do. Properties serve as the bridge between human-readable specifications and machine-verifiable correctness guarantees.*

### Property Reflection

Before defining properties, let's analyze acceptance criteria for testability:

**1.1 Dataset loading format validation**
Thoughts: This is about ensuring correct parsing of different formats. We can test by loading sample files and checking column presence and data types.
Testable: yes - property

**1.2 Feature engineering correctness**
Thoughts: Features should be generated correctly without data leakage. We can test by checking lag features don't use future data.
Testable: yes - property

**1.3 Model training convergence**
Thoughts: Models should train successfully and produce valid predictions. We can test by checking R² > threshold.
Testable: yes - property

**2.1 Universal loader auto-detection**
Thoughts: Should correctly identify dataset type. We can test with known UCI and PELITA samples.
Testable: yes - example

**3.1 MCB trip prediction accuracy**
Thoughts: For PELITA, should predict high risk when power approaches VA limit. We can test with synthetic overload scenarios.
Testable: yes - property

**4.1 Comparative metrics consistency**
Thoughts: All models should be evaluated with same metrics. We can test by checking metric keys match.
Testable: yes - property

**5.1 Recommendation actionability**
Thoughts: Recommendations should be specific and measurable. This is subjective quality.
Testable: no

**6.1 Model persistence round-trip**
Thoughts: Saved model should produce same predictions after loading. Classic round-trip property.
Testable: yes - property

### Correctness Properties

**Property 1: Data Loading Preserves Temporal Order**
*For any* dataset (UCI or PELITA), after loading and preprocessing, the Datetime index should be monotonically increasing with no gaps larger than the sampling interval.
**Validates: Requirements 1.1, 2.2, 2.3**

**Property 2: Feature Engineering No Data Leakage**
*For any* lag or rolling feature at timestamp t, the feature value should only depend on data from timestamps < t, never from t or future timestamps.
**Validates: Requirements 1.2, 2.4, 8.5**

**Property 3: Model Prediction Bounds**
*For any* trained model and test input, predicted Global_active_power should be within physically realistic bounds (0 to max_observed * 1.5), never negative or impossibly high.
**Validates: Requirements 1.5, 2.6, 8.4**

**Property 4: VA Capacity Constraint (PELITA)**
*For any* PELITA prediction where MCB_Tripped=0, the predicted power should respect VA capacity: predicted_power_kW * 1000 / power_factor <= VA_capacity.
**Validates: Requirements 1.4, 1.6, 5.1**

**Property 5: Feature Importance Consistency**
*For any* model trained multiple times with same data and random_state, the top 5 most important features should remain the same (order may vary slightly).
**Validates: Requirements 4.4, 8.1**

**Property 6: Cross-Validation Score Stability**
*For any* model, the standard deviation of cross-validation R² scores should be < 0.15, indicating stable performance across folds.
**Validates: Requirements 8.1, 8.3**

**Property 7: Model Persistence Round-Trip**
*For any* trained model, saving to disk and loading back should produce identical predictions (within floating-point precision ε=1e-6) for the same input.
**Validates: Requirements 7.1, 7.2, 7.3**

**Property 8: Comparative Metrics Completeness**
*For any* comparative analysis, all models (Delta, Nadir, Epsilon) should have metrics computed for the same test set with keys: MAE, RMSE, R², MAPE, training_time.
**Validates: Requirements 4.1, 4.2, 4.5**

**Property 9: PELITA MCB Risk Prediction**
*For any* PELITA data point where actual MCB_Tripped=1, the model should predict power within 10% of the VA limit, indicating high utilization.
**Validates: Requirements 1.6, 6.1**

**Property 10: Universal Loader Type Detection**
*For any* dataset with columns ['MCB_Tripped', 'VA', 'House_Type'], the UniversalLoader should detect type as 'PELITA'; otherwise 'uci'.
**Validates: Requirements 2.1**

## Error Handling

### Data Loading Errors
- **Missing File**: Raise `FileNotFoundError` with clear message
- **Invalid Format**: Raise `ValueError` with format expectations
- **Missing Required Columns**: Raise `KeyError` with list of missing columns
- **Date Parsing Failure**: Raise `ValueError` with problematic date examples

### Feature Engineering Errors
- **Insufficient Data for Lag**: Warn and fill with 0 or mean
- **All-NaN Rolling Window**: Fill with global mean, log warning
- **Data Leakage Detection**: Raise `DataLeakageError` if future data detected

### Model Training Errors
- **Convergence Failure**: Log warning, return best model so far
- **Memory Error**: Suggest reducing n_estimators or using chunking
- **Invalid Hyperparameters**: Raise `ValueError` with valid ranges

### Prediction Errors
- **Out-of-Bounds Prediction**: Clip to valid range, log warning
- **Missing Features**: Raise `ValueError` with list of required features
- **Model Not Trained**: Raise `RuntimeError` with training instructions

### Recommendation Errors
- **Insufficient Data**: Return generic recommendations with disclaimer
- **Invalid VA Capacity**: Raise `ValueError` for PELITA-specific recommendations

## Testing Strategy

### Unit Testing
- Test each data loader with sample files (UCI and PELITA formats)
- Test feature engineering functions with known inputs/outputs
- Test model training with small synthetic dataset (fast execution)
- Test recommendation generation with mock feature importance
- Test metric calculation with known predictions

### Property-Based Testing
Using `hypothesis` library for Python:

**Property Test 1: Temporal Order Preservation**
```python
@given(st.lists(st.datetimes(), min_size=10))
def test_temporal_order_preserved(datetimes):
    df = create_dataframe_from_datetimes(datetimes)
    loaded = loader.load(df)
    assert loaded.index.is_monotonic_increasing
```

**Property Test 2: No Data Leakage in Lag Features**
```python
@given(st.lists(st.floats(min_value=0, max_value=10), min_size=200))
def test_lag_no_future_data(power_values):
    df = create_dataframe(power_values)
    df_eng = engineer.create_lag_features(df, 'power')
    for i in range(1, len(df_eng)):
        assert df_eng.iloc[i]['lag_1'] == df.iloc[i-1]['power']
```

**Property Test 3: Prediction Bounds**
```python
@given(st.data())
def test_prediction_bounds(data):
    X_test = data.draw(generate_valid_features())
    y_pred = model.predict(X_test)
    assert (y_pred >= 0).all()
    assert (y_pred <= max_realistic_power).all()
```

**Property Test 4: VA Capacity Constraint**
```python
@given(st.integers(min_value=900, max_value=2200).filter(lambda x: x in [900, 1300, 2200]))
def test_va_capacity_respected(va_capacity):
    X_test = generate_PELITA_features(va=va_capacity)
    y_pred = model.predict(X_test)
    apparent_power = (y_pred * 1000) / 0.9  # Assume PF=0.9
    assert (apparent_power <= va_capacity * 1.1).all()  # 10% tolerance
```

**Property Test 5: Model Persistence Round-Trip**
```python
def test_model_persistence_roundtrip():
    model.train(X_train, y_train)
    y_pred_before = model.predict(X_test)
    
    model.save('temp_model.joblib')
    loaded_model = Model.load('temp_model.joblib')
    y_pred_after = loaded_model.predict(X_test)
    
    assert np.allclose(y_pred_before, y_pred_after, atol=1e-6)
```

### Integration Testing
- Test full pipeline: load → engineer → train → evaluate → recommend
- Test comparative analysis with all 3 models
- Test cross-dataset validation (train UCI, test PELITA)
- Test model persistence and loading in production-like scenario

### Performance Testing
- Benchmark training time for each model (target: <10 minutes on standard laptop)
- Benchmark inference time (target: <100ms per prediction)
- Memory profiling for large datasets (PELITA 100 houses × 2016 hours)

### Validation Testing
- K-fold cross-validation with TimeSeriesSplit (k=5)
- Walk-forward validation for time series
- Stratified validation by VA class for PELITA
- Out-of-sample testing with holdout set (last 20% chronologically)

---

## Implementation Notes

### Technology Stack
- **Python**: 3.8+
- **Core Libraries**: pandas, numpy, scikit-learn, xgboost
- **Visualization**: matplotlib, seaborn
- **Testing**: pytest, hypothesis
- **Model Persistence**: joblib
- **Notebook**: Jupyter

### File Structure
```
Program/Model/
├── Model_Delta.ipynb          # Existing UCI model
├── Model_Nadir.ipynb          # New PELITA-optimized
├── Model_Epsilon.ipynb        # New Universal model
├── model_comparison.ipynb     # Comparative analysis
├── modules/
│   ├── __init__.py
│   ├── data_loaders.py       # UCI, PELITA, Universal loaders
│   ├── feature_engineering.py # Feature engineers
│   ├── models.py             # Model classes
│   ├── evaluation.py         # Evaluators and comparators
│   └── recommendations.py    # Recommendation engines
├── tests/
│   ├── test_loaders.py
│   ├── test_features.py
│   ├── test_models.py
│   └── test_properties.py    # Property-based tests
└── outputs/
    ├── delta_trained.joblib
    ├── nadir_trained.joblib
    ├── epsilon_trained.joblib
    └── comparison_report.md
```

### Development Workflow
1. Implement modules in `modules/` directory
2. Write unit tests for each module
3. Create Model_Nadir.ipynb using modules
4. Create Model_Epsilon.ipynb using modules
5. Enhance Model_Delta.ipynb to use modules
6. Create model_comparison.ipynb for analysis
7. Run property-based tests
8. Generate final comparison report

This design ensures modularity, testability, and clear separation of concerns while supporting all three model variants and comprehensive comparative analysis.
