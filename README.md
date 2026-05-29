# Energy Prediction Models - Household Consumption Forecasting

Sistem prediksi konsumsi energi rumah tangga menggunakan Random Forest dengan 3 pendekatan model berbeda untuk dataset UCI (internasional) dan PELITA (Indonesia).

## 🎯 Models

### 1. Model Delta (UCI Baseline)
- **Dataset**: UCI Individual Household Electric Power Consumption
- **Optimization**: Tuned untuk karakteristik dataset Prancis
- **Use Case**: International benchmark, validasi dengan dataset standar

### 2. Model Nadir (PELITA Optimized)
- **Dataset**: PELITA Indonesia Synthetic Dataset
- **Optimization**: Memanfaatkan fitur VA, MCB_Tripped, House_Type
- **Use Case**: Prediksi untuk konteks Indonesia dengan MCB trip prevention

### 3. Model Epsilon (Universal)
- **Dataset**: UCI dan PELITA (auto-detect)
- **Optimization**: Adaptive hyperparameters
- **Use Case**: General purpose, cross-dataset validation

## 📁 Project Structure

```
Program/Model/
├── modules/                    # Core modules (reusable)
│   ├── __init__.py
│   ├── data_loaders.py        # UCI, PELITA, Universal loaders
│   ├── feature_engineering.py # Feature engineers
│   ├── models.py              # Model classes
│   ├── evaluation.py          # Evaluators and comparators
│   └── recommendations.py     # Recommendation engines
├── tests/                      # Test suite
│   ├── __init__.py
│   ├── conftest.py            # Pytest fixtures
│   ├── test_loaders.py
│   ├── test_features.py
│   ├── test_models.py
│   └── test_properties.py     # Property-based tests
├── outputs/                    # Trained models and reports
│   ├── delta_trained.joblib
│   ├── nadir_trained.joblib
│   ├── epsilon_trained.joblib
│   └── comparison_report.md
├── Model_Delta.ipynb          # UCI baseline model
├── Model_Nadir.ipynb          # PELITA-optimized model
├── Model_Epsilon.ipynb        # Universal model
├── model_comparison.ipynb     # Comparative analysis
├── requirements.txt           # Dependencies
├── pytest.ini                 # Test configuration
└── README.md                  # This file
```

## 🚀 Quick Start

### Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Verify installation
python -c "import modules; print(modules.__version__)"
```

### Usage

#### 1. Load and Predict with Model Delta (UCI)
```python
from modules import UCILoader, DeltaModel

# Load UCI dataset
loader = UCILoader()
df = loader.load('path/to/household_power_consumption.txt')

# Load trained model
model = DeltaModel.load('outputs/delta_trained.joblib')

# Predict
predictions = model.predict(X_test)
```

#### 2. Load and Predict with Model Nadir (PELITA)
```python
from modules import PELITALoader, NadirModel

# Load PELITA dataset
loader = PELITALoader()
df = loader.load('path/to/dataset_energi_rumah_indonesia_PELITA_Fix.csv')

# Load trained model
model = NadirModel.load('outputs/nadir_trained.joblib')

# Predict with MCB risk assessment
predictions = model.predict(X_test)
risk_level, message = model.predict_mcb_risk(predictions, va_capacity=1300)
```

#### 3. Universal Model (Auto-detect)
```python
from modules import UniversalLoader, EpsilonModel

# Auto-detect and load
loader = UniversalLoader()
df = loader.load('path/to/dataset.csv')  # Works with both UCI and PELITA

# Load trained model
model = EpsilonModel.load('outputs/epsilon_trained.joblib')

# Predict
predictions = model.predict(X_test)
```

## 🧪 Testing

```bash
# Run all tests
pytest

# Run specific test categories
pytest -m unit          # Unit tests only
pytest -m property      # Property-based tests only
pytest -m integration   # Integration tests only

# Run with coverage report
pytest --cov=modules --cov-report=html

# Run specific test file
pytest tests/test_loaders.py -v
```

## 📊 Model Comparison

Run the comparative analysis notebook:

```bash
jupyter notebook model_comparison.ipynb
```

This will generate:
- Performance metrics comparison (MAE, RMSE, R², MAPE)
- Feature importance comparison
- Training time analysis
- Cross-dataset validation results
- Model selection recommendations

## 🎓 Research Context

**Thesis**: Sistem Prediksi Konsumsi Energi Rumah Tangga Menggunakan Random Forest  
**Author**: Nofal Rafif  
**Institution**: Universitas Pamulang  
**Year**: 2025

### Key Contributions

1. **PELITA Dataset**: Synthetic dataset dengan karakteristik Indonesia (MCB, VA, House_Type)
2. **Comparative Study**: Perbandingan 3 pendekatan model (specialized vs universal)
3. **MCB Trip Prevention**: Sistem rekomendasi untuk mencegah overload
4. **Property-Based Testing**: Validasi correctness dengan 10 properties

## 📝 Citation

```bibtex
@thesis{rafif2025energy,
  title={Sistem Prediksi Konsumsi Energi Rumah Tangga Menggunakan Random Forest},
  author={Rafif, Nofal},
  year={2025},
  school={Universitas Pamulang}
}
```

## 📄 License

This project is part of academic research at Universitas Pamulang.

## 🤝 Contact

For questions or collaboration:
- **Author**: Nofal Rafif
- **Project**: Energy Prediction Models
- **Year**: 2025
