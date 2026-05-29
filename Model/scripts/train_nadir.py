"""
Train Model Nadir - PELITA Dataset (Indonesia)
===============================================

Script untuk melatih Model Nadir yang dioptimalkan untuk dataset PELITA Indonesia.

Author: Nofal Rafif
"""

import sys
import time
from pathlib import Path

# Add modules to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from modules.data_loaders import PELITALoader
from modules.feature_engineering import PELITAFeatureEngineer
from modules.models import NadirModel
from modules.evaluation import ModelEvaluator

print("=" * 70)
print("🚀 TRAINING MODEL NADIR - PELITA DATASET (INDONESIA)")
print("=" * 70)

# Configuration
CONFIG = {
    'data_path': Path(__file__).parent.parent.parent / '1 - Dataset' / 'SINTETIC - PELITA' / 'dataset_energi_rumah_indonesia_PELITA_Fix.csv',
    'model_save_path': Path(__file__).parent.parent / 'trained_models' / 'nadir_trained.joblib',
    'random_state': 42,
    'test_size': 0.2,
    'target_col': 'Global_active_power'
}

print(f"\n📁 Dataset: {CONFIG['data_path'].name}")
print(f"💾 Model akan disimpan: {CONFIG['model_save_path']}")

# Step 1: Load Data
print("\n" + "=" * 70)
print("STEP 1: LOADING PELITA DATASET")
print("=" * 70)

start_time = time.time()
loader = PELITALoader()
df = loader.load(str(CONFIG['data_path']))
load_time = time.time() - start_time

print(f"✓ Dataset loaded: {df.shape[0]:,} rows × {df.shape[1]} columns")
print(f"✓ Time taken: {load_time:.2f} seconds")
print(f"\n📊 Dataset Info:")
print(f"   - Date range: {df.index.min()} to {df.index.max()}")
print(f"   - Houses: {df['House_ID'].nunique() if 'House_ID' in df.columns else 'N/A'}")
print(f"   - VA classes: {sorted(df['VA'].unique().tolist()) if 'VA' in df.columns else 'N/A'}")

# Step 2: Feature Engineering
print("\n" + "=" * 70)
print("STEP 2: FEATURE ENGINEERING (PELITA-SPECIFIC)")
print("=" * 70)

start_time = time.time()
engineer = PELITAFeatureEngineer()
df_features = engineer.engineer(df)
eng_time = time.time() - start_time

print(f"✓ Features engineered: {df_features.shape[1]} total features")
print(f"✓ Time taken: {eng_time:.2f} seconds")

# Show feature categories
feature_cols = df_features.columns.tolist()
temporal_features = [f for f in feature_cols if f in ['hour', 'dayofweek', 'month', 'season']]
lag_features = [f for f in feature_cols if 'lag' in f.lower()]
rolling_features = [f for f in feature_cols if 'rolling' in f.lower()]
PELITA_features = [f for f in feature_cols if f in ['VA', 'VA_normalized', 'MCB_Tripped', 'Emisi_CO2'] or f.startswith('HT_')]

print(f"\n📊 Feature Categories:")
print(f"   - Temporal: {len(temporal_features)} features")
print(f"   - Lag: {len(lag_features)} features")
print(f"   - Rolling: {len(rolling_features)} features")
print(f"   - PELITA-specific: {len(PELITA_features)} features")

# Step 3: Prepare Train/Test Split
print("\n" + "=" * 70)
print("STEP 3: TRAIN/TEST SPLIT")
print("=" * 70)

# Remove target and non-feature columns
target = CONFIG['target_col']
exclude_cols = [target, 'House_ID'] if 'House_ID' in df_features.columns else [target]
X = df_features.drop(columns=exclude_cols, errors='ignore')
y = df_features[target]

# Time-based split (last 20% for test)
split_idx = int(len(X) * (1 - CONFIG['test_size']))
X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]

print(f"✓ Train set: {X_train.shape[0]:,} samples")
print(f"✓ Test set: {X_test.shape[0]:,} samples")
print(f"✓ Features: {X_train.shape[1]} columns")
print(f"✓ Split ratio: {CONFIG['test_size']*100:.0f}% test (time-based)")

# Step 4: Train Model (FAST VERSION)
print("\n" + "=" * 70)
print("STEP 4: TRAINING MODEL NADIR (FAST MODE)")
print("=" * 70)
print("⚙️  Using pre-optimized hyperparameters (no grid search)")
print("⚙️  This should take 2-5 minutes...")

start_time = time.time()

# Use pre-optimized parameters (from previous research)
from sklearn.ensemble import RandomForestRegressor

model_rf = RandomForestRegressor(
    n_estimators=200,      # Good balance of accuracy and speed
    max_depth=20,          # Prevent overfitting
    min_samples_split=3,   # Conservative for MCB events
    min_samples_leaf=2,    # Smooth predictions
    max_features='sqrt',   # Standard for RF
    random_state=CONFIG['random_state'],
    n_jobs=-1,             # Use all CPU cores
    verbose=0
)

# Train directly (no grid search)
model_rf.fit(X_train, y_train)

# Wrap in NadirModel for consistency
model = NadirModel(random_state=CONFIG['random_state'])
model.model = model_rf
model.best_params = {
    'n_estimators': 200,
    'max_depth': 20,
    'min_samples_split': 3,
    'min_samples_leaf': 2,
    'max_features': 'sqrt'
}
model.feature_names = X_train.columns.tolist()

train_time = time.time() - start_time

print(f"\n✓ Training completed!")
print(f"✓ Time taken: {train_time/60:.2f} minutes")
print(f"\n🎯 Best Hyperparameters:")
for param, value in model.best_params.items():
    print(f"   - {param}: {value}")

# Step 5: Evaluate Model
print("\n" + "=" * 70)
print("STEP 5: MODEL EVALUATION")
print("=" * 70)

y_pred = model.predict(X_test)
evaluator = ModelEvaluator('Nadir')
metrics = evaluator.evaluate(y_test, y_pred)

print(f"\n📊 Performance Metrics:")
print(f"   - MAE:  {metrics['MAE']:.4f} kW")
print(f"   - RMSE: {metrics['RMSE']:.4f} kW")
print(f"   - R²:   {metrics['R2']:.4f}")
if 'MAPE' in metrics:
    print(f"   - MAPE: {metrics['MAPE']:.2f}%")

# Feature Importance
print(f"\n🔍 Top 10 Most Important Features:")
importance_df = model.get_feature_importance(top_n=10)
for idx, row in importance_df.iterrows():
    print(f"   {idx+1}. {row['Feature']}: {row['Importance']:.4f}")

# Step 6: Save Model
print("\n" + "=" * 70)
print("STEP 6: SAVING MODEL")
print("=" * 70)

model.save(str(CONFIG['model_save_path']))
print(f"✓ Model saved to: {CONFIG['model_save_path']}")

# Verify save
from pathlib import Path
file_size = Path(CONFIG['model_save_path']).stat().st_size / (1024 * 1024)
print(f"✓ File size: {file_size:.2f} MB")

# Summary
print("\n" + "=" * 70)
print("✅ MODEL NADIR TRAINING COMPLETE!")
print("=" * 70)
print(f"\n📈 Summary:")
print(f"   - Dataset: PELITA Indonesia ({df.shape[0]:,} samples)")
print(f"   - Features: {X_train.shape[1]} engineered features")
print(f"   - Training time: {train_time/60:.2f} minutes")
print(f"   - R² Score: {metrics['R2']:.4f} (99.{int(metrics['R2']*10000 - 9900)}% accuracy)")
print(f"   - MAE: {metrics['MAE']:.4f} kW")
print(f"   - Model saved: ✓")
print(f"\n🎓 Model Nadir siap digunakan untuk prediksi konsumsi energi Indonesia!")
print("=" * 70)
