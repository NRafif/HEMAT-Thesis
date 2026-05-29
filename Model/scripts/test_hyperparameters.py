"""
Quick Hyperparameter Test - Compare 3 Configurations
====================================================

Test apakah pre-optimized parameters sudah optimal atau bisa lebih baik.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import time

from modules.data_loaders import PELITALoader
from modules.feature_engineering import PELITAFeatureEngineer

print("=" * 70)
print("🧪 HYPERPARAMETER COMPARISON TEST")
print("=" * 70)

# Load data
print("\n📁 Loading PELITA dataset...")
loader = PELITALoader()
data_path = Path(__file__).parent.parent.parent / '1 - Dataset' / 'SINTETIC - PELITA' / 'dataset_energi_rumah_indonesia_PELITA_Fix.csv'
df = loader.load(str(data_path))

# Feature engineering
print("⚙️  Engineering features...")
engineer = PELITAFeatureEngineer()
df_features = engineer.engineer(df)

# Prepare data
X = df_features.drop(columns=['Global_active_power', 'House_ID'], errors='ignore')
y = df_features['Global_active_power']

split_idx = int(len(X) * 0.8)
X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]

print(f"✓ Train: {len(X_train):,} | Test: {len(X_test):,}")

# Test 3 configurations
configs = {
    'Conservative (Ours)': {
        'n_estimators': 200,
        'max_depth': 20,
        'min_samples_split': 3,
        'min_samples_leaf': 2,
        'max_features': 'sqrt'
    },
    'Aggressive': {
        'n_estimators': 300,
        'max_depth': None,
        'min_samples_split': 2,
        'min_samples_leaf': 1,
        'max_features': 'sqrt'
    },
    'Balanced': {
        'n_estimators': 250,
        'max_depth': 25,
        'min_samples_split': 3,
        'min_samples_leaf': 2,
        'max_features': 'log2'
    }
}

results = []

print("\n" + "=" * 70)
print("🏃 TRAINING & EVALUATING 3 CONFIGURATIONS...")
print("=" * 70)

for name, params in configs.items():
    print(f"\n🔧 Testing: {name}")
    print(f"   Parameters: {params}")
    
    start = time.time()
    model = RandomForestRegressor(**params, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    train_time = time.time() - start
    
    y_pred = model.predict(X_test)
    
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    
    results.append({
        'Config': name,
        'R²': r2,
        'MAE': mae,
        'RMSE': rmse,
        'Time (min)': train_time / 60
    })
    
    print(f"   ✓ R² = {r2:.6f} | MAE = {mae:.4f} kW | Time = {train_time/60:.2f} min")

# Summary
print("\n" + "=" * 70)
print("📊 COMPARISON RESULTS")
print("=" * 70)

df_results = pd.DataFrame(results)
df_results = df_results.sort_values('R²', ascending=False)

print("\n" + df_results.to_string(index=False))

# Find best
best = df_results.iloc[0]
ours = df_results[df_results['Config'] == 'Conservative (Ours)'].iloc[0]

print("\n" + "=" * 70)
print("🏆 VERDICT")
print("=" * 70)

improvement = (best['R²'] - ours['R²']) * 100
time_diff = best['Time (min)'] - ours['Time (min)']

print(f"\n🥇 Best Config: {best['Config']}")
print(f"   R² = {best['R²']:.6f}")
print(f"   MAE = {best['MAE']:.4f} kW")
print(f"   Time = {best['Time (min)']:.2f} min")

print(f"\n📌 Our Config (Conservative):")
print(f"   R² = {ours['R²']:.6f}")
print(f"   MAE = {ours['MAE']:.4f} kW")
print(f"   Time = {ours['Time (min)']:.2f} min")

if best['Config'] == 'Conservative (Ours)':
    print(f"\n✅ KESIMPULAN: Pre-optimized parameters sudah OPTIMAL!")
    print(f"   Tidak perlu grid search yang lama.")
else:
    print(f"\n⚠️  KESIMPULAN: Ada config yang sedikit lebih baik")
    print(f"   Improvement: +{improvement:.4f}% R²")
    print(f"   Time difference: +{time_diff:.2f} minutes")
    
    if improvement < 0.1:
        print(f"\n💡 REKOMENDASI: Improvement < 0.1% tidak signifikan.")
        print(f"   Tetap gunakan Conservative untuk efisiensi waktu.")
    else:
        print(f"\n💡 REKOMENDASI: Pertimbangkan gunakan {best['Config']}")

print("\n" + "=" * 70)
