# Requirements Document: Energy Prediction Models Enhancement

## Introduction

Pengembangan sistem prediksi konsumsi energi rumah tangga menggunakan Random Forest dengan 3 pendekatan berbeda untuk membandingkan performa dan kesesuaian dengan dataset UCI (internasional) dan PELITA (Indonesia). Penelitian ini bertujuan untuk menghasilkan model yang optimal untuk konteks Indonesia sambil memvalidasi dengan benchmark internasional.

## Glossary

- **Model Delta**: Model existing yang dioptimalkan untuk dataset UCI
- **Model Nadir**: Model baru yang dioptimalkan khusus untuk dataset PELITA Indonesia
- **Model Epsilon**: Model universal yang dapat bekerja dengan kedua dataset (UCI dan PELITA)
- **UCI Dataset**: Dataset Individual Household Electric Power Consumption dari UCI Machine Learning Repository
- **PELITA Dataset**: Dataset sintetis energi rumah Indonesia dengan fitur MCB, VA, dan karakteristik lokal
- **MCB (Miniature Circuit Breaker)**: Pemutus arus otomatis pada instalasi listrik rumah
- **VA (Volt-Ampere)**: Kapasitas daya listrik rumah (900VA, 1300VA, 2200VA)
- **Random Forest**: Algoritma ensemble learning untuk prediksi regresi
- **Feature Engineering**: Proses pembuatan fitur baru dari data mentah
- **Time Series Split**: Metode validasi untuk data temporal yang mempertahankan urutan waktu

## Requirements

### Requirement 1: Model Nadir - PELITA-Optimized Model

**User Story:** Sebagai peneliti, saya ingin model yang dioptimalkan khusus untuk dataset PELITA Indonesia, sehingga dapat memanfaatkan semua fitur unik konteks lokal dan menghasilkan prediksi yang akurat untuk kondisi Indonesia.

#### Acceptance Criteria

1. WHEN Model Nadir memuat dataset PELITA THEN sistem SHALL membaca format CSV dengan separator koma dan kolom Datetime yang sudah terformat
2. WHEN Model Nadir melakukan feature engineering THEN sistem SHALL menggunakan fitur yang sudah ada (Lag_1, Lag_24, Lag_168, Rolling_mean_3, Rolling_mean_24, Rolling_std_24) tanpa regenerasi
3. WHEN Model Nadir memilih fitur THEN sistem SHALL memasukkan fitur spesifik PELITA (VA, MCB_Tripped, House_Type, Emisi_CO2) sebagai predictor
4. WHEN Model Nadir melakukan preprocessing THEN sistem SHALL menghindari outlier clipping pada data MCB trip (power=0 adalah valid)
5. WHEN Model Nadir melatih model THEN sistem SHALL menggunakan data dari 100 rumah dengan stratifikasi berdasarkan VA class
6. WHEN Model Nadir menghasilkan rekomendasi THEN sistem SHALL memberikan saran pencegahan MCB trip berdasarkan VA capacity
7. WHEN Model Nadir mengevaluasi performa THEN sistem SHALL menghitung metrik terpisah untuk setiap VA class (900VA, 1300VA, 2200VA)

### Requirement 2: Model Epsilon - Universal Model

**User Story:** Sebagai peneliti, saya ingin model universal yang dapat bekerja dengan dataset UCI dan PELITA, sehingga dapat membandingkan performa lintas dataset dan memvalidasi generalisasi model.

#### Acceptance Criteria

1. WHEN Model Epsilon menerima input dataset THEN sistem SHALL mendeteksi tipe dataset secara otomatis berdasarkan struktur kolom
2. WHEN Model Epsilon memproses dataset UCI THEN sistem SHALL melakukan parsing Date+Time, resampling per jam, dan IQR outlier clipping
3. WHEN Model Epsilon memproses dataset PELITA THEN sistem SHALL menggunakan Datetime existing, skip resampling, dan conditional outlier handling
4. WHEN Model Epsilon melakukan feature engineering THEN sistem SHALL membuat fitur universal yang ada di kedua dataset (hour, dayofweek, month, season, lag, rolling)
5. WHEN Model Epsilon memilih fitur THEN sistem SHALL menggunakan intersection features yang ada di kedua dataset plus optional features jika tersedia
6. WHEN Model Epsilon melatih model THEN sistem SHALL menggunakan hyperparameter yang optimal untuk kedua jenis dataset
7. WHEN Model Epsilon mengevaluasi THEN sistem SHALL menghasilkan comparative report untuk UCI vs PELITA performance
8. WHEN Model Epsilon menghasilkan rekomendasi THEN sistem SHALL menyesuaikan saran berdasarkan karakteristik dataset yang digunakan

### Requirement 3: Model Delta Enhancement

**User Story:** Sebagai peneliti, saya ingin Model Delta yang existing tetap dipertahankan sebagai baseline untuk dataset UCI, sehingga dapat membandingkan improvement dari model baru.

#### Acceptance Criteria

1. WHEN Model Delta dijalankan THEN sistem SHALL mempertahankan semua fungsi existing untuk dataset UCI
2. WHEN Model Delta dibandingkan dengan model lain THEN sistem SHALL menggunakan dataset UCI yang sama untuk fair comparison
3. WHEN Model Delta dievaluasi THEN sistem SHALL menghasilkan metrik baseline (MAE, RMSE, R²) untuk referensi

### Requirement 4: Comparative Analysis System

**User Story:** Sebagai peneliti, saya ingin sistem perbandingan komprehensif antara ketiga model, sehingga dapat mengidentifikasi kelebihan dan kekurangan masing-masing pendekatan.

#### Acceptance Criteria

1. WHEN sistem melakukan comparative analysis THEN sistem SHALL menjalankan ketiga model dengan kondisi yang sama (random seed, train/test split)
2. WHEN sistem mengevaluasi performa THEN sistem SHALL menghitung metrik standar (MAE, RMSE, R², MAPE) untuk setiap model
3. WHEN sistem membandingkan model THEN sistem SHALL menghasilkan visualization (bar chart, line plot) untuk perbandingan metrik
4. WHEN sistem menganalisis feature importance THEN sistem SHALL membandingkan top 10 features dari setiap model
5. WHEN sistem mengevaluasi training time THEN sistem SHALL mencatat dan membandingkan waktu training dan inference
6. WHEN sistem menghasilkan laporan THEN sistem SHALL membuat summary table dengan rekomendasi penggunaan setiap model
7. WHEN sistem menguji generalisasi THEN sistem SHALL melakukan cross-dataset testing (train UCI test PELITA, dan sebaliknya) untuk Model Epsilon

### Requirement 5: PELITA-Specific Features Integration

**User Story:** Sebagai peneliti, saya ingin memanfaatkan fitur unik PELITA untuk meningkatkan akurasi prediksi, sehingga model dapat memberikan insight spesifik untuk konteks Indonesia.

#### Acceptance Criteria

1. WHEN sistem menggunakan fitur VA THEN sistem SHALL mengkategorikan atau normalize berdasarkan kapasitas (900/1300/2200)
2. WHEN sistem menggunakan MCB_Tripped THEN sistem SHALL memperlakukan sebagai binary feature dan menganalisis korelasi dengan overload
3. WHEN sistem menggunakan House_Type THEN sistem SHALL melakukan one-hot encoding untuk categorical feature (working_class, retired, mixed)
4. WHEN sistem menggunakan Emisi_CO2 THEN sistem SHALL menganalisis sebagai target alternatif atau feature tambahan
5. WHEN sistem menggunakan House_ID THEN sistem SHALL mempertimbangkan strategi multi-house (aggregate, per-house, atau feature)

### Requirement 6: Enhanced Recommendation System

**User Story:** Sebagai pengguna akhir, saya ingin rekomendasi penghematan energi yang spesifik dan actionable, sehingga dapat mengurangi konsumsi dan mencegah MCB trip.

#### Acceptance Criteria

1. WHEN sistem menghasilkan rekomendasi untuk PELITA THEN sistem SHALL memberikan warning jika prediksi mendekati VA limit (>80% capacity)
2. WHEN sistem mendeteksi pola overload THEN sistem SHALL menyarankan load shedding priority (AC > Laundry > Electronics)
3. WHEN sistem menganalisis konsumsi THEN sistem SHALL memberikan estimasi penghematan dalam Rupiah berdasarkan tarif PLN
4. WHEN sistem memberikan rekomendasi waktu THEN sistem SHALL mempertimbangkan tarif listrik off-peak vs peak hours
5. WHEN sistem menghasilkan insight THEN sistem SHALL membandingkan konsumsi user dengan rata-rata rumah dengan VA yang sama

### Requirement 7: Model Persistence and Deployment

**User Story:** Sebagai developer, saya ingin model yang terlatih dapat disimpan dan dimuat kembali, sehingga dapat digunakan dalam aplikasi production tanpa retraining.

#### Acceptance Criteria

1. WHEN sistem melatih model THEN sistem SHALL menyimpan model dengan naming convention (nadir_trained.joblib, epsilon_trained.joblib)
2. WHEN sistem menyimpan model THEN sistem SHALL menyertakan metadata (dataset type, training date, hyperparameters, performance metrics)
3. WHEN sistem memuat model THEN sistem SHALL memvalidasi compatibility dengan input data format
4. WHEN sistem deploy model THEN sistem SHALL menyediakan API wrapper untuk inference
5. WHEN sistem versioning model THEN sistem SHALL menggunakan timestamp atau version number untuk tracking

### Requirement 8: Validation and Testing

**User Story:** Sebagai peneliti, saya ingin validasi yang rigorous untuk memastikan model tidak overfitting dan dapat generalize dengan baik.

#### Acceptance Criteria

1. WHEN sistem melakukan cross-validation THEN sistem SHALL menggunakan TimeSeriesSplit dengan minimum 5 folds
2. WHEN sistem menguji model THEN sistem SHALL melakukan walk-forward validation untuk time series
3. WHEN sistem mendeteksi overfitting THEN sistem SHALL membandingkan train vs test performance dan memberikan warning jika gap >15%
4. WHEN sistem menguji robustness THEN sistem SHALL melakukan sensitivity analysis terhadap missing values dan outliers
5. WHEN sistem validasi PELITA features THEN sistem SHALL memastikan tidak ada data leakage dari future timestamps

### Requirement 9: Documentation and Reproducibility

**User Story:** Sebagai peneliti, saya ingin dokumentasi lengkap dan reproducible results, sehingga penelitian dapat diverifikasi dan dikembangkan lebih lanjut.

#### Acceptance Criteria

1. WHEN sistem menjalankan eksperimen THEN sistem SHALL mencatat semua hyperparameters dan random seeds
2. WHEN sistem menghasilkan hasil THEN sistem SHALL menyimpan log lengkap (training progress, metrics, timestamps)
3. WHEN sistem membuat visualisasi THEN sistem SHALL menyimpan plot dalam format high-resolution (PNG/PDF)
4. WHEN sistem menulis kode THEN sistem SHALL mengikuti PEP 8 style guide dan menyertakan docstrings
5. WHEN sistem membuat notebook THEN sistem SHALL mengorganisir dalam sections yang jelas (Data Loading, EDA, Training, Evaluation, Comparison)

### Requirement 10: Performance Optimization

**User Story:** Sebagai developer, saya ingin model yang efisien dalam training dan inference, sehingga dapat dijalankan pada resource terbatas.

#### Acceptance Criteria

1. WHEN sistem melatih model THEN sistem SHALL menggunakan n_jobs=-1 untuk parallel processing
2. WHEN sistem melakukan grid search THEN sistem SHALL menggunakan parameter grid yang reasonable (tidak exhaustive)
3. WHEN sistem memproses data besar THEN sistem SHALL menggunakan chunking atau batch processing jika memory terbatas
4. WHEN sistem melakukan inference THEN sistem SHALL mengoptimalkan untuk latency <100ms per prediction
5. WHEN sistem menyimpan model THEN sistem SHALL menggunakan compression jika file size >100MB

---

## Summary

Dokumen ini mendefinisikan requirements untuk pengembangan 3 model prediksi energi:
1. **Model Nadir**: Optimized untuk PELITA (Indonesia-specific)
2. **Model Epsilon**: Universal model untuk UCI dan PELITA
3. **Model Delta**: Baseline untuk UCI (existing)

Dengan comparative analysis yang komprehensif untuk mengidentifikasi model terbaik untuk setiap use case.
