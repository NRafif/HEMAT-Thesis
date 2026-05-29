# ⚡ PELITA - Sistem Prediksi Hemat Listrik

**Web Application untuk Prediksi Konsumsi Energi & Rekomendasi Hemat Listrik**

## 🎯 Fitur Utama

### 🏠 Dashboard
- **Metrics Real-time**: Daya terpasang, konsumsi saat ini, estimasi biaya
- **Status Risiko MCB**: Peringatan visual jika konsumsi mendekati batas
- **Gauge Chart**: Visualisasi utilisasi daya yang mudah dipahami
- **Pola Konsumsi**: Grafik konsumsi harian 24 jam
- **Distribusi Peralatan**: Pie chart konsumsi per peralatan

### 💡 Rekomendasi Cerdas
- Rekomendasi prioritas (HIGH/MEDIUM/LOW)
- Estimasi penghematan dalam Rupiah
- Tips khusus berdasarkan tipe hunian
- Peringatan MCB yang user-friendly

### 🎮 Simulasi Penghematan
- Slider interaktif untuk target pengurangan
- Estimasi hemat bulanan dan tahunan
- Perbandingan biaya sebelum/sesudah

### 📊 Analisis Model
- Perbandingan 3 model ML (Delta, Nadir, Epsilon)
- Visualisasi akurasi dan feature importance
- Detail teknis untuk akademisi

## 🚀 Cara Menjalankan

### 1. Install Dependencies
```bash
cd Program/App
pip install streamlit plotly pandas numpy
```

### 2. Jalankan Aplikasi
```bash
streamlit run app.py
```

### 3. Buka Browser
Aplikasi akan terbuka di `http://localhost:8501`

## 📱 Cara Penggunaan

1. **Isi Data Rumah** di sidebar:
   - Pilih daya listrik (VA)
   - Pilih tipe hunian
   - Atur konsumsi saat ini (Watt)
   - Centang peralatan yang aktif

2. **Lihat Dashboard**:
   - Metrics utama di bagian atas
   - Gauge utilisasi daya
   - Grafik pola konsumsi

3. **Baca Rekomendasi**:
   - Prioritas tinggi (merah) = segera lakukan
   - Prioritas sedang (kuning) = perhatikan
   - Prioritas rendah (hijau) = tips umum

4. **Simulasi Penghematan**:
   - Geser slider untuk target pengurangan
   - Lihat estimasi penghematan

## 🎨 Design Philosophy

- **User-Centric**: Bahasa awam, bukan istilah teknis
- **Visual Appeal**: Warna gradient, animasi, icon
- **Actionable**: Fokus pada rekomendasi praktis
- **Indonesia-Specific**: Tarif PLN, VA capacity, Rupiah

## 📊 Model Performance

| Model | R² Score | MAE | Best For |
|-------|----------|-----|----------|
| Delta | 0.9961 | 1.6188 kW | UCI/International |
| Nadir | 0.9956 | 0.0161 kW | PELITA/Indonesia |
| Epsilon | 0.9960 | 1.4934 kW | Universal |

## 🔧 Tech Stack

- **Frontend**: Streamlit
- **Visualization**: Plotly
- **Data**: Pandas, NumPy
- **ML**: Scikit-learn (Random Forest)
- **Language**: Python 3.12

## 📝 Credits

**Pengembang:** Nofal Rafif  
**NIM:** 221011402613  
**Universitas:** Universitas Pamulang  
**Tahun:** 2024

---

⚡ **PELITA** - Zero Energy Negligence for Indonesian Households
