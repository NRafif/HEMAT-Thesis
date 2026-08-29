"""
🏠 H.E.M.A.T - Home Energy Management and Recommendation Technology
========================================================

Enhanced version with:
- Model ID (PELITA/Indonesia) — satu-satunya dataset aktif di dashboard ini.
  [FIX #9] Sebelumnya tertulis "Dataset selection (France vs Indonesia)" dan
  "Model selection (FR and ID)" sebagai fitur -- ini sudah tidak akurat sejak
  `is_indonesia = True` di-hardcode di main() dan tidak ada toggle apa pun
  di UI untuk memilih dataset Prancis. Beberapa fungsi (prepare_model_input_
  for_hour, predict_daily_curve, generate_pdf_report) masih membawa parameter
  is_indonesia sebagai sisa infrastruktur dari versi lama yang pernah
  mendukung 2 dataset -- dipertahankan agar tidak perlu ubah banyak
  signature, tapi jalur France sudah tidak dapat diakses dari UI.
- Preset scenarios (Profil Kepemilikan Rumah)
- Clear analysis logic
- PDF export (termasuk konteks simulasi Weekday/Weekend)
- Analysis button to trigger breakdown
- Recommendation database (RECOMMENDATION_DB + COMBINATION_RULES) untuk
  rekomendasi yang lebih personal per alat & kombinasi alat

Author: Nofal Rafif
"""

import logging
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import plotly.graph_objects as go
from pathlib import Path
import joblib
import os
import base64
from io import BytesIO
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, PageBreak

# [FIX #10] Logger modul -- dipakai predict_daily_curve() supaya kegagalan
# prediksi per-jam tidak lagi ditelan diam-diam tanpa jejak (lihat except
# block di dalam predict_daily_curve untuk detail).
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger("hemat")

# Page config
st.set_page_config(
    page_title="HEMAT - Savings Energy",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .metric-card {
        background: white;
        border-radius: 15px;
        padding: 20px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        margin: 10px 0;
        transition: transform 0.3s ease;
    }
    .metric-card:hover { transform: translateY(-5px); }
    
    .risk-low {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        color: white !important; 
        padding: 15px 25px; 
        border-radius: 10px; 
        font-weight: bold;
    }
    .risk-low h4, .risk-low h2, .risk-low p {
        color: white !important;
    }
    
    .risk-medium {
        background: linear-gradient(135deg, #F2994A 0%, #F2C94C 100%);
        color: white !important; 
        padding: 15px 25px; 
        border-radius: 10px; 
        font-weight: bold;
    }
    .risk-medium h4, .risk-medium h2, .risk-medium p {
        color: white !important;
    }
    
    .risk-high {
        background: linear-gradient(135deg, #eb3349 0%, #f45c43 100%);
        color: white !important; 
        padding: 15px 25px; 
        border-radius: 10px; 
        font-weight: bold;
        border-left: 5px solid #c0392b;
    }
    .risk-high h4, .risk-high h2, .risk-high p {
        color: white !important;
    }
    
    .rec-card {
        border-left: 4px solid;
        padding: 15px;
        margin: 10px 0;
        border-radius: 0 10px 10px 0;
        background: white;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    }
    .rec-card strong {
        color: #2c3e50 !important;
        font-size: 1.05rem;
        display: block;
        margin-bottom: 8px;
    }
    .rec-card p {
        color: #555 !important;
        margin: 5px 0;
    }
    .rec-high { 
        border-color: #eb3349; 
        background: #fff5f5; 
    }
    .rec-medium { 
        border-color: #F2994A; 
        background: #fffaf0; 
    }
    .rec-low { 
        border-color: #11998e; 
        background: #f0fff4; 
    }
    
    .analysis-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-left: 4px solid #38ef7d;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
        color: white !important;
        line-height: 1.8;
    }
    .analysis-box strong {
        color: white !important;
        font-weight: bold;
    }
    .analysis-box br {
        display: block;
        content: "";
        margin-top: 8px;
    }
    
    .stButton>button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 10px 30px;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# DATA DEFINITIONS
# ============================================================================

APPLIANCES_FRANCE = {
    'dishwasher': ('🍽️ Dishwasher', 1200, 1.5),  # 1.5 jam/hari
    'electric_heater': ('🔥 Electric Heater', 2000, 6),  # 6 jam/hari
    'water_heater': ('🚿 Water Heater', 1500, 2),  # 2 jam/hari
    'laundry': ('🧺 Laundry Machine', 500, 1),  # 1 jam/hari
    'kettle': ('☕ Electric Kettle', 1800, 0.5),  # 30 menit/hari
    'refrigerator': ('🧊 Refrigerator', 150, 24),  # 24 jam
    'tv': ('📺 TV', 100, 6),  # 6 jam/hari
    'computer': ('💻 Computer', 200, 8)  # 8 jam/hari
}

APPLIANCES_INDONESIA = {
    # Kategori Dapur & Pemanggang
    'refrigerator': ('🧊 Kulkas (2 Pintu)', 125, 24),
    'dispenser': ('🥤 Dispenser (Hot/Normal)', 350, 24),
    'rice_cooker': ('🍚 Rice Cooker', [('COOK', 350, 0.5), ('WARM', 50, 3.5)]),
    'blender': ('🌪️ Blender / Mixer', 300, 0.2),
    'air_fryer': ('🍟 Air Fryer', 800, 0.5),
    'microwave': ('🍱 Microwave', 800, 0.2),
    
    # Kategori Pencucian & Perawatan
    'washing_machine': ('🧺 Mesin Cuci (2 Tabung)', 350, 1.5),
    'iron': ('👔 Setrika Listrik', 300, 1.5),
    'vacuum': ('🧹 Vacuum Cleaner', 400, 0.5),
    'hair_dryer': ('💨 Hair Dryer', 400, 0.2),
    
    # Kategori Iklim & Tata Air
    'water_pump': ('💧 Pompa Air (Sumur Dangkal)', 250, 1.5),
    'ac': ('❄️ AC (1/2 PK Standard)', 400, 8),
    'fan': ('🌀 Kipas Angin', 50, 4),
    'water_heater': ('🚿 Water Heater (Low Watt)', 350, 1),
    
    # Kategori Hiburan & Beban Dasar
    'tv': ('📺 Televisi (32" LED)', 50, 6),
    'pc_laptop': ('💻 Laptop / PC Desktop', 100, 8),
    'lamp_80': ('💡 Lampu Rumah Sederhana', 80, 12),
    'lamp_120': ('💡 Lampu Rumah Menengah', 120, 12),
    'lamp_180': ('💡 Lampu Rumah Modern', 180, 12),
    'wifi_router': ('📡 Router WiFi', 10, 24)
}

# [FIX Celah Kritis #1] lamp_80/120/180 merepresentasikan TIGA ASUMSI SKALA
# PENCAHAYAAN untuk sistem lampu yang SAMA di satu rumah (Sederhana/Menengah/
# Modern) -- bukan tiga inventaris lampu terpisah yang boleh menyala bareng.
# Konstanta ini dipakai di sidebar untuk saling mengunci (mutual exclusion)
# ketiga checkbox tersebut: begitu satu dicentang, dua lainnya di-disable
# (abu-abu, tidak bisa diklik) sampai yang aktif di-uncheck lagi.
LAMP_TIER_KEYS = ['lamp_80', 'lamp_120', 'lamp_180']

# Profil kepemilikan — merepresentasikan INVENTARIS rumah, bukan waktu pemakaian.
# Digunakan untuk mempercepat pengisian form tanpa harus mencentang satu-satu.
OWNERSHIP_PROFILES = {
    'Keluarga Minimalis':         ['refrigerator', 'rice_cooker', 'lamp_80', 'iron', 'fan', 'wifi_router'],
    'Keluarga Menengah':          ['refrigerator', 'rice_cooker', 'tv', 'lamp_120', 'fan', 'iron', 'water_pump', 'washing_machine', 'wifi_router'],
    'Keluarga Modern (Full)':     ['refrigerator', 'rice_cooker', 'tv', 'lamp_180', 'ac', 'iron', 'water_pump', 'dispenser', 'washing_machine', 'blender', 'wifi_router'],
    'Custom (Pilih Manual)':      []
}

# ============================================================================
# RECOMMENDATION DATABASE & COMBINATION RULES (Data-Driven)
# Rekomendasi per-peralatan & aturan kombinasi disimpan sebagai data dictionary.
# Logika pemrosesan terpisah di helper functions — scalable & maintainable.
# ============================================================================

RECOMMENDATION_DB = {
    # ── Kategori Dapur & Pemanggang ──────────────────────────────────────────
    'refrigerator': {
        'base_priority': 'MEDIUM', 'icon': '🧊',
        'action': 'Atur suhu {name} pada 3–5°C dan freezer –18°C',
        'reason_template': (
            'Suhu terlalu dingin menambah konsumsi energi secara signifikan. '
            'Pastikan karet pintu rapat dan bersihkan kondensor secara berkala.'
        ),
        'saving_method': 'TEMPERATURE', 'saving_percent': 0.05, 'saving_hours': 0,
        'category': 'cost_saving',
    },
    'dispenser': {
        'base_priority': 'HIGH', 'icon': '🥤',
        'action': 'Matikan {name} pada malam hari (22.00–06.00)',
        'reason_template': (
            'Elemen pemanas {max_power}W tetap aktif meskipun tidak digunakan. '
            'Mematikan selama 8 jam mengurangi konsumsi signifikan.'
        ),
        'saving_method': 'OFF_TIME', 'saving_hours': 8, 'saving_percent': 0,
        'category': 'cost_saving',
    },
    'rice_cooker': {
        'base_priority': 'MEDIUM', 'icon': '🍚',
        'action': 'Pindahkan nasi ke wadah terpisah setelah {name} selesai memasak',
        'reason_template': (
            'Mode Warm (50W) yang berjalan terlalu lama menyumbang porsi besar '
            'dari konsumsi harian. Memindahkan nasi segera mengeliminasi konsumsi Warm.'
        ),
        'saving_method': 'STANDBY', 'saving_percent': 0.40, 'saving_hours': 0,
        'category': 'cost_saving',
    },
    'blender': {
        'base_priority': 'LOW', 'icon': '🌪️',
        'action': 'Gunakan {name} di luar jam puncak (17.00–22.00)',
        'reason_template': (
            'Daya {max_power}W dengan pemakaian singkat; menggeser ke luar '
            'jam puncak mengurangi beban Maghrib Peak.'
        ),
        'saving_method': 'SHIFT_PEAK', 'saving_hours': 0, 'saving_percent': 0.05,
        'category': 'load_reduction',
    },
    'air_fryer': {
        'base_priority': 'HIGH', 'icon': '🍟',
        'action': 'Jangan gunakan {name} bersamaan dengan microwave',
        'reason_template': (
            'Daya {max_power}W; penggunaan simultan dengan alat berdaya tinggi '
            'lainnya dapat memicu MCB trip.'
        ),
        'saving_method': 'OFF_TIME', 'saving_hours': 0, 'saving_percent': 0,
        'category': 'overload_prevention',
    },
    'microwave': {
        'base_priority': 'LOW', 'icon': '🍱',
        'action': 'Gunakan fitur defrost/reheat {name} dengan durasi tepat',
        'reason_template': (
            'Daya {max_power}W; penggunaan efisien fitur bawaan mengurangi '
            'waktu operasi dan konsumsi energi.'
        ),
        'saving_method': 'TEMPERATURE', 'saving_percent': 0.10, 'saving_hours': 0,
        'category': 'cost_saving',
    },

    # ── Kategori Pencucian & Perawatan ───────────────────────────────────────
    'washing_machine': {
        'base_priority': 'MEDIUM', 'icon': '🧺',
        'action': 'Jalankan {name} setelah jam 22.00 atau sebelum jam 06.00',
        'reason_template': (
            'Daya {max_power}W selama {duration:.1f} jam; menggeser ke luar '
            'jam puncak mengurangi risiko overload.'
        ),
        'saving_method': 'SHIFT_PEAK', 'saving_hours': 0, 'saving_percent': 0.10,
        'category': 'load_reduction',
    },
    'iron': {
        'base_priority': 'MEDIUM', 'icon': '👔',
        'action': 'Setrika semua pakaian sekaligus dalam satu sesi ({name})',
        'reason_template': (
            'Daya {max_power}W; pemanasan berulang membuang energi. '
            'Menyetrika satu kali lebih efisien daripada beberapa kali.'
        ),
        'saving_method': 'SHIFT_PEAK', 'saving_hours': 0, 'saving_percent': 0.15,
        'category': 'load_reduction',
    },
    'vacuum': {
        'base_priority': 'LOW', 'icon': '🧹',
        'action': 'Pastikan filter {name} bersih untuk hisapan optimal',
        'reason_template': (
            'Daya {max_power}W; filter bersih mengurangi waktu '
            'pemakaian hingga 15%.'
        ),
        'saving_method': 'TEMPERATURE', 'saving_percent': 0.15, 'saving_hours': 0,
        'category': 'cost_saving',
    },
    'hair_dryer': {
        'base_priority': 'LOW', 'icon': '💨',
        'action': 'Keringkan rambut dengan handuk dulu sebelum pakai {name}',
        'reason_template': (
            'Daya {max_power}W; mengeringkan sebagian dengan handuk '
            'mempersingkat durasi penggunaan.'
        ),
        'saving_method': 'TEMPERATURE', 'saving_percent': 0.20, 'saving_hours': 0,
        'category': 'cost_saving',
    },

    # ── Kategori Iklim & Tata Air ────────────────────────────────────────────
    'water_pump': {
        'base_priority': 'MEDIUM', 'icon': '💧',
        'action': 'Isi tandon air pagi hari (06.00–08.00) dengan {name}',
        'reason_template': (
            'Daya {max_power}W; penggunaan di jam puncak Maghrib '
            'memberatkan kapasitas listrik.'
        ),
        'saving_method': 'SHIFT_PEAK', 'saving_hours': 0, 'saving_percent': 0.10,
        'category': 'load_reduction',
    },
    'ac': {
        'base_priority': 'MEDIUM', 'icon': '❄️',
        'action': 'Bersihkan filter {name} setiap 2 minggu dan pastikan suhu 24–26°C',
        'reason_template': (
            'Daya {max_power}W selama {duration:.1f} jam/hari; filter kotor '
            'menambah konsumsi 5–10%. Suhu terlalu rendah juga meningkatkan konsumsi.'
        ),
        'saving_method': 'TEMPERATURE', 'saving_percent': 0.20, 'saving_hours': 0,
        'category': 'cost_saving',
    },
    'fan': {
        'base_priority': 'LOW', 'icon': '🌀',
        'action': 'Gunakan timer atau matikan {name} saat tidur',
        'reason_template': (
            'Daya {max_power}W; kipas hanya berguna saat ada orang. '
            'Mematikan 2 jam lebih awal menghemat energi.'
        ),
        'saving_method': 'OFF_TIME', 'saving_hours': 2, 'saving_percent': 0,
        'category': 'cost_saving',
    },
    'water_heater': {
        'base_priority': 'MEDIUM', 'icon': '🚿',
        'action': 'Hidupkan {name} hanya 15–20 menit sebelum mandi',
        'reason_template': (
            'Daya {max_power}W; pemanasan berlebihan membuang energi. '
            'Gunakan di luar jam puncak.'
        ),
        'saving_method': 'OFF_TIME', 'saving_hours': 0.5, 'saving_percent': 0,
        'category': 'cost_saving',
    },

    # ── Kategori Hiburan & Beban Dasar ───────────────────────────────────────
    'tv': {
        'base_priority': 'LOW', 'icon': '📺',
        'action': 'Aktifkan sleep timer {name}, matikan total (bukan standby)',
        'reason_template': (
            'Daya {max_power}W; mode standby masih mengonsumsi daya. '
            'Matikan sepenuhnya saat tidak ditonton.'
        ),
        'saving_method': 'STANDBY', 'saving_percent': 0.05, 'saving_hours': 0,
        'category': 'cost_saving',
    },
    'pc_laptop': {
        'base_priority': 'LOW', 'icon': '💻',
        'action': 'Atur power plan ke power saver pada {name}',
        'reason_template': (
            'Daya {max_power}W selama {duration:.1f} jam; matikan monitor '
            'saat idle dan cabut charger setelah penuh.'
        ),
        'saving_method': 'STANDBY', 'saving_percent': 0.10, 'saving_hours': 0,
        'category': 'cost_saving',
    },
    'lamp_80': {
        'base_priority': 'MEDIUM', 'icon': '💡',
        'action': 'Matikan {name} yang tidak terpakai, pasang sensor gerak area luar',
        'reason_template': (
            'Total daya lampu {max_power}W; mematikan lampu tidak terpakai '
            'mengurangi konsumsi baseload.'
        ),
        'saving_method': 'OFF_TIME', 'saving_hours': 4, 'saving_percent': 0,
        'category': 'cost_saving',
    },
    'lamp_120': {
        'base_priority': 'MEDIUM', 'icon': '💡',
        'action': 'Matikan {name} yang tidak terpakai, pasang sensor gerak area luar',
        'reason_template': (
            'Total daya lampu {max_power}W; mematikan lampu tidak terpakai '
            'mengurangi konsumsi baseload.'
        ),
        'saving_method': 'OFF_TIME', 'saving_hours': 4, 'saving_percent': 0,
        'category': 'cost_saving',
    },
    'lamp_180': {
        'base_priority': 'MEDIUM', 'icon': '💡',
        'action': 'Matikan {name} yang tidak terpakai, pasang sensor gerak area luar',
        'reason_template': (
            'Total daya lampu {max_power}W; mematikan lampu tidak terpakai '
            'mengurangi konsumsi baseload.'
        ),
        'saving_method': 'OFF_TIME', 'saving_hours': 4, 'saving_percent': 0,
        'category': 'cost_saving',
    },
    'wifi_router': {
        'base_priority': 'LOW', 'icon': '📡',
        'action': 'Matikan {name} saat jam tidur (23.00–05.00)',
        'reason_template': (
            'Daya {max_power}W menyala 24 jam; mematikan 6 jam/hari '
            'mengurangi konsumsi baseload.'
        ),
        'saving_method': 'OFF_TIME', 'saving_hours': 6, 'saving_percent': 0,
        'category': 'cost_saving',
    },
}

COMBINATION_RULES = [
    {
        'pair': ('ac', 'iron'),
        'priority': 'HIGH', 'icon': '⚡',
        'action': 'Hindari menyalakan AC dan Setrika bersamaan',
        'reason': (
            # [FIX #6] Sebelumnya tertulis "Keduanya beban tinggi" -- AC
            # (400W) dan Setrika (300W) masuk kategori "sedang" (200-500W)
            # menurut calculate_realistic_peak sendiri, bukan "tinggi"
            # (>=500W). Disamakan dengan istilah yang dipakai rule ac+
            # water_heater di bawah agar konsisten dengan kategorisasi
            # Simultaneity Factor yang dijelaskan di expander UI.
            'Keduanya beban sedang-tinggi; penggunaan simultan meningkatkan '
            'risiko MCB trip. Matikan AC sebelum menyetrika.'
        ),
        'category': 'overload_prevention',
    },
    {
        'pair': ('air_fryer', 'microwave'),
        'priority': 'HIGH', 'icon': '⚡',
        'action': 'Jangan gunakan Air Fryer dan Microwave bersamaan',
        'reason': (
            'Keduanya berdaya 800W; penggunaan simultan memakan '
            '1.600W dan dapat memicu MCB trip.'
        ),
        'category': 'overload_prevention',
    },
    {
        'pair': ('washing_machine', 'water_pump'),
        'priority': 'MEDIUM', 'icon': '💧',
        'action': 'Jalankan Mesin Cuci setelah Pompa Air selesai mengisi tandon',
        'reason': (
            'Gunakan secara bergantian agar beban tidak '
            'menumpuk di satu waktu.'
        ),
        'category': 'load_reduction',
    },
    {
        'pair': ('ac', 'water_heater'),
        'priority': 'MEDIUM', 'icon': '🔄',
        'action': 'Gunakan AC dan Water Heater secara bergantian',
        'reason': (
            'Keduanya beban sedang-tinggi; penggunaan bersamaan '
            'menambah beban signifikan pada kapasitas listrik.'
        ),
        'category': 'load_reduction',
    },
]

# [FIX #4] Satu-satunya sumber label kategori rekomendasi, dipakai bersama
# oleh _build_appliance_recommendations() dan _build_combination_recommendations()
# DAN oleh semua rekomendasi "legacy" hardcoded di generate_recommendations().
# Sebelumnya ada 2 dict category_labels lokal yang tidak identik (satu di
# antaranya bahkan tidak punya entri 'cost_saving'), dan rekomendasi legacy
# tidak diberi 'category'/'triggers' sama sekali -- membuat kartu HIGH-priority
# (yang justru paling sering dilihat duluan) tampil tanpa badge & pill,
# sementara kartu sekunder tampil lengkap dengan dekorasi baru.
CATEGORY_LABELS = {
    'cost_saving': '\U0001f7e2 Penghematan Biaya',
    'load_reduction': '\U0001f7e0 Pengurangan Beban',
    'overload_prevention': '\U0001f534 Pencegahan Overload',
}

# [FIX #7] Ambang batas "daya besar" dipakai di 3 tempat berbeda dengan nilai
# berbeda. Ini BUKAN kebetulan/kelalaian -- didokumentasikan di sini sebagai
# keputusan desain yang disengaja, dengan tujuan masing-masing:
#
#   POWER_THRESHOLD_HIGH (500W)   -> dipakai calculate_realistic_peak() untuk
#                                     kategorisasi Simultaneity Factor IEC 60364
#                                     (>=500W = "tinggi", faktor puncak penuh).
#   POWER_THRESHOLD_MEDIUM (200W) -> ambang bawah kategori "sedang" (200-500W)
#                                     pada fungsi yang sama.
#   CONFLICT_DETECTION_THRESHOLD (250W) -> ambang KHUSUS untuk deteksi benturan
#                                     beban generik (Conflict Detection). Sengaja
#                                     LEBIH RENDAH dari 500W supaya sistem tetap
#                                     memperingatkan kombinasi alat "sedang" yang
#                                     jumlahnya signifikan (mis. AC 400W + alat
#                                     sedang lain), bukan cuma alat kategori
#                                     "tinggi" murni -- ini pilihan konservatif
#                                     untuk keselamatan, bukan inkonsistensi.
#
# COMBINATION_RULES sengaja TIDAK punya ambang W sama sekali karena aturannya
# berbasis pasangan alat spesifik (curated), bukan kategorisasi otomatis.
POWER_THRESHOLD_HIGH = 500
POWER_THRESHOLD_MEDIUM = 200
CONFLICT_DETECTION_THRESHOLD = 250

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

@st.cache_resource
def load_model(dataset_type):
    """Load trained model based on dataset type
    
    Supports two formats:
    1. Bundled dict: {'model': ..., 'features': [...], 'metrics': {...}, 'metadata': {...}}
    2. Legacy: separate model.joblib and features.joblib files
    """
    try:
        # Dapatkan lokasi file app.py
        current_file = Path(__file__).resolve()
        
        # Naik 2 level ke root project (H.E.M.A.T)
        project_root = current_file.parent.parent
        
        # Path ke folder trained_models
        trained_models_path = project_root / 'Model' / 'trained_models'
        
        if dataset_type == "indonesia":
            model_path = trained_models_path / 'model_id.joblib'
            features_path = trained_models_path / 'model_features_id.joblib'  # Legacy fallback
        else:  # france
            model_path = trained_models_path / 'model_fr.joblib'
            features_path = trained_models_path / 'model_fr_features.joblib'
        
        # Debug: tampilkan path yang dicoba
        if not model_path.exists():
            st.error(f"Model tidak ditemukan di: {model_path}")
            st.info(f"Current file: {current_file}")
            st.info(f"Project root: {project_root}")
            return None, None
        
        # Load the model file
        loaded_obj = joblib.load(str(model_path))
        
        # Check if it's the new bundled format (dict with 'model' and 'features' keys)
        if isinstance(loaded_obj, dict) and 'model' in loaded_obj and 'features' in loaded_obj:
            # New bundled format: extract model and features from dict
            model = loaded_obj['model']
            features = loaded_obj['features']
            
            # Optional: log metadata if available
            if 'metadata' in loaded_obj:
                metadata = loaded_obj['metadata']
                # st.info(f"Model loaded: {metadata.get('model_name', 'Unknown')}, trained on {metadata.get('training_date', 'Unknown')}")
        else:
            # Legacy format: model is directly in the file, features in separate file
            model = loaded_obj
            
            if features_path.exists():
                features = joblib.load(str(features_path))
            else:
                # Try to get features from model attributes (sklearn models)
                if hasattr(model, 'feature_names_in_'):
                    features = list(model.feature_names_in_)
                else:
                    st.warning("Features file tidak ditemukan dan model tidak memiliki feature_names_in_")
                    return None, None
        
        return model, features
        
    except Exception as e:
        st.warning(f"Error loading model: {e}")
        return None, None

def get_risk_level(utilization):
    """Get risk level based on utilization percentage - Updated for household users"""
    if utilization >= 100:
        return "BAHAYA", "risk-high", "🚨 OVERLOAD!"
    elif utilization >= 95:
        return "KRITIS", "risk-high", "🔴 KRITIS"
    elif utilization >= 85:
        return "BERAT", "risk-medium", "⚡ BEBAN BERAT"
    elif utilization >= 70:
        return "PERHATIAN", "risk-medium", "🟡 PERHATIAN"
    else:
        return "AMAN", "risk-low", "✅ AMAN"

def calculate_utilization(power_kw, va):
    """Calculate correct utilization percentage"""
    # Max usable power = VA * power factor (0.85) converted to kW
    max_power_kw = (va * 0.85) / 1000
    utilization = (power_kw / max_power_kw) * 100
    return utilization  # Tidak di-cap 100% agar status BAHAYA bisa terpicu

def calculate_realistic_peak(instant_watt, selected_appliances, appliance_dict, va=None, custom_durations=None):
    """Calculate realistic peak power with simultaneity factor.
    
    Menggunakan Simultaneity Factor standar IEC 60364 untuk memperkirakan
    beban puncak realistis. Tidak ada pemotongan paksa (capping) ke batas MCB
    agar status OVERLOAD dan CRITICAL tetap bisa terpicu sebagai peringatan.

    [FIX Celah Kritis - "Alat Hantu" / Phantom Device Bug]
    Sebelumnya fungsi ini hanya menerima `selected_appliances` (asal
    checkbox = "dimiliki"), tanpa peduli slider durasi. Akibatnya, alat yang
    slider-nya sudah digeser ke 0 jam (pengguna memutuskan TIDAK memakainya
    hari ini) tetap dihitung penuh sebagai kontributor beban puncak --
    checkbox "dimiliki" keliru diperlakukan sama dengan "sedang dipakai".
    Sekarang custom_durations diperhitungkan: alat dengan durasi persis 0
    dikecualikan dari perhitungan puncak.
    """
    custom_durations = custom_durations or {}

    # Kategori beban
    high_power = []   # >500W
    medium_power = [] # 200-500W
    low_power = []    # <200W

    for app in selected_appliances:
        if app in appliance_dict:
            # Alat yang slider durasinya di-set eksplisit ke 0 jam berarti
            # tidak dipakai hari ini -- jangan ikut jadi kontributor puncak.
            if custom_durations.get(app) == 0:
                continue

            data = appliance_dict[app]
            if isinstance(data[1], list):
                power = max([mode[1] for mode in data[1]])
            else:
                power = data[1]

            if power >= POWER_THRESHOLD_HIGH:
                high_power.append(power)
            elif power >= POWER_THRESHOLD_MEDIUM:
                medium_power.append(power)
            else:
                low_power.append(power)

    # Simultaneity factors (IEC 60364)
    high_power_peak = max(high_power) if high_power else 0
    if len(high_power) > 1:
        high_power_peak += sum(high_power[1:]) * 0.3

    medium_power_peak = sum(medium_power) * 0.7
    low_power_peak    = sum(low_power) * 0.9

    realistic_peak = high_power_peak + medium_power_peak + low_power_peak

    return realistic_peak

def get_pln_tariff(va_option_string):
    """Get PLN tariff per kWh based on specific VA category string.
    
    Membedakan 900 VA Subsidi dan Non-Subsidi sesuai Permen ESDM terbaru.
    """
    tariffs = {
        "900 VA (Subsidi)":      605,
        "900 VA (Non-Subsidi)": 1352,
        "1300 VA":             1445,
        "2200 VA":             1445
    }
    return tariffs.get(va_option_string, 1445)

def calculate_total_power(appliances, appliance_dict):
    """Calculate total power from selected appliances with multi-mode support"""
    instant_power = 0  # Daya sesaat (semua nyala)
    daily_kwh = 0  # Konsumsi harian dalam kWh
    breakdown = []
    
    for app in appliances:
        if app in appliance_dict:
            data = appliance_dict[app]
            name = data[0]
            
            # Check if multi-mode (rice cooker)
            if isinstance(data[1], list):
                # Multi-mode appliance
                modes = data[1]
                total_kwh = 0
                max_power = 0
                mode_details = []
                
                for mode_name, power, hours in modes:
                    daily_energy = (power / 1000) * hours
                    total_kwh += daily_energy
                    max_power = max(max_power, power)
                    mode_details.append(f"{mode_name} {hours}h")
                
                instant_power += max_power  # Use max power for instant calculation
                daily_kwh += total_kwh
                
                # Store with mode details
                breakdown.append((name, max_power, modes, total_kwh, True))  # True = multi-mode
            else:
                # Single-mode appliance
                power = data[1]
                hours = data[2]
                instant_power += power
                daily_energy = (power / 1000) * hours  # kWh per hari
                daily_kwh += daily_energy
                breakdown.append((name, power, hours, daily_energy, False))  # False = single-mode
    
    return instant_power, daily_kwh, breakdown

def calculate_total_power_dynamic(appliances, appliance_dict, custom_durations):
    """
    Calculate total power from selected appliances with user-defined durations.
    
    Args:
        appliances: List of appliance keys
        appliance_dict: Dictionary of appliance data
        custom_durations: Dict of {appliance_key: hours} from user sliders
    
    Returns:
        instant_power, daily_kwh, breakdown
    """
    instant_power = 0
    daily_kwh = 0
    breakdown = []
    
    for app in appliances:
        if app in appliance_dict:
            data = appliance_dict[app]
            name = data[0]
            
            # Get user-defined duration or fall back to default
            user_hours = custom_durations.get(app, None)
            
            if isinstance(data[1], list):
                # Multi-mode appliance (e.g., rice cooker)
                modes = data[1]
                max_power = max([mode[1] for mode in modes])

                if user_hours is not None:
                    # Proporsional: mode pertama (COOK) dikunci ke durasi fix-nya,
                    # sisa waktu dari slider masuk ke mode kedua (WARM).
                    # Contoh: slider 6 jam → COOK 0.5h + WARM 5.5h
                    cook_time = min(modes[0][2], user_hours)          # maks durasi cook default
                    warm_time = max(0.0, user_hours - cook_time)       # sisa masuk warm
                    cook_kwh  = (modes[0][1] / 1000) * cook_time
                    warm_kwh  = (modes[1][1] / 1000) * warm_time if len(modes) > 1 else 0.0
                    total_kwh = cook_kwh + warm_kwh
                    # Bungkus detail mode agar tabel PDF/UI bisa mem-parsingnya
                    mode_details = [(modes[0][0], modes[0][1], cook_time)]
                    if len(modes) > 1:
                        mode_details.append((modes[1][0], modes[1][1], warm_time))
                else:
                    total_kwh = sum([(mode[1] / 1000) * mode[2] for mode in modes])
                    mode_details = modes

                instant_power += max_power
                daily_kwh += total_kwh
                # Flag True = multi-mode, agar tabel menampilkan "COOK 0.5h + WARM 3.5h"
                breakdown.append((name, max_power, mode_details, total_kwh, True))
            else:
                # Single-mode appliance
                power = data[1]
                hours = user_hours if user_hours is not None else data[2]
                instant_power += power
                daily_energy = (power / 1000) * hours
                daily_kwh += daily_energy
                breakdown.append((name, power, hours, daily_energy, False))
    
    return instant_power, daily_kwh, breakdown

def get_appliance_hourly_load(app_key, data, duration, dayofweek):
    """
    Menghitung profil pemakaian daya per jam (0-23) untuk peralatan tertentu (dalam kW).
    Menggunakan algoritma Time-Anchor Circular Queue dan Sequential Phase Distribution (2 Tahap).
    Mendukung spillover energi jika melebihi jendela heuristik, serta mematuhi nameplate rating (hukum fisika).
    """
    hourly_kw = [0.0] * 24
    
    # 1. Tentukan fase-fase (Sequential Phase Distribution)
    #    Tuple: (nama_fase, max_power_kw, target_kwh, is_continuous)
    #    is_continuous=True → fase parasitik yang mengisi jam secara linear
    #    tanpa putus (tidak mematuhi window heuristik), contoh: mode WARM.
    phases = []
    if isinstance(data[1], list):
        # Multi-mode (contoh: rice cooker COOK kemudian WARM)
        modes = data[1]
        cook_default_time = modes[0][2]
        cook_time = min(cook_default_time, duration)
        warm_time = max(0.0, duration - cook_time)
        
        cook_kwh = (modes[0][1] / 1000.0) * cook_time
        phases.append(('COOK', modes[0][1] / 1000.0, cook_kwh, False))
        
        if len(modes) > 1:
            warm_kwh = (modes[1][1] / 1000.0) * warm_time
            # WARM adalah Continuous Phase: mengisi jam secara linear tanpa putus
            phases.append(('WARM', modes[1][1] / 1000.0, warm_kwh, True))
    else:
        # Single-mode
        power_kw = data[1] / 1000.0
        target_kwh = power_kw * duration
        phases.append(('SINGLE', power_kw, target_kwh, False))

    # 2. Tentukan anchor jam mulai dan jendela heuristik
    anchor = 8
    window = set(range(24))
    is_weekend = dayofweek >= 5
    
    if app_key in ['ac', 'electric_heater']:
        anchor = 18
        window = {17, 18, 19, 20, 21, 22, 23, 0, 1, 2, 3, 4, 5, 6}
    elif app_key in ['lamp', 'lamp_80', 'lamp_120', 'lamp_180']:
        anchor = 18
        window = {18, 19, 20, 21, 22, 23, 0, 1, 2, 3, 4, 5, 6}
    elif app_key in ['tv', 'computer', 'pc_laptop']:
        anchor = 17
        if is_weekend:
            window = {12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22}
        else:
            window = {17, 18, 19, 20, 21, 22}
    elif app_key in ['rice_cooker']:
        anchor = 6
        window = {6, 7, 8, 11, 12, 13, 17, 18, 19}
    elif app_key in ['blender']:
        anchor = 11
        window = {10, 11, 12, 13, 16, 17, 18}
    elif app_key in ['air_fryer']:
        anchor = 17
        window = {11, 12, 13, 17, 18, 19}
    elif app_key in ['microwave']:
        anchor = 11
        window = {6, 7, 8, 11, 12, 13, 17, 18, 19}
    elif app_key in ['washing_machine', 'laundry', 'dishwasher']:
        anchor = 9
        window = {9, 10, 11, 19, 20, 21}
    elif app_key in ['iron']:
        anchor = 9
        if is_weekend:
            window = {9, 10, 11, 12, 13, 14}
        else:
            window = set()
    elif app_key in ['vacuum']:
        anchor = 9
        window = {8, 9, 10, 11, 16, 17, 18}
    elif app_key in ['hair_dryer']:
        anchor = 6
        window = {6, 7, 8, 16, 17, 18}
    elif app_key in ['water_pump']:
        anchor = 6
        window = {5, 6, 7, 8, 9, 17, 18, 19}
    elif app_key in ['fan']:
        anchor = 12
        window = {11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22}
    elif app_key in ['water_heater']:
        anchor = 6
        window = {5, 6, 7, 8, 17, 18, 19, 20}
    elif app_key in ['kettle']:
        anchor = 7
        window = {6, 7, 8, 11, 12, 13, 17, 18, 19}
    elif app_key in ['refrigerator', 'dispenser', 'wifi_router']:
        anchor = 0
        window = set(range(24))

    # Packing menggunakan Time-Anchor Circular Queue
    current_anchor = anchor
    for phase_name, max_power_kw, energy_fase, is_continuous in phases:
        if energy_fase <= 0:
            continue
            
        circular_hours = [(current_anchor + i) % 24 for i in range(24)]
        last_allocated_hour = current_anchor
        
        if is_continuous:
            # Continuous Phase (contoh: WARM) — isi jam secara linear tanpa putus
            # dari current_anchor, tidak peduli window heuristik.
            for h in circular_hours:
                if energy_fase > 0:
                    allocation = min(max_power_kw, energy_fase)
                    hourly_kw[h] += allocation
                    energy_fase -= allocation
                    last_allocated_hour = h
        else:
            # Discrete Phase — hormati window heuristik, lalu spillover
            # 1. Alokasikan di dalam jendela heuristik
            for h in circular_hours:
                if h in window and energy_fase > 0:
                    allocation = min(max_power_kw, energy_fase)
                    hourly_kw[h] += allocation
                    energy_fase -= allocation
                    last_allocated_hour = h
                    
            # 2. Spillover ke jam-jam di luar jendela heuristik terdekat dari anchor
            if energy_fase > 0:
                for h in circular_hours:
                    if h not in window and energy_fase > 0:
                        allocation = min(max_power_kw, energy_fase)
                        hourly_kw[h] += allocation
                        energy_fase -= allocation
                        last_allocated_hour = h
                    
        # Update anchor untuk fase berikutnya ke jam sesudah alokasi terakhir
        current_anchor = (last_allocated_hour + 1) % 24
        
    return hourly_kw

def prepare_model_input_for_hour(selected_appliances, appliance_dict, va, house_type, is_indonesia, target_hour, target_dayofweek=None, target_month=None, custom_durations=None, precomputed_profiles=None):
    """
    Prepare input features for ML model prediction at a SPECIFIC hour.
    Single source of truth load calculations using get_appliance_hourly_load.
    
    Args:
        precomputed_profiles: Optional dict {app_key: [24 floats]}.
            Jika disediakan, fungsi ini melakukan O(1) lookup per alat
            alih-alih menghitung ulang profil 24 jam (menghindari O(N²)).
    """
    current_time = datetime.now()
    hour = target_hour
    dayofweek = target_dayofweek if target_dayofweek is not None else current_time.weekday()
    month = target_month if target_month is not None else 5
    
    sub1_total = 0
    sub2_total = 0
    sub3_total = 0
    
    for app_key in selected_appliances:
        if app_key in appliance_dict:
            # O(1) lookup dari cache jika tersedia, O(24) fallback jika tidak
            if precomputed_profiles and app_key in precomputed_profiles:
                app_profile = precomputed_profiles[app_key]
            else:
                duration = 24
                if custom_durations and app_key in custom_durations:
                    duration = custom_durations[app_key]
                app_profile = get_appliance_hourly_load(app_key, appliance_dict[app_key], duration, dayofweek)
            
            eff_power_kw = app_profile[target_hour]
            
            if app_key in ['refrigerator', 'dispenser', 'rice_cooker', 'blender', 'air_fryer', 'microwave', 'wifi_router', 'kettle']:
                sub1_total += eff_power_kw
            elif app_key in ['washing_machine', 'laundry', 'iron', 'vacuum', 'hair_dryer', 'dishwasher']:
                sub2_total += eff_power_kw
            elif app_key in ['water_pump', 'ac', 'fan', 'water_heater', 'electric_heater']:
                sub3_total += eff_power_kw
            elif app_key in ['tv', 'pc_laptop', 'computer', 'lamp', 'lamp_80', 'lamp_120', 'lamp_180']:
                sub1_total += eff_power_kw * 0.3
                sub3_total += eff_power_kw * 0.7
                
    total_active = sub1_total + sub2_total + sub3_total
    
    if is_indonesia:
        input_dict = {
            'Sub_metering_1': sub1_total,
            'Sub_metering_2': sub2_total,
            'Sub_metering_3': sub3_total,
            'Hour': hour,
            'Dayofweek': dayofweek,
            'Month': month,
            'Lag_1': total_active,
            'Lag_24': total_active,
            'Lag_168': total_active,
            'Rolling_mean_3': total_active,
            'Rolling_mean_24': total_active,
            'Rolling_std_24': 0.1,
            'is_weekend': 1 if dayofweek >= 5 else 0,
            'VA_900': 1 if va == 900 else 0,
            'VA_1300': 1 if va == 1300 else 0,
            'VA_2200': 1 if va == 2200 else 0,
            'House_mixed': 1 if house_type == 'Rumah Campuran' else 0,
            'House_retired': 1 if house_type == 'Rumah Pensiunan' else 0,
            'House_working_class': 1 if house_type == 'Rumah Pekerja' else 0,
            'is_maghrib_peak': 1 if 17 <= hour <= 22 else 0,
        }
    else:
        input_dict = {
            'hour': hour,
            'dayofweek': dayofweek,
            'lag_1': total_active,
            'lag_2': total_active,
            'lag_3': total_active,
            'lag_6': total_active,
            'lag_12': total_active,
            'lag_24': total_active,
            'lag_168': total_active,
            'rate_of_change': 0,
            'rolling_mean_3': total_active,
            'rolling_std_3': 0.1,
            'rolling_std_24': 0.15,
            'rolling_mean_168': total_active,
            'rolling_std_168': 0.2,
        }
        
    return input_dict

def build_heuristic_daily_profile(selected_appliances, appliance_dict, custom_durations, dayofweek):
    """
    Bangun profil daya aktif per jam (0-23) menggunakan logika heuristik dengan sinkronisasi
    durasi slider (Hukum Konservasi Energi).
    """
    profile = [0.0] * 24
    for app_key in selected_appliances:
        if app_key not in appliance_dict:
            continue
        duration = custom_durations.get(app_key, 24) if custom_durations else 24
        app_profile = get_appliance_hourly_load(app_key, appliance_dict[app_key], duration, dayofweek)
        for h in range(24):
            profile[h] += app_profile[h]
    return profile

def predict_daily_curve(model, features, selected_appliances, appliance_dict, va, house_type, is_indonesia, is_weekend=False, custom_durations=None):
    """
    Generate 24-hour load predictions using the ML model.

    UPGRADED: Sekarang menggunakan build_heuristic_daily_profile untuk mengisi
    fitur Lag dan Rolling dengan nilai historis yang berfluktuasi per jam —
    bukan nilai identik (flat). Ini membuktikan model ML merespons konteks waktu.

    Returns:
        list of 24 predictions (one per hour), or None if model unavailable
    """
    if model is None or features is None:
        return None

    dayofweek = 5 if is_weekend else 1  # Saturday vs Tuesday
    # Hardcode bulan Mei (5) — data training PELITA hanya April-Juni.
    # Mencegah ekstrapolasi buta jika sidang/demo di luar rentang training.
    month = 5

    # ── Step 1: bangun profil heuristik 24 jam ──────────────────────────────
    daily_profile = build_heuristic_daily_profile(
        selected_appliances, appliance_dict, custom_durations, dayofweek
    )

    # ── Step 1b: pre-compute per-appliance profiles (O(N) sekali) ─────────
    precomputed_profiles = {}
    for app_key in selected_appliances:
        if app_key in appliance_dict:
            duration = custom_durations.get(app_key, 24) if custom_durations else 24
            precomputed_profiles[app_key] = get_appliance_hourly_load(
                app_key, appliance_dict[app_key], duration, dayofweek
            )

    predictions = []
    for hour in range(24):
        try:
            # ── Step 2: O(1) lookup sub-meter dari cache ─────────────────────
            input_dict = prepare_model_input_for_hour(
                selected_appliances, appliance_dict, va, house_type, is_indonesia,
                target_hour=hour,
                target_dayofweek=dayofweek,
                target_month=month,
                custom_durations=custom_durations,
                precomputed_profiles=precomputed_profiles,
            )

            # ── Step 3: timpa lag & rolling dengan nilai dari daily_profile ──
            # Gunakan siklus (%) agar jam 0 bisa pakai profil jam 23 sebagai lag
            p = daily_profile  # alias pendek
            if is_indonesia:
                lag1  = p[(hour - 1) % 24]
                lag24 = p[(hour - 24) % 24]
                lag168 = p[hour % 24]          # aproksimasi: sama jam, hari lalu
                window3  = [p[(hour - i) % 24] for i in range(1, 4)]
                window24 = [p[(hour - i) % 24] for i in range(1, 25)]
                input_dict.update({
                    'Lag_1':          lag1,
                    'Lag_24':         lag24,
                    'Lag_168':        lag168,
                    'Rolling_mean_3': float(np.mean(window3)),
                    'Rolling_mean_24': float(np.mean(window24)),
                    'Rolling_std_24': float(np.std(window24)) + 0.01,
                })
            else:
                lag1  = p[(hour - 1) % 24]
                lag2  = p[(hour - 2) % 24]
                lag3  = p[(hour - 3) % 24]
                lag6  = p[(hour - 6) % 24]
                lag12 = p[(hour - 12) % 24]
                lag24 = p[(hour - 24) % 24]
                window3   = [p[(hour - i) % 24] for i in range(1, 4)]
                window24  = [p[(hour - i) % 24] for i in range(1, 25)]
                window168 = [p[hour % 24]] * 7  # aproksimasi 7-hari
                input_dict.update({
                    'lag_1':  lag1,  'lag_2':  lag2,  'lag_3':  lag3,
                    'lag_6':  lag6,  'lag_12': lag12, 'lag_24': lag24,
                    'lag_168': p[hour % 24],
                    'rate_of_change': lag1 - lag2,
                    'rolling_mean_3':   float(np.mean(window3)),
                    'rolling_std_3':    float(np.std(window3)) + 0.01,
                    'rolling_mean_24':  float(np.mean(window24)),
                    'rolling_std_24':   float(np.std(window24)) + 0.01,
                    'rolling_mean_168': float(np.mean(window168)),
                    'rolling_std_168':  float(np.std(window168)) + 0.01,
                })

            # ── Step 4: predict ──────────────────────────────────────────────
            input_df = pd.DataFrame([input_dict])
            for feat in features:
                if feat not in input_df.columns:
                    input_df[feat] = 0
            input_df = input_df[features]
            pred = float(model.predict(input_df)[0])
            predictions.append(max(pred, 0))

        except Exception as e:
            # [FIX #10] Sebelumnya "except Exception: predictions.append(0)"
            # menelan SEMUA jenis error tanpa jejak apa pun -- kalau suatu
            # saat ada bug di pembentukan fitur (mis. key hilang, tipe data
            # salah), kurva ML akan diam-diam berisi nol di jam tersebut
            # tanpa pesan error yang bisa dipakai untuk debug. Sekarang
            # dicatat ke logger (server log / konsol) sebelum tetap fallback
            # ke 0 -- perilaku curve tidak berubah (tetap tidak crash di
            # tengah rendering Streamlit), tapi kegagalannya tidak lagi bisu.
            logger.warning(f"predict_daily_curve: prediksi gagal di jam {hour:02d}:00 -> {type(e).__name__}: {e}")
            predictions.append(0)

    return predictions

# prepare_model_input has been removed as predictions are now unified through predict_daily_curve

def get_input_hash(va, house_type, scenario, appliances):
    """Generate hash of current input state"""
    return hash((va, house_type, scenario, tuple(sorted(appliances))))

# ============================================================================
# RECOMMENDATION ENGINE HELPERS
# Fungsi-fungsi pembantu untuk mesin rekomendasi data-driven.
# Memproses RECOMMENDATION_DB & COMBINATION_RULES menjadi rekomendasi personal.
# ============================================================================

def _calculate_priority_score(base_priority, risk, duration, max_power, va):
    """Hitung prioritas dinamis berdasarkan skor komposit.

    Scoring:
        Base:     HIGH=3, MEDIUM=2, LOW=1
        Risk:     OVERLOAD/CRITICAL +2, HEAVY +1, lainnya +0
        Durasi:   >=12h +2, >=8h +1
        Power/VA: >50%% kapasitas VA +1

    Hasil: 1 -> LOW, 2-3 -> MEDIUM, >=4 -> HIGH
    """
    base_scores = {'HIGH': 3, 'MEDIUM': 2, 'LOW': 1}
    score = base_scores.get(base_priority, 1)

    # Risk modifier
    if risk in ('OVERLOAD', 'CRITICAL'):
        score += 2
    elif risk == 'HEAVY':
        score += 1

    # Duration modifier
    if duration >= 12:
        score += 2
    elif duration >= 8:
        score += 1

    # Power-to-VA ratio modifier
    va_watts = va * 0.85
    if va_watts > 0 and (max_power / va_watts) > 0.5:
        score += 1

    if score >= 4:
        return 'HIGH'
    elif score >= 2:
        return 'MEDIUM'
    return 'LOW'


def _compute_trigger_reasons(app_key, duration, max_power, utilization, peak_hour, risk):
    """Hasilkan trigger reason secara dinamis berdasarkan kondisi aktual."""
    triggers = []

    if duration >= 8:
        triggers.append('Durasi tinggi')
    if max_power >= 300:
        triggers.append('Beban daya besar')
    if peak_hour is not None and 17 <= peak_hour <= 22:
        triggers.append('Jam puncak Maghrib')
    if risk in ('OVERLOAD', 'CRITICAL'):
        triggers.append('Status beban kritis')
    elif risk == 'HEAVY':
        triggers.append('Status beban berat')
    if app_key in ('refrigerator', 'dispenser', 'wifi_router'):
        triggers.append('Phantom load')
    if utilization >= 80:
        triggers.append('Utilisasi tinggi')

    return triggers if triggers else ['Optimasi umum']


def _calculate_saving_text(method, max_power, tariff, saving_hours, saving_percent, daily_kwh_app):
    """Hitung estimasi penghematan dengan bahasa akademis yang hati-hati.

    Methods:
        OFF_TIME    - penghematan dari mematikan selama N jam
        TEMPERATURE - penghematan dari optimasi suhu/efisiensi (%%)
        STANDBY     - penghematan dari eliminasi konsumsi standby (%%)
        SHIFT_PEAK  - penghematan dari penggeseran ke luar jam puncak (%%)
    """
    monthly_saving = 0

    if method == 'OFF_TIME' and saving_hours > 0:
        monthly_saving = (max_power / 1000) * saving_hours * tariff * 30
    elif method in ('TEMPERATURE', 'STANDBY', 'SHIFT_PEAK') and saving_percent > 0:
        monthly_saving = daily_kwh_app * tariff * 30 * saving_percent

    if monthly_saving > 0:
        return (
            f'Estimasi penghematan \u2248Rp {monthly_saving:,.0f}/bulan '
            f'(berdasarkan pola penggunaan saat ini)'
        )
    if method == 'OFF_TIME' and saving_hours == 0:
        return 'Mencegah MCB trip & kerusakan peralatan'
    return 'Potensi penghematan signifikan'


def _build_appliance_recommendations(
    appliances, appliance_dict, custom_durations, tariff,
    daily_kwh, utilization, risk, peak_hour, va, is_used_fn
):
    """Bangun rekomendasi personal untuk setiap peralatan yang dipakai hari ini.

    Membaca RECOMMENDATION_DB, menghitung prioritas dinamis, trigger reason,
    dan estimasi saving — lalu menghasilkan list of dict rekomendasi.
    """
    recs = []
    # [FIX #4] Sebelumnya category_labels didefinisikan lokal di sini (dan
    # sedikit berbeda dari versi di _build_combination_recommendations) --
    # sekarang pakai CATEGORY_LABELS terpusat di dekat COMBINATION_RULES.

    for app_key in appliances:
        if not is_used_fn(app_key):
            continue
        rec_data = RECOMMENDATION_DB.get(app_key)
        if rec_data is None:
            continue
        data = appliance_dict.get(app_key)
        if data is None:
            continue

        name_raw = data[0]
        # Strip emoji prefix dari nama (APPLIANCES_INDONESIA menyimpan icon di nama)
        # Contoh: '🧊 Kulkas (2 Pintu)' → 'Kulkas (2 Pintu)'
        name = name_raw.split(' ', 1)[1] if ' ' in name_raw and not name_raw[0].isalpha() else name_raw

        # Daya maksimal & durasi default
        if isinstance(data[1], list):
            max_power = max(m[1] for m in data[1])
            default_duration = sum(m[2] for m in data[1])
        else:
            max_power = data[1]
            default_duration = data[2]

        duration = custom_durations.get(app_key, default_duration)
        if duration == 0:
            continue

        # kWh harian peralatan ini
        if isinstance(data[1], list):
            modes = data[1]
            cook_time = min(modes[0][2], duration)
            warm_time = max(0.0, duration - cook_time)
            daily_kwh_app = (modes[0][1] / 1000) * cook_time
            if len(modes) > 1:
                daily_kwh_app += (modes[1][1] / 1000) * warm_time
        else:
            daily_kwh_app = (max_power / 1000) * duration

        # Prioritas dinamis (scoring komposit)
        dynamic_priority = _calculate_priority_score(
            rec_data['base_priority'], risk, duration, max_power, va
        )

        # Trigger reason dinamis
        triggers = _compute_trigger_reasons(
            app_key, duration, max_power, utilization, peak_hour, risk
        )

        # Format teks action & reason
        action = rec_data['action'].format(
            name=name, max_power=max_power, duration=duration
        )
        reason = rec_data['reason_template'].format(
            name=name, max_power=max_power, duration=duration
        )

        # Estimasi saving
        saving = _calculate_saving_text(
            rec_data['saving_method'], max_power, tariff,
            rec_data['saving_hours'], rec_data['saving_percent'],
            daily_kwh_app
        )

        recs.append({
            'priority': dynamic_priority,
            'icon': rec_data['icon'],
            'action': action,
            'reason': reason,
            'saving': saving,
            'category': CATEGORY_LABELS.get(rec_data['category'], ''),
            'triggers': triggers,
        })

    return recs


def _appliances_time_overlap(app_a, app_b, appliance_dict, custom_durations, dayofweek):
    """[FIX #5] Cek apakah dua alat BENAR-BENAR punya jam operasional yang
    tumpang tindih, memakai mesin penjadwalan yang sama dengan kurva ML
    (get_appliance_hourly_load) -- bukan cuma asumsi "kedua-duanya dipakai
    hari ini" seperti sebelumnya. Sebelumnya COMBINATION_RULES dan
    get_appliance_hourly_load adalah dua subsistem yang tidak saling
    berbicara: aturan AC+Setrika bisa menyala walau jadwal heuristik AC
    (malam) dan Setrika weekday (spillover mulai jam 9 pagi) nyaris tidak
    pernah bersinggungan.
    """
    def _duration_of(app_key):
        data = appliance_dict[app_key]
        default_duration = sum(m[2] for m in data[1]) if isinstance(data[1], list) else data[2]
        return custom_durations.get(app_key, default_duration)

    profile_a = get_appliance_hourly_load(app_a, appliance_dict[app_a], _duration_of(app_a), dayofweek)
    profile_b = get_appliance_hourly_load(app_b, appliance_dict[app_b], _duration_of(app_b), dayofweek)
    return any(profile_a[h] > 0 and profile_b[h] > 0 for h in range(24))


def _build_combination_recommendations(
    appliances, appliance_dict, custom_durations,
    utilization, risk, is_used_fn, dayofweek
):
    """Bangun rekomendasi berdasarkan aturan kombinasi antar-alat."""
    recs = []
    # [FIX #4] CATEGORY_LABELS terpusat (lihat komentar di dekat COMBINATION_RULES).

    for rule in COMBINATION_RULES:
        app_a, app_b = rule['pair']
        # Kedua alat harus ada di daftar peralatan yang dipilih DAN dipakai
        if app_a not in appliances or app_b not in appliances:
            continue
        if not (is_used_fn(app_a) and is_used_fn(app_b)):
            continue

        # [FIX #5] Cek jadwal aktual dari mesin penjadwalan yang sama
        # dengan kurva ML, bukan cuma "keduanya dipakai hari ini".
        overlaps = _appliances_time_overlap(
            app_a, app_b, appliance_dict, custom_durations, dayofweek
        )

        # Prioritas dinamis berdasarkan risk
        priority = rule['priority']
        if risk in ('OVERLOAD', 'CRITICAL'):
            priority = 'HIGH'
        elif risk == 'HEAVY' and priority == 'LOW':
            priority = 'MEDIUM'

        if overlaps:
            action = rule['action']
            reason = rule['reason']
            triggers = ['Jadwal pemakaian berpotensi tumpang tindih (estimasi)']
        else:
            # Jadwal heuristik bilang kedua alat ini BIASANYA tidak aktif
            # bersamaan -- turunkan jadi catatan pencegahan umum (bukan
            # peringatan konflik aktif), tapi JANGAN dihilangkan total:
            # pengguna tetap bisa menyalakan keduanya di luar kebiasaan,
            # jadi info keselamatannya tetap relevan sebagai jaga-jaga.
            priority = 'LOW' if priority != 'HIGH' else 'MEDIUM'
            action = f"{rule['action']} (di luar jam pemakaian biasa)"
            reason = (
                f"{rule['reason']} Catatan: berdasarkan jadwal pemakaian yang "
                f"Anda atur, kedua alat ini biasanya TIDAK aktif di jam yang "
                f"sama -- peringatan ini berlaku sebagai jaga-jaga jika Anda "
                f"menggunakannya di luar kebiasaan tersebut."
            )
            triggers = ['Jadwal biasa tidak tumpang tindih -- pencegahan umum']

        recs.append({
            'priority': priority,
            'icon': rule['icon'],
            'action': action,
            'reason': reason,
            'saving': 'Mencegah MCB trip & menjaga stabilitas listrik',
            'category': CATEGORY_LABELS.get(rule['category'], ''),
            'triggers': triggers,
        })

    return recs


def _deduplicate_recommendations(recommendations):
    """Hapus rekomendasi duplikat berdasarkan kemiripan teks action."""
    seen = set()
    unique = []
    for rec in recommendations:
        key = rec['action'].lower().strip()
        if key not in seen:
            seen.add(key)
            unique.append(rec)
    return unique


def _sort_and_limit_recommendations(recommendations):
    """Urutkan berdasarkan prioritas (HIGH->MEDIUM->LOW) dan batasi jumlah.

    Batas: HIGH = semua, MEDIUM = maks 5, LOW = maks 3.
    """
    priority_order = {'HIGH': 0, 'MEDIUM': 1, 'LOW': 2}
    sorted_recs = sorted(
        recommendations,
        key=lambda r: priority_order.get(r['priority'], 3)
    )

    result = []
    counts = {'MEDIUM': 0, 'LOW': 0}
    limits = {'MEDIUM': 5, 'LOW': 3}

    for rec in sorted_recs:
        p = rec['priority']
        if p == 'HIGH':
            result.append(rec)
        elif p in limits:
            if counts[p] < limits[p]:
                result.append(rec)
                counts[p] += 1

    return result


def generate_recommendations(power_kw, va, house_type, hour, appliances, realistic_kw, ml_curve=None, custom_durations=None, tariff=None, breakdown=None, daily_kwh=0.0, appliance_dict=None, dayofweek=1):
    """
    Generate smart, ML-aware recommendations.

    Args:
        tariff: tarif PLN dalam Rp/kWh.
        ml_curve: list of 24 float predictions from predict_daily_curve.
        custom_durations: dict {app_key: hours} dari slider pengguna.
        breakdown: list of tuples hasil kalkulasi daya dinamis.
        daily_kwh: total kwh harian dari kalkulasi dinamis.
        appliance_dict: dictionary peralatan aktif.
        dayofweek: 5 (Sabtu/simulasi weekend) atau 1 (Selasa/simulasi
            weekday) -- dipakai _build_combination_recommendations() untuk
            cek tumpang tindih jadwal lewat get_appliance_hourly_load(),
            konsisten dengan konvensi yang sama dipakai predict_daily_curve.

    [FIX #8] Parameter `dataset` yang sebelumnya ada di sini dihapus --
    dicek lewat grep, tidak pernah dipakai di dalam fungsi ini sama sekali
    (dead parameter, sisa dari versi lama yang mendukung 2 dataset).
    """
    if appliance_dict is None:
        appliance_dict = APPLIANCES_INDONESIA
    # [FIX Celah Kritis - Phantom Device Bug] Normalisasi sekali di sini.
    custom_durations = custom_durations or {}

    def _is_used_today(app_key):
        """Alat dianggap 'dipakai hari ini' kalau dicentang (dimiliki) DAN
        slider durasinya TIDAK di-set persis ke 0 jam. Dipakai di seluruh
        fungsi ini menggantikan `app_key in appliances` mentah, supaya alat
        yang sengaja dinolkan penggunanya tidak lagi memicu peringatan
        benturan/rekomendasi seolah-olah masih menyala."""
        return app_key in appliances and custom_durations.get(app_key) != 0

    recommendations = []
    utilization = calculate_utilization(realistic_kw, va)
    if tariff is None:
        tariff = 1444.70

    # ── Analisis kurva ML jika tersedia ─────────────────────────────────────
    peak_hour  = None
    peak_value = None
    min_hour   = None
    if ml_curve and any(v > 0 for v in ml_curve):
        peak_hour  = ml_curve.index(max(ml_curve))
        peak_value = max(ml_curve)
        min_hour   = ml_curve.index(min(ml_curve))

    if utilization >= 100:
        risk = "OVERLOAD"
    elif utilization >= 95:
        risk = "CRITICAL"
    elif utilization >= 85:
        risk = "HEAVY"
    elif utilization >= 70:
        risk = "MEDIUM"
    else:
        risk = "NORMAL"

    # Definisikan dictionary untuk pencarian biaya per alat dari breakdown
    biaya_per_alat = {}
    if breakdown:
        for item in breakdown:
            app_name = item[0]
            app_kwh = item[3]
            biaya_per_alat[app_name] = app_kwh * tariff * 30 # Estimasi biaya bulanan

    # ── 1. Conflict Detection (Deteksi Benturan Beban) ────────────────────────
    # [FIX Celah Kritis - Phantom Device Bug] Sebelumnya blok ini memakai
    # `appliances` mentah (checkbox = "dimiliki") tanpa cek custom_durations
    # sama sekali -- padahal parameter custom_durations sudah diterima
    # fungsi ini sejak awal, hanya tidak pernah dipakai. Akibatnya, alat
    # yang slidernya di-set ke 0 jam (user memutuskan tidak dipakai hari
    # ini) tetap dianggap "aktif bersamaan" dan memicu peringatan
    # "Benturan Beban" yang salah -- penguji bisa langsung mematahkan ini
    # saat demo ("lha kan sudah saya nolkan, kenapa masih dianggap nyala?").
    high_power_active = []
    for app in appliances:
        if app in appliance_dict:
            if not _is_used_today(app):
                continue
            data = appliance_dict[app]
            power = max([m[1] for m in data[1]]) if isinstance(data[1], list) else data[1]
            if power >= CONFLICT_DETECTION_THRESHOLD:  # lihat dokumentasi ambang di dekat COMBINATION_RULES
                high_power_active.append((data[0], power))

    high_power_active = sorted(high_power_active, key=lambda x: x[1], reverse=True)
    if len(high_power_active) >= 2 and utilization >= 80:
        alat1, daya1 = high_power_active[0]
        alat2, daya2 = high_power_active[1]
        recommendations.append({
            'priority': 'HIGH', 'icon': '⚠️',
            'action': f'Cegah Benturan Beban: {alat1} & {alat2}',
            'reason': f'Sistem mendeteksi {alat1} ({daya1}W) dan {alat2} ({daya2}W) aktif bersamaan. Penggunaan simultan memakan {daya1+daya2}W. Disarankan memberi jeda 1-2 jam.',
            'saving': 'Mencegah MCB Trip & Penurunan Umur Kabel',
            'category': CATEGORY_LABELS.get('overload_prevention', ''),
            'triggers': [f'Utilisasi {utilization:.0f}%', 'Dua alat berdaya besar aktif bersamaan'],
        })

    # ── 2. Rekomendasi berbasis kurva ML (Peak Shifting) ─────────────────────
    if peak_hour is not None and peak_value is not None:
        if _is_used_today('ac'):
            ac_name = appliance_dict['ac'][0]
            ac_cost = biaya_per_alat.get(ac_name, 0)
            saving_rp = ac_cost * 0.3  # geser AC hemat ~30% beban AC bulanan
            after_peak = (peak_hour + 2) % 24
            recommendations.append({
                'priority': 'HIGH' if peak_value > (va * 0.85 / 1000) else 'MEDIUM',
                'icon': '⏰',
                'action': f'Geser waktu menyala AC dari jam {peak_hour:02d}:00 ke jam {after_peak:02d}:00',
                'reason': (
                    f'Model ML memprediksi puncak konsumsi Anda terjadi jam {peak_hour:02d}:00 '
                    f'({peak_value:.2f} kW). Menggeser operasional AC menghindari Maghrib Peak.'
                ),
                'saving': f'~Rp {saving_rp:,.0f}/bulan',
                'category': CATEGORY_LABELS.get('load_reduction', ''),
                'triggers': [f'Jam puncak ML: {peak_hour:02d}:00', 'AC aktif hari ini'],
            })

        if min_hour is not None:
            if any(_is_used_today(a) for a in ['washing_machine', 'laundry', 'dishwasher', 'iron']):
                recommendations.append({
                    'priority': 'LOW',
                    'icon': '🔋',
                    'action': f'Jalankan mesin cuci / setrika jam {min_hour:02d}:00 (beban paling rendah)',
                    'reason': (
                        f'Model ML mendeteksi beban terendah jam {min_hour:02d}:00. '
                        f'Memindahkan peralatan berat ke jam ini menjaga beban tetap seimbang.'
                    ),
                    'saving': f'~Rp {0.15 * tariff * 30:,.0f}/bulan',
                    'category': CATEGORY_LABELS.get('load_reduction', ''),
                    'triggers': [f'Jam beban terendah ML: {min_hour:02d}:00'],
                })

    # ── 3. Rekomendasi berbasis risk level & Dynamic Savings ──────────────────
    if risk == "OVERLOAD":
        recommendations.append({
            'priority': 'HIGH', 'icon': '🚨',
            'action': 'MATIKAN peralatan daya besar SEKARANG!',
            'reason': f'Beban {utilization:.0f}% - MCB akan trip! Matikan AC atau setrika segera.',
            'saving': f'Rp {realistic_kw * 0.3 * tariff * 24:,.0f}/hari',
            'category': CATEGORY_LABELS.get('overload_prevention', ''),
            'triggers': [f'Utilisasi {utilization:.0f}% (OVERLOAD)'],
        })
        recommendations.append({
            'priority': 'HIGH', 'icon': '🔴',
            'action': f'Pertimbangkan upgrade daya ke {va*2}VA',
            'reason': f'Kapasitas {va}VA tidak cukup untuk kebutuhan Anda.',
            'saving': 'Investasi untuk kenyamanan & keamanan',
            'category': CATEGORY_LABELS.get('overload_prevention', ''),
            'triggers': ['Kapasitas VA tidak mencukupi'],
        })

    if risk in ["OVERLOAD", "CRITICAL", "HEAVY"]:
        if _is_used_today('ac'):
            ac_name = appliance_dict['ac'][0]
            ac_cost = biaya_per_alat.get(ac_name, 0)
            recommendations.append({
                'priority': 'HIGH', 'icon': '❄️',
                'action': 'Gunakan AC secara bijak',
                'reason': 'Set suhu AC 24-26°C, gunakan timer, dan matikan saat tidak di rumah.',
                'saving': f'~Rp {ac_cost * 0.2:,.0f}/bulan (hemat 20%)',
                'category': CATEGORY_LABELS.get('cost_saving', ''),
                'triggers': [f'Status beban: {risk}'],
            })

        if _is_used_today('iron'):
            iron_name = appliance_dict['iron'][0]
            iron_cost = biaya_per_alat.get(iron_name, 0)
            recommendations.append({
                'priority': 'MEDIUM', 'icon': '👔',
                'action': 'Setrika saat AC mati',
                'reason': 'Hindari menyetrika bersamaan dengan peralatan berat lainnya.',
                'saving': f'~Rp {iron_cost * 0.1:,.0f}/bulan',
                'category': CATEGORY_LABELS.get('load_reduction', ''),
                'triggers': [f'Status beban: {risk}'],
            })

        if _is_used_today('rice_cooker'):
            rc_name = appliance_dict['rice_cooker'][0]
            rc_cost = biaya_per_alat.get(rc_name, 0)
            potensi_hemat = rc_cost * 0.4  # Hemat 40% dari pengurangan durasi warm mode
            recommendations.append({
                'priority': 'MEDIUM', 'icon': '🍚',
                'action': 'Manajemen Mode Penghangat (Warm) Rice Cooker',
                'reason': 'Durasi mode Warm yang terlalu lama memakan porsi energi signifikan. Cabut steker jika nasi tinggal sedikit atau saat rumah kosong.',
                'saving': f'Potensi hemat hingga Rp {potensi_hemat:,.0f}/bulan',
                'category': CATEGORY_LABELS.get('cost_saving', ''),
                'triggers': [f'Status beban: {risk}'],
            })

    # ── 4. Baseload Profiling (Analisis Beban Dasar) ─────────────────────────
    baseload_kwh = 0
    for app in ['refrigerator', 'dispenser', 'wifi_router']:
        if _is_used_today(app):
            app_name = appliance_dict[app][0]
            app_cost = biaya_per_alat.get(app_name, 0)
            baseload_kwh += app_cost / (tariff * 30)

    if daily_kwh > 0:
        baseload_ratio = (baseload_kwh / daily_kwh) * 100
        if baseload_ratio > 30:
            recommendations.append({
                'priority': 'LOW', 'icon': '🔌',
                'action': 'Evaluasi Beban Siaga (Baseload)',
                'reason': f'Alat yang menyala 24 jam memakan porsi {baseload_ratio:.1f}% dari total energi harian.',
                'saving': 'Mereduksi Pemborosan Konstan',
                'category': CATEGORY_LABELS.get('cost_saving', ''),
                'triggers': [f'Rasio baseload {baseload_ratio:.1f}% (>30%)'],
            })

    # ── Rekomendasi berbasis pola beban (bukan jam komputer) ────────────────
    # Gunakan peak_hour dari ML curve jika ada, agar rekomendasi tetap relevan
    # meskipun demo dilakukan di pagi hari.
    effective_peak = peak_hour if peak_hour is not None else hour
    if 17 <= effective_peak <= 22:
        # Hitung daya alat berat spesifik yang membebani jam Maghrib Peak (laundry, pompa air, water heater, AC)
        deferrable_watts = 0
        deferrable_names = []
        for app_key in appliances:
            if app_key in ['washing_machine', 'laundry', 'water_pump', 'water_heater', 'ac']:
                if app_key in appliance_dict and _is_used_today(app_key):
                    data = appliance_dict[app_key]
                    power = max([m[1] for m in data[1]]) if isinstance(data[1], list) else data[1]
                    deferrable_watts += power
                    deferrable_names.append(data[0])
        
        # Jika tidak ada alat spesifik, gunakan estimasi 15% dari total daya realistis
        if deferrable_watts == 0:
            deferrable_watts = (realistic_kw * 1000) * 0.15
            
        deferrable_kw = deferrable_watts / 1000
        # Potensi hemat bulanan jika memindahkan beban ini dari beban puncak (asumsi efisiensi ~10% dari durasi 2 jam/hari)
        saving_monthly = deferrable_kw * 2 * tariff * 30 * 0.10
        
        app_list_str = f" ({', '.join(deferrable_names)})" if deferrable_names else ""
        recommendations.append({
            'priority': 'MEDIUM', 'icon': '🕐',
            'action': 'Tunda peralatan besar ke jam 22:00+',
            'reason': f'Model ML mendeteksi puncak beban Anda jatuh di rentang Maghrib Peak (17:00-22:00). Menunda penggunaan alat berat{app_list_str} ke larut malam mengurangi risiko MCB trip.',
            'saving': f'~Rp {saving_monthly:,.0f}/bulan (estimasi optimasi)',
            'category': CATEGORY_LABELS.get('load_reduction', ''),
            'triggers': [f'Jam puncak: {effective_peak:02d}:00 (Maghrib Peak)'],
        })

    if house_type == 'Rumah Pekerja':
        recommendations.append({
            'priority': 'LOW', 'icon': '⏰',
            'action': 'Gunakan timer untuk peralatan saat rumah kosong',
            'reason': 'Rumah kosong siang hari. Timer mencegah pemborosan standby.',
            'saving': f'Rp {1.0 * tariff * 30:,.0f}/bulan',
            'category': CATEGORY_LABELS.get('cost_saving', ''),
            'triggers': ['Tipe hunian: Rumah Pekerja (kosong siang hari)'],
        })

    if risk in ["NORMAL", "MEDIUM"]:
        recommendations.append({
            'priority': 'LOW', 'icon': '🔌',
            'action': 'Cabut charger & peralatan standby',
            'reason': 'Phantom load bisa 5-10% dari tagihan. Cabut TV, charger saat tidak dipakai.',
            'saving': f'Rp {0.2 * 30 * tariff:,.0f}/bulan',
            'category': CATEGORY_LABELS.get('cost_saving', ''),
            'triggers': ['Potensi phantom load'],
        })
        recommendations.append({
            'priority': 'LOW', 'icon': '💡',
            'action': 'Ganti ke lampu LED',
            'reason': 'Lampu LED 80% lebih hemat dari lampu biasa.',
            'saving': f'Rp {0.5 * tariff * 30:,.0f}/bulan per lampu',
            'category': CATEGORY_LABELS.get('cost_saving', ''),
            'triggers': ['Potensi efisiensi pencahayaan'],
        })
    # ── 5. Rekomendasi Spesifik per Peralatan (Dictionary-Driven) ────────────
    appliance_recs = _build_appliance_recommendations(
        appliances, appliance_dict, custom_durations, tariff,
        daily_kwh, utilization, risk, peak_hour, va, _is_used_today
    )
    recommendations.extend(appliance_recs)

    # ── 6. Rekomendasi Kombinasi Antar-Alat ──────────────────────────────
    combo_recs = _build_combination_recommendations(
        appliances, appliance_dict, custom_durations,
        utilization, risk, _is_used_today, dayofweek
    )
    recommendations.extend(combo_recs)

    # ── 7. Deduplikasi, Urutkan, dan Batasi ──────────────────────────────
    recommendations = _deduplicate_recommendations(recommendations)
    recommendations = _sort_and_limit_recommendations(recommendations)

    return recommendations

def generate_pdf_report(
    va, house_type, selected_appliances, appliance_dict,
    instant_watt, daily_kwh, current_kw, predicted_instant_kw,
    utilization_realistic, utilization_worst, risk_level, risk_text, daily_cost, monthly_cost,
    tariff, breakdown, recommendations, is_indonesia, selected_model,
    prediction_method, ml_prediction, realistic_peak_watt,
    va_option_string=None, is_weekend_sim=False
):
    """Generate comprehensive PDF report"""
    
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, topMargin=2*cm, bottomMargin=2*cm)
    story = []
    styles = getSampleStyleSheet()
    
    # Custom styles
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        textColor=colors.HexColor('#667eea'),
        spaceAfter=30,
        alignment=1  # Center
    )
    
    heading_style = ParagraphStyle(
        'CustomHeading',
        parent=styles['Heading2'],
        fontSize=16,
        textColor=colors.HexColor('#667eea'),
        spaceAfter=12,
        spaceBefore=20
    )
    
    normal_style = ParagraphStyle(
        'CustomNormal',
        parent=styles['Normal'],
        fontSize=10,
        spaceAfter=12
    )
    
    # Header
    story.append(Paragraph("⚡ H.E.M.A.T", title_style))
    story.append(Paragraph("Home Energy Management And Recommendation Technology", normal_style))
    # [FIX Celah Minor - Amnesia Konteks PDF] Sebelumnya laporan PDF tidak
    # menyebutkan sama sekali apakah analisis ini simulasi Weekday atau
    # Weekend, padahal toggle "Tipe Hari Simulasi" di sidebar adalah Global
    # Control yang mengubah kurva ML & rekomendasi. Pembaca PDF terpisah
    # (mis. dosen penguji) kehilangan konteks kenapa pola grafiknya begitu.
    # Sekarang ditulis eksplisit di subjudul DAN tabel Informasi Rumah.
    day_type_label = "Akhir Pekan (Weekend)" if is_weekend_sim else "Hari Kerja (Weekday)"
    story.append(Paragraph(
        f"Laporan Analisis Konsumsi Energi - {datetime.now().strftime('%d %B %Y, %H:%M')} "
        f"&mdash; Simulasi: {day_type_label}",
        normal_style
    ))
    story.append(Spacer(1, 0.5*cm))
    
    # Garis pemisah
    story.append(Paragraph("<hr/>", normal_style))
    story.append(Spacer(1, 0.3*cm))
    
    # Section 1: Informasi Rumah
    story.append(Paragraph("🏠 INFORMASI RUMAH", heading_style))
    
    house_data = [
        ['Daya Terpasang', va_option_string if va_option_string else f'{va} VA'],
        ['Tipe Hunian', house_type],
        ['Simulasi Tipe Hari', day_type_label],
        ['Tarif PLN', f'Rp {tariff:,}/kWh'],
        ['Jumlah Peralatan Aktif', f'{len(selected_appliances)} unit'],
        ['Model Prediksi', f'{selected_model} ({prediction_method})'],
        ['Dataset', 'Indonesia (PELITA)' if is_indonesia else 'Prancis (UCI)']
    ]
    
    house_table = Table(house_data, colWidths=[7*cm, 9*cm])
    house_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#f0f0f0')),
        ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
        ('TOPPADDING', (0, 0), (-1, -1), 8),
        ('GRID', (0, 0), (-1, -1), 1, colors.grey)
    ]))
    story.append(house_table)
    story.append(Spacer(1, 0.5*cm))
    
    # Section 2: Ringkasan Konsumsi
    story.append(Paragraph("⚡ RINGKASAN KONSUMSI", heading_style))
    
    consumption_data = [
        ['Daya Teoritis (Semua 100%)', f'{instant_watt:,} Watt', f'{current_kw:.3f} kW'],
        ['Daya Puncak Realistis', f'{realistic_peak_watt:,.0f} Watt', f'{realistic_peak_watt/1000:.3f} kW'],
        ['Prediksi ML Model', f'{predicted_instant_kw:.3f} kW' if ml_prediction is not None else 'N/A', 
         f'{abs(predicted_instant_kw - current_kw):.3f} kW selisih' if ml_prediction is not None else '-'],
        ['Konsumsi Harian Realistis', f'{daily_kwh:.2f} kWh', f'~{daily_kwh * 30:.1f} kWh/bulan'],
        ['Utilisasi Realistis', f'{utilization_realistic:.1f}%', risk_text],
        ['Utilisasi Worst Case', f'{utilization_worst:.1f}%', '(jika semua nyala 100%)'],
        ['Daya Aman Maksimal', f'{(va * 0.85):.0f} Watt', f'{(va * 0.85 / 1000):.2f} kW'],
    ]
    
    consumption_table = Table(consumption_data, colWidths=[6*cm, 5*cm, 5*cm])
    consumption_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#f0f0f0')),
        ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
        ('ALIGN', (1, 0), (-1, -1), 'RIGHT'),
        ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
        ('TOPPADDING', (0, 0), (-1, -1), 8),
        ('GRID', (0, 0), (-1, -1), 1, colors.grey)
    ]))
    story.append(consumption_table)
    story.append(Spacer(1, 0.5*cm))
    
    # Section 3: Estimasi Biaya
    story.append(Paragraph("💰 ESTIMASI BIAYA", heading_style))
    
    cost_data = [
        ['Biaya Harian', f'Rp {daily_cost:,.0f}'],
        ['Biaya Bulanan (30 hari)', f'Rp {monthly_cost:,.0f}'],
        ['Biaya Tahunan (365 hari)', f'Rp {daily_cost * 365:,.0f}'],
    ]
    
    cost_table = Table(cost_data, colWidths=[8*cm, 8*cm])
    cost_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#f0f0f0')),
        ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
        ('ALIGN', (1, 0), (1, -1), 'RIGHT'),
        ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
        ('FONTNAME', (1, 0), (1, -1), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 11),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 10),
        ('TOPPADDING', (0, 0), (-1, -1), 10),
        ('GRID', (0, 0), (-1, -1), 1, colors.grey),
        ('BACKGROUND', (0, 1), (-1, 1), colors.HexColor('#fff3cd')),
        ('BACKGROUND', (0, 2), (-1, 2), colors.HexColor('#d4edda')),
    ]))
    story.append(cost_table)
    story.append(Spacer(1, 0.5*cm))
    
    # Section 4: Breakdown Per Peralatan
    story.append(Paragraph("🔌 BREAKDOWN KONSUMSI PER PERALATAN", heading_style))
    
    breakdown_data = [['Peralatan', 'Daya (W)', 'Durasi', 'kWh/Hari', 'Biaya/Hari', 'Biaya/Bulan']]
    for item in breakdown:
        if item[4]:  # Multi-mode
            name, max_power, modes, total_kwh, _ = item
            mode_str = " + ".join([f"{m[0]} {m[2]}h" for m in modes])
            daily_item_cost = total_kwh * tariff
            monthly_item_cost = daily_item_cost * 30
            breakdown_data.append([
                name, 
                f'{max_power:,}', 
                mode_str,
                f'{total_kwh:.2f}',
                f'Rp {daily_item_cost:,.0f}',
                f'Rp {monthly_item_cost:,.0f}'
            ])
        else:  # Single-mode
            name, power, hours, kwh, _ = item
            daily_item_cost = kwh * tariff
            monthly_item_cost = daily_item_cost * 30
            breakdown_data.append([
                name, 
                f'{power:,}', 
                f'{hours:.1f}h',
                f'{kwh:.2f}',
                f'Rp {daily_item_cost:,.0f}',
                f'Rp {monthly_item_cost:,.0f}'
            ])
    
    # Add total row
    breakdown_data.append([
        'TOTAL',
        f'{instant_watt:,}',
        '-',
        f'{daily_kwh:.2f}',
        f'Rp {daily_cost:,.0f}',
        f'Rp {monthly_cost:,.0f}'
    ])
    
    breakdown_table = Table(breakdown_data, colWidths=[5*cm, 2*cm, 2*cm, 2*cm, 2.5*cm, 2.5*cm])
    breakdown_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#667eea')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 10),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 10),
        ('TOPPADDING', (0, 0), (-1, 0), 10),
        ('BACKGROUND', (0, 1), (-1, -2), colors.beige),
        ('BACKGROUND', (0, -1), (-1, -1), colors.HexColor('#d4edda')),
        ('FONTNAME', (0, -1), (-1, -1), 'Helvetica-Bold'),
        ('GRID', (0, 0), (-1, -1), 1, colors.grey)
    ]))
    story.append(breakdown_table)
    story.append(Spacer(1, 0.5*cm))
    
    # Page break before recommendations
    story.append(PageBreak())
    
    # Section 5: Simultaneity Factor Explanation
    story.append(Paragraph("📊 TENTANG SIMULTANEITY FACTOR", heading_style))
    story.append(Paragraph(
        f"Sistem menggunakan Simultaneity Factor berdasarkan standar IEC 60364. "
        f"Tidak semua peralatan nyala 100% bersamaan dalam kondisi normal:",
        normal_style
    ))
    
    simul_data = [
        ['Kategori', 'Simultaneity', 'Penjelasan'],
        ['Daya Tinggi (>500W)', '100% + 30%', 'Hanya 1 nyala penuh, lainnya cycling'],
        ['Daya Sedang (200-500W)', '70%', 'Sebagian besar nyala bersamaan'],
        ['Daya Rendah (<200W)', '90%', 'Hampir selalu nyala'],
    ]
    
    simul_table = Table(simul_data, colWidths=[5*cm, 4*cm, 7*cm])
    simul_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#667eea')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('GRID', (0, 0), (-1, -1), 1, colors.grey),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
    ]))
    story.append(simul_table)
    story.append(Spacer(1, 0.5*cm))
    
    story.append(Paragraph(
        f"<b>Contoh:</b> Rice cooker (400W) + Setrika (300W) jarang nyala 100% bersamaan. "
        f"Setrika memiliki thermostat (cycling on-off), rice cooker setelah matang hanya warm mode (~50W).",
        normal_style
    ))
    story.append(Spacer(1, 0.5*cm))
    
    # Section 6: Rekomendasi
    story.append(Paragraph("💡 REKOMENDASI HEMAT ENERGI", heading_style))
    story.append(Paragraph("Berikut adalah rekomendasi yang diprioritaskan berdasarkan analisis konsumsi Anda:", normal_style))
    story.append(Spacer(1, 0.3*cm))
    
    for idx, rec in enumerate(recommendations, 1):
        priority_color = {
            'HIGH': colors.HexColor('#eb3349'),
            'MEDIUM': colors.HexColor('#F2994A'),
            'LOW': colors.HexColor('#11998e')
        }.get(rec['priority'], colors.grey)
        
        rec_data = [
            [f"#{idx}", rec['action']],
            ['Prioritas:', rec['priority']],
            ['Alasan:', rec['reason']],
            ['Potensi Hemat:', rec['saving']]
        ]
        
        rec_table = Table(rec_data, colWidths=[3*cm, 13*cm])
        rec_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), priority_color),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 11),
            ('SPAN', (1, 0), (1, 0)),
            ('ALIGN', (0, 0), (0, -1), 'CENTER'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
            ('TOPPADDING', (0, 0), (-1, -1), 8),
            ('GRID', (0, 0), (-1, -1), 1, colors.grey),
            ('BACKGROUND', (0, 1), (0, -1), colors.HexColor('#f8f9fa'))
        ]))
        story.append(rec_table)
        story.append(Spacer(1, 0.3*cm))
    
    # Footer
    story.append(Spacer(1, 1*cm))
    story.append(Paragraph("<hr/>", normal_style))
    
    footer_style = ParagraphStyle(
        'Footer',
        parent=styles['Normal'],
        fontSize=8,
        textColor=colors.grey,
        alignment=1
    )
    
    story.append(Paragraph(f"Laporan dibuat oleh H.E.M.A.T", footer_style))
    story.append(Paragraph(f"© 2026 Nofal Rafif - Universitas Pamulang", footer_style))
    story.append(Paragraph(f"Disclaimer: Estimasi biaya berdasarkan asumsi pemakaian rata-rata dengan Simultaneity Factor IEC 60364. Biaya aktual dapat bervariasi.", footer_style))
    
    # Build PDF
    doc.build(story)
    buffer.seek(0)
    return buffer

# ============================================================================
# MAIN APP
# ============================================================================

def main():
    # Initialize session state for analysis and input hash
    if 'show_analysis' not in st.session_state:
        st.session_state.show_analysis = False
    if 'last_input_hash' not in st.session_state:
        st.session_state.last_input_hash = None
        
    # Initialize session state for all appliance checkboxes to avoid state mismatch/locking
    for key in APPLIANCES_INDONESIA.keys():
        if key not in st.session_state:
            st.session_state[key] = False
    
    # Header
    st.markdown('<h1 style="text-align: center; color: #667eea;">⚡H.E.M.A.T</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; color: #666;">Home Energy Management And Recommendation Technology</p>', unsafe_allow_html=True)
    
    # Tabs
    tab1, tab2, tab3 = st.tabs(["🏠 Dashboard", "📊 Analisis Model", "ℹ️ Tentang"])
    
    # ========================================================================
    # SIDEBAR - CONFIGURATION
    # ========================================================================
    with st.sidebar:
        st.markdown("## ⚙️ Konfigurasi")

        # --- Hardcoded: aplikasi fokus pada dataset Indonesia ---
        is_indonesia = True
        appliance_dict = APPLIANCES_INDONESIA.copy()
        selected_model = "ID"
        dataset = "🇮🇩 Indonesia (ID)"

        st.markdown("---")

        # House Configuration
        st.markdown("### 🏠 Data Rumah")

        va_option = st.selectbox(
            "💡 Daya Listrik (Golongan Tarif)",
            options=[
                "900 VA (Subsidi)",
                "900 VA (Non-Subsidi)",
                "1300 VA",
                "2200 VA"
            ],
            # [FIX] index=2 sebelumnya menunjuk ke "1300 VA", bukan "900 VA
            # (Non-Subsidi)" seperti yang dimaksud komentar aslinya. options[1]
            # adalah "900 VA (Non-Subsidi)" -- itu yang benar untuk index default.
            index=1,  # Default: 900 VA Non-Subsidi (paling umum)
            help="Pilih golongan tarif sesuai tagihan/struk PLN Anda. 900 VA Subsidi khusus penerima DTKS."
        )
        # Ekstrak nilai integer VA untuk perhitungan teknis
        va = int(va_option.split()[0])

        house_type = st.radio(
            "🏡 Tipe Hunian",
            options=["Rumah Pekerja", "Rumah Pensiunan", "Rumah Campuran"]
        )

        st.markdown("---")

        # Day type selector (Global Control)
        day_type = st.radio(
            "📅 Tipe Hari Simulasi",
            options=["Hari Kerja (Weekday)", "Akhir Pekan (Weekend)"],
            help="Menentukan pola konsumsi waktu (Weekday vs Weekend) untuk model prediksi ML."
        )
        is_weekend_sim = "Weekend" in day_type

        st.markdown("---")

        # Profil Kepemilikan — merepresentasikan inventaris, bukan waktu
        st.markdown("### 🏡 Profil Kepemilikan Rumah")
        profile_options = list(OWNERSHIP_PROFILES.keys())

        # Define callback function to force state updates
        def update_profile_state():
            selected = st.session_state.profile_selector
            preset = OWNERSHIP_PROFILES.get(selected, [])
            for key in appliance_dict.keys():
                st.session_state[key] = (key in preset if selected != 'Custom (Pilih Manual)' else False)

        selected_profile = st.selectbox(
            "Pilih Profil Rumah",
            options=profile_options,
            index=len(profile_options) - 1,  # Default ke 'Custom'
            key="profile_selector",
            on_change=update_profile_state,
            help="Pilih profil yang paling sesuai untuk mengisi peralatan secara otomatis."
        )

        selected_scenario = selected_profile  # alias untuk hash & badge
        
        st.markdown("### 🏠 Peralatan yang Dimiliki")
        st.caption("✓ Centang peralatan listrik yang Anda miliki")
        st.caption("💡 Sistem akan estimasi biaya berdasarkan pola pemakaian normal harian")
        
        selected_appliances = []
        cols = st.columns(2)
        
        for idx, (key, data) in enumerate(appliance_dict.items()):
            col = cols[idx % 2]
            with col:
                name = data[0]
                # Get power and default hours
                if isinstance(data[1], list):
                    power = max([mode[1] for mode in data[1]])
                else:
                    power = data[1]

                if key in LAMP_TIER_KEYS:
                    # [FIX Celah Kritis #1 - Phantom Lamp Multiplication]
                    # lamp_80/120/180 mewakili tiga ASUMSI SKALA untuk sistem
                    # lampu yang sama, bukan tiga inventaris terpisah. Kalau
                    # ketiganya boleh dicentang bebas, dayanya dijumlah
                    # (80+120+180=380W) dan memicu 3 kartu rekomendasi
                    # terpisah untuk keputusan yang seharusnya satu.
                    #
                    # Solusi: tetap checkbox (setara alat lain, sesuai
                    # preferensi desain), tapi begitu SATU tier lampu
                    # dicentang, DUA tier lainnya otomatis di-disable
                    # (abu-abu, tidak bisa diklik) sampai yang aktif itu
                    # di-uncheck lagi -- meniru perilaku radio button tanpa
                    # mengubah komponennya.
                    other_lamp_checked = any(
                        st.session_state.get(k, False)
                        for k in LAMP_TIER_KEYS if k != key
                    )
                    is_checked = st.checkbox(
                        f"{name} ({power}W)",
                        key=key,
                        disabled=other_lamp_checked,
                        help="Pilih SATU skala lampu yang paling mewakili rumah Anda. "
                             "Dua opsi lain otomatis terkunci selama ini masih dicentang."
                             if not other_lamp_checked else
                             "Terkunci karena skala lampu lain sudah dipilih. "
                             "Uncheck opsi itu dulu untuk mengganti skala lampu."
                    )
                    if is_checked:
                        selected_appliances.append(key)
                else:
                    # Streamlit automatically binds the checkbox value to st.session_state[key]
                    if st.checkbox(f"{name} ({power}W)", key=key):
                        selected_appliances.append(key)
        
        # Duration sliders section (only show if appliances selected)
        if selected_appliances:
            st.markdown("### ⏱️ Durasi Pemakaian")
            st.caption("Atur durasi pemakaian harian (jam)")
            
            custom_durations = {}
            for app_key in selected_appliances:
                if app_key in appliance_dict:
                    data = appliance_dict[app_key]
                    name = data[0]
                    
                    if isinstance(data[1], list):
                        default_hours = sum([m[2] for m in data[1]])
                        max_hours = 24.0
                    else:
                        default_hours = float(data[2])
                        max_hours = 24.0
                    
                    # Slider for each appliance
                    hours = st.slider(
                        f"{name}",
                        min_value=0.0,
                        max_value=max_hours,
                        value=default_hours,
                        step=0.5,
                        key=f"slider_{app_key}"
                    )
                    custom_durations[app_key] = hours
            
            # Store in session state for use elsewhere
            st.session_state['custom_durations'] = custom_durations
        else:
            st.session_state['custom_durations'] = {}
        
        st.markdown("---")
        st.markdown("""
        <div style="text-align: center; font-size: 0.8rem; color: #888;">
            <p>⚡ H.E.M.A.T</p>
            <p>Nofal Rafif Setiawan Reserved 2026</p>
        </div>
        """, unsafe_allow_html=True)
    
    # ========================================================================
    # TAB 1: DASHBOARD
    # ========================================================================
    with tab1:
        # Check if input changed - if yes, reset analysis
        current_input_hash = get_input_hash(va_option, house_type, selected_scenario, selected_appliances)
        if st.session_state.last_input_hash != current_input_hash:
            st.session_state.show_analysis = False
            st.session_state.last_input_hash = current_input_hash
        
        # Calculate power with custom durations from sliders
        custom_durations = st.session_state.get('custom_durations', {})
        instant_watt, daily_kwh, breakdown = calculate_total_power_dynamic(
            selected_appliances, appliance_dict, custom_durations
        )
        current_kw = instant_watt / 1000
        
        # Show model badge
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 10px; border-radius: 5px; text-align: center; color: white;">
            🤖 <strong style="color: white;">Model Aktif:</strong> Random Forest (ID) |
            📍 <strong style="color: white;">Dataset:</strong> Indonesia (PELITA) |
            ⚡ <strong style="color: white;">Tarif {va_option}:</strong> Rp {get_pln_tariff(va_option):,.0f}/kWh
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Only show metrics if there are selected appliances
        if len(selected_appliances) == 0:
            st.info("👈 Silakan pilih peralatan di sidebar untuk mulai menganalisis konsumsi energi Anda")
            return
        
        # Metrics
        current_hour = datetime.now().hour
        
        # Calculate heuristic profile for dynamic manual hourly power comparison (Apple-to-Apple)
        heuristic_profile = build_heuristic_daily_profile(
            selected_appliances, appliance_dict, custom_durations, dayofweek=5 if is_weekend_sim else 1
        )
        current_kw_heur = heuristic_profile[current_hour]
        
        # Calculate REALISTIC peak (with simultaneity)
        realistic_peak_watt = calculate_realistic_peak(instant_watt, selected_appliances, appliance_dict, va=va, custom_durations=custom_durations)
        realistic_peak_kw = realistic_peak_watt / 1000
        
        utilization_worst = calculate_utilization(current_kw, va)  # Jika SEMUA nyala
        utilization_realistic = calculate_utilization(realistic_peak_kw, va)  # Realistis
        
        risk_level, risk_class, risk_text = get_risk_level(utilization_realistic)
        
        # Calculate cost correctly based on ACTUAL daily usage
        tariff = get_pln_tariff(va_option)  # Gunakan string golongan tarif
        daily_cost = daily_kwh * tariff      # kWh harian × Rp/kWh
        monthly_cost = daily_cost * 30       # Asumsi prabayar/token (30 hari)
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <h4 style="color: #666; margin: 0;">💡 Daya Terpasang</h4>
                <h2 style="color: #667eea; margin: 10px 0;">{va} VA</h2>
                <p style="color: #999; margin: 0;">Tarif: Rp {tariff:,}/kWh</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="metric-card">
                <h4 style="color: #666; margin: 0;">⚡ Daya Puncak</h4>
                <h2 style="color: #667eea; margin: 10px 0;">{realistic_peak_kw:.2f} kW</h2>
                <p style="color: #999; margin: 0;">{realistic_peak_watt:,.0f}W (realistis) | {instant_watt:,}W (teoritis)</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div class="metric-card">
                <h4 style="color: #666; margin: 0;">💰 Estimasi Biaya</h4>
                <h2 style="color: #667eea; margin: 10px 0;">Rp {daily_cost:,.0f}/hari</h2>
                <p style="color: #999; margin: 0;">~Rp {monthly_cost:,.0f}/bulan ({daily_kwh:.2f} kWh/hari)</p>
                <p style="color: #bbb; margin: 2px 0; font-size: 0.78rem;">*Asumsi Prabayar/Token</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            # Determine display message based on risk level
            if risk_level == "BAHAYA":
                status_msg = "OVERLOAD - Risiko Trip!"
            elif risk_level == "KRITIS":
                status_msg = "Hampir Penuh"
            elif risk_level == "BERAT":
                status_msg = "Beban Tinggi"
            elif risk_level == "PERHATIAN":
                status_msg = "Beban Sedang"
            else:
                status_msg = "Beban Normal"
            
            st.markdown(f"""
            <div class="metric-card {risk_class}">
                <h4 style="margin: 0;">Status Beban</h4>
                <h2 style="margin: 10px 0;">{risk_text}</h2>
                <p style="margin: 0;">{status_msg}</p>
                <p style="margin: 5px 0; font-size: 0.85rem;">{utilization_realistic:.1f}% kapasitas</p>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Analysis button
        col_btn1, col_btn2, col_btn3 = st.columns([1, 1, 1])
        with col_btn2:
            if st.button("🔍 Lakukan Analisis", use_container_width=True, type="primary"):
                st.session_state.show_analysis = True
                st.rerun()
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Show analysis only if button clicked
        if st.session_state.show_analysis:
            # Load model
            dataset_type = "indonesia" if is_indonesia else "france"
            model, features = load_model(dataset_type)
            
            ml_prediction = None
            ml_curve = None
            prediction_method = "Perhitungan Manual"
            
            if model is not None and features is not None:
                with st.spinner('🔮 Menganalisis pola 24 jam dengan ML model...'):
                    import time
                    ml_curve = predict_daily_curve(
                        model, features, selected_appliances, appliance_dict,
                        va, house_type, is_indonesia, is_weekend=is_weekend_sim,
                        custom_durations=st.session_state.get('custom_durations', {})
                    )
                    
                    if ml_curve and any(v > 0 for v in ml_curve):
                        # Ambil prediksi spesifik untuk jam SAAT INI langsung dari kurva
                        ml_prediction = ml_curve[current_hour]
                        prediction_method = f"Machine Learning Model ({selected_model})"
                        
                    # Add delay to show processing
                    time.sleep(0.8)
            
            # Use ML prediction if available, else use calculated heuristic
            predicted_instant_kw = ml_prediction if ml_prediction is not None else current_kw_heur
            
            # Analysis Logic Section
            st.markdown("### 🔍 Analisis Sistem")
            
            # Show prediction comparison
            if ml_prediction is not None:
                col_pred1, col_pred2 = st.columns(2)
                with col_pred1:
                    st.metric(
                        "🤖 Prediksi ML Model (Jam Ini)", 
                        f"{predicted_instant_kw:.3f} kW",
                        help="Prediksi konsumsi daya realistis dari model ML pada jam berjalan"
                    )
                with col_pred2:
                    st.metric(
                        "🧮 Manual Dinamis (Jam Ini)", 
                        f"{current_kw_heur:.3f} kW",
                        help="Estimasi manual konsumsi daya berdasarkan jadwal aktif peralatan pada jam berjalan"
                    )
                
                # Show difference
                diff_percent = abs(predicted_instant_kw - current_kw_heur) / max(current_kw_heur, 0.001) * 100
                if diff_percent < 15:
                    st.success(f"✅ Model ML dan estimasi manual jam ini sangat sesuai (selisih {diff_percent:.1f}%)")
                else:
                    st.info(f"ℹ️ Selisih {diff_percent:.1f}% — Model ML menyesuaikan dengan faktor temporal eksternal & lag konsumsi")
                
                st.caption("⚠️ Catatan: Perbandingan di atas bersifat Apple-to-Apple pada jam berjalan, bukan membandingkan total seluruh daya peralatan rumah.")
            
            # ==================================================================
            # 📈 DAILY LOAD CURVE - The KEY ML Feature
            # ==================================================================
            st.markdown("### 📈 Prediksi Konsumsi 24 Jam (ML)")
            st.caption("Grafik ini menunjukkan bagaimana model ML memprediksi pola konsumsi Anda sepanjang hari")
            
            if ml_curve and any(v > 0 for v in ml_curve):
                # Static line (flat calculation for comparison)
                static_line = [current_kw] * 24
                
                hours = list(range(24))
                hour_labels = [f"{h:02d}:00" for h in hours]
                
                # Create the chart
                fig_curve = go.Figure()
                
                # ML Prediction curve
                fig_curve.add_trace(go.Scatter(
                    x=hour_labels,
                    y=ml_curve,
                    mode='lines+markers',
                    name='Prediksi ML',
                    line=dict(color='#667eea', width=3),
                    marker=dict(size=6),
                    fill='tozeroy',
                    fillcolor='rgba(102, 126, 234, 0.2)'
                ))
                
                # Static calculation line
                fig_curve.add_trace(go.Scatter(
                    x=hour_labels,
                    y=static_line,
                    mode='lines',
                    name='Kalkulator Statis',
                    line=dict(color='#888', width=2, dash='dash')
                ))
                
                # Capacity limit line
                max_capacity = (va * 0.85) / 1000
                fig_curve.add_trace(go.Scatter(
                    x=hour_labels,
                    y=[max_capacity] * 24,
                    mode='lines',
                    name=f'Batas Aman ({va}VA)',
                    line=dict(color='#eb3349', width=2, dash='dot')
                ))
                
                # Highlight peak hours (17:00-22:00)
                fig_curve.add_vrect(
                    x0="17:00", x1="22:00",
                    fillcolor="rgba(235, 51, 73, 0.1)",
                    layer="below",
                    line_width=0,
                    annotation_text="Peak Hours",
                    annotation_position="top left"
                )
                
                fig_curve.update_layout(
                    title=f"Pola Konsumsi Harian - {'Weekend' if is_weekend_sim else 'Weekday'}",
                    xaxis_title="Jam",
                    yaxis_title="Konsumsi (kW)",
                    height=400,
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                    hovermode='x unified'
                )
                
                st.plotly_chart(fig_curve, use_container_width=True)
                
                # Key insight box
                peak_hour = ml_curve.index(max(ml_curve))
                min_hour = ml_curve.index(min(ml_curve))
                peak_value = max(ml_curve)
                
                st.markdown(f"""
                <div style="background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%); 
                            padding: 15px; border-radius: 10px; color: white;">
                    <strong style="color: white;">🎯 Insight dari Model ML:</strong><br>
                    • Jam puncak diprediksi: <strong>{peak_hour:02d}:00</strong> ({peak_value:.2f} kW)<br>
                    • Jam paling efisien: <strong>{min_hour:02d}:00</strong><br>
                    • Ini berbeda dari kalkulator biasa yang menganggap beban konstan sepanjang hari.
                </div>
                """, unsafe_allow_html=True)
            else:
                st.info("📊 Model ML tidak tersedia atau mengembalikan prediksi kosong. Menggunakan estimasi daya statis.")
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            # Analysis box dengan native Streamlit (bukan HTML)
            st.markdown("### 📝 Evaluasi Rumah Tangga")
            
            # Create columns for better layout
            analysis_col1, analysis_col2 = st.columns([1, 1])
            
            with analysis_col1:
                st.markdown(f"""
                <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                            padding: 20px; border-radius: 10px; color: white; margin-bottom: 10px;">
                    <h4 style="color: white; margin-top: 0;">📊 Peralatan & Konsumsi</h4>
                    <p style="color: white; margin: 5px 0;">✓ Peralatan Dimiliki: <strong>{len(selected_appliances)} item</strong></p>
                    <p style="color: white; margin: 5px 0;">✓ Konsumsi Harian: <strong>{daily_kwh:.2f} kWh/hari</strong></p>
                    <p style="color: white; margin: 5px 0;">✓ Biaya Harian: <strong>Rp {daily_cost:,.0f}</strong></p>
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown(f"""
                <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                            padding: 20px; border-radius: 10px; color: white;">
                    <h4 style="color: white; margin-top: 0;">⚡ Beban Listrik</h4>
                    <p style="color: white; margin: 5px 0;">✓ Daya Teoritis: <strong>{instant_watt}W</strong></p>
                    <p style="color: white; margin: 5px 0; font-size: 0.85rem;">&nbsp;&nbsp;&nbsp;(jika semua nyala 100%)</p>
                    <p style="color: white; margin: 5px 0;">✓ Daya Puncak Realistis: <strong>{realistic_peak_watt:.0f}W</strong></p>
                    <p style="color: white; margin: 5px 0;">✓ Kapasitas Aman: <strong>{(va * 0.85):.0f}W</strong></p>
                    <p style="color: white; margin: 5px 0;">✓ Utilisasi: <strong>{utilization_realistic:.1f}%</strong> dari {va}VA</p>
                    <p style="color: white; margin: 5px 0;">✓ Status: <strong>{risk_text}</strong></p>
                </div>
                """, unsafe_allow_html=True)
            
            with analysis_col2:
                st.markdown(f"""
                <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                            padding: 20px; border-radius: 10px; color: white; margin-bottom: 10px;">
                    <h4 style="color: white; margin-top: 0;">🔧 Detail Teknis</h4>
                    <p style="color: white; margin: 5px 0;">✓ Metode: <strong>{prediction_method}</strong></p>
                    <p style="color: white; margin: 5px 0;">✓ Waktu: <strong>{current_hour}:00</strong> ({'Maghrib peak' if 17 <= current_hour <= 22 else 'Off-peak'})</p>
                    <p style="color: white; margin: 5px 0;">✓ Tipe Rumah: <strong>{house_type}</strong></p>
                    <p style="color: white; margin: 5px 0;">✓ Tarif PLN: <strong>Rp {tariff:,}/kWh</strong></p>
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown("""
                <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                            padding: 20px; border-radius: 10px; color: white;">
                    <h4 style="color: white; margin-top: 0;">💡 Catatan Penting</h4>
                    <p style="color: white; margin: 5px 0; font-size: 0.9rem;">• Peralatan dicentang = yang DIMILIKI</p>
                    <p style="color: white; margin: 5px 0; font-size: 0.9rem;">• Sistem hitung pola pemakaian normal</p>
                    <p style="color: white; margin: 5px 0; font-size: 0.9rem;">• Simultaneity Factor aktif</p>
                    <p style="color: white; margin: 5px 0; font-size: 0.9rem;">• Proyeksi: Harian × 30/365</p>
                </div>
                """, unsafe_allow_html=True)
            
            # Main content
            col_left, col_right = st.columns([2, 1])
            
            with col_left:
                # Power breakdown
                if breakdown:
                    st.markdown("### 🔌 Breakdown Konsumsi")
                    
                    # Build dataframe with multi-mode support
                    breakdown_rows = []
                    for item in breakdown:
                        if item[4]:  # Multi-mode
                            name, max_power, modes, total_kwh, _ = item
                            # Main row
                            mode_str = " + ".join([f"{m[0]} {m[2]}h" for m in modes])
                            breakdown_rows.append({
                                'Peralatan': f"{name} ({mode_str})",
                                'Daya (W)': max_power,
                                'kWh/Hari': total_kwh,
                                'Biaya/Hari (Rp)': int(total_kwh * tariff),
                                'Biaya/Bulan (Rp)': int(total_kwh * tariff * 30)
                            })
                        else:  # Single-mode
                            name, power, hours, kwh, _ = item
                            breakdown_rows.append({
                                'Peralatan': name,
                                'Daya (W)': power,
                                'kWh/Hari': kwh,
                                'Biaya/Hari (Rp)': int(kwh * tariff),
                                'Biaya/Bulan (Rp)': int(kwh * tariff * 30)
                            })
                    
                    breakdown_df = pd.DataFrame(breakdown_rows)
                    st.dataframe(breakdown_df, use_container_width=True, hide_index=True)
                    
                    # Pie chart - by daily cost
                    pie_labels = []
                    pie_values = []
                    for item in breakdown:
                        if item[4]:  # Multi-mode
                            name, _, _, total_kwh, _ = item
                            pie_labels.append(name)
                            pie_values.append(total_kwh * tariff)
                        else:  # Single-mode
                            name, _, _, kwh, _ = item
                            pie_labels.append(name)
                            pie_values.append(kwh * tariff)
                    
                    fig_pie = go.Figure(data=[go.Pie(
                        labels=pie_labels,
                        values=pie_values,
                        hole=0.4,
                        textinfo='label+percent',
                        hovertemplate='%{label}<br>Rp %{value:,.0f}/hari<extra></extra>'
                    )])
                    fig_pie.update_layout(
                        title="Distribusi Biaya Harian per Peralatan",
                        height=350, 
                        margin=dict(l=20, r=20, t=40, b=20)
                    )
                    st.plotly_chart(fig_pie, use_container_width=True)
            
            with col_right:
                st.markdown("### 💡 Rekomendasi Hemat")
                recommendations = generate_recommendations(
                    current_kw, va, house_type, current_hour,
                    selected_appliances, realistic_peak_kw,
                    ml_curve=ml_curve,
                    custom_durations=st.session_state.get('custom_durations', {}),
                    tariff=tariff,
                    breakdown=breakdown,
                    daily_kwh=daily_kwh,
                    appliance_dict=appliance_dict,
                    dayofweek=5 if is_weekend_sim else 1
                )
                
                for rec in recommendations:
                    priority_class = f"rec-{rec['priority'].lower()}"

                    # Category badge (jika tersedia dari engine baru)
                    cat = rec.get('category', '')
                    category_html = (
                        f'<span style="font-size:0.73rem;opacity:0.85;">'
                        f'{cat}</span><br>'
                    ) if cat else ''

                    # Trigger reason pills (jika tersedia dari engine baru)
                    trigger_html = ''
                    triggers = rec.get('triggers')
                    if triggers:
                        pills = ' '.join(
                            f'<span style="background:rgba(0,0,0,0.06);'
                            f'padding:2px 8px;border-radius:10px;'
                            f'font-size:0.72rem;color:#666;'
                            f'margin-right:4px;">\u2713 {t}</span>'
                            for t in triggers
                        )
                        trigger_html = f'<div style="margin-top:8px;">{pills}</div>'

                    # Build HTML tanpa indentasi multiline agar Streamlit
                    # tidak memperlakukannya sebagai code block markdown
                    card_html = (
                        f'<div class="rec-card {priority_class}">'
                        f'{category_html}'
                        f'<strong style="color:#2c3e50 !important;">'
                        f'{rec["icon"]} {rec["action"]}</strong>'
                        f'<p style="margin:8px 0 5px 0;color:#555 !important;'
                        f'font-size:0.95rem;">{rec["reason"]}</p>'
                        f'<p style="margin:5px 0 0 0;color:#11998e !important;'
                        f'font-weight:bold;font-size:1rem;">'
                        f'\U0001f4b0 {rec["saving"]}</p>'
                        f'{trigger_html}'
                        f'</div>'
                    )
                    st.markdown(card_html, unsafe_allow_html=True)
            
            # Penjelasan tambahan berdasarkan risk level
            if risk_level == "BAHAYA":
                st.error(f"""
                🚨 **OVERLOAD - MCB AKAN TRIP KAPAN SAJA!**

                **Beban Anda: {utilization_realistic:.1f}% dari kapasitas {va}VA**

                ⚠️ **TINDAKAN SEGERA DIPERLUKAN:**
                - MATIKAN AC atau heater sekarang
                - MATIKAN setrika atau rice cooker
                - Gunakan peralatan daya besar secara bergantian
                - Pertimbangkan upgrade daya listrik ke {va*2}VA

                💡 **Kenapa berbahaya?**
                Beban di atas 100% akan membuat MCB trip (listrik padam otomatis).
                """)
            elif risk_level == "KRITIS":
                st.warning(f"""
                🔴 **KRITIS - HAMPIR PENUH!**

                **Beban Anda: {utilization_realistic:.1f}% dari kapasitas {va}VA**

                ⚠️ Hindari menyalakan peralatan besar tambahan — lonjakan kecil bisa MCB trip.
                💡 Sisa margin: {max(0, 100-utilization_realistic):.1f}% — sangat sedikit!
                """)
            elif risk_level == "BERAT":
                st.info(f"""
                ⚡ **BEBAN BERAT - PERLU PERHATIAN**

                **Beban Anda: {utilization_realistic:.1f}% dari kapasitas {va}VA**

                ✓ Masih aman — jangan tambah peralatan daya besar (>500W) bersamaan.
                📈 Pertimbangkan upgrade ke {va*1.5:.0f}VA untuk kenyamanan lebih.
                """) 
            elif risk_level == "PERHATIAN":
                # Khusus 900/1300 VA dengan interlocking aktif — tampilkan status yang lebih informatif
                if va <= 1300:
                    st.info(f"""
                    🔄 **MANAJEMEN BEBAN AKTIF**

                    **Utilisasi realistis: {utilization_realistic:.1f}%** (setelah Interlocking Factor diterapkan)

                    Sistem mendeteksi bahwa jika seluruh peralatan Anda nyala bersamaan, beban melebihi
                    kapasitas {va}VA. Namun dalam praktik, pengguna daya rendah secara alami
                    **menggunakan peralatan secara bergantian** — itulah Interlocking Factor.

                    ✓ Selama Anda tidak nyalakan AC + Rice Cooker + Setrika bersamaan, kondisi aman.
                    💡 Tips: Matikan AC dulu sebelum menyalakan setrika atau pompa air.
                    """)
                else:
                    st.info(f"""
                    🟡 **BEBAN SEDANG**

                    **Beban Anda: {utilization_realistic:.1f}% dari kapasitas {va}VA**

                    ✓ Kondisi normal. Sisa margin: {max(0, 100-utilization_realistic):.1f}%
                    💡 Tips: Gunakan AC dan rice cooker/setrika secara bergantian.
                    """)
            else:
                st.success(f"""
                ✅ **AMAN - BEBAN NORMAL**

                **Beban Anda: {utilization_realistic:.1f}% dari kapasitas {va}VA**

                💚 Beban listrik normal dan aman.
                ✓ Sisa margin: {max(0, 100-utilization_realistic):.1f}% untuk peralatan tambahan.
                """)
            
            # Info tambahan tentang simultaneity
            with st.expander("📊 Detail Perhitungan Beban"):
                st.markdown(f"""
                ### Bagaimana Sistem Menghitung Beban?
                
                **1. Daya Teoritis (jika semua 100% bersamaan):**
                - Total: {instant_watt}W
                - Utilisasi: {utilization_worst:.1f}%
                
                **2. Daya Puncak Realistis (dengan Simultaneity Factor):**
                - Total: {realistic_peak_watt:.0f}W  
                - Utilisasi: {utilization_realistic:.1f}% ← **Yang digunakan untuk status**
                
                **3. Kenapa ada perbedaan?**
                
                Sistem menggunakan **Simultaneity Factor** berdasarkan standar IEC 60364:
                - **Peralatan daya tinggi (>500W):** Hanya 1 nyala penuh, lainnya cycling (30%)
                  - Contoh: AC tidak full 900W terus (compressor cycling on-off)
                - **Peralatan daya sedang (200-500W):** 70% bersamaan
                  - Contoh: Rice cooker dan pompa jarang nyala 100% bersamaan
                - **Peralatan daya rendah (<200W):** 90% bersamaan
                  - Contoh: Lampu, kipas kecil hampir selalu nyala
                
                **Analogi:**
                Seperti jalan tol dengan kapasitas 1000 mobil. Meski ada 1500 mobil terdaftar (dimiliki),
                tidak semua masuk bersamaan. Yang masuk sekaligus ~800 mobil (realistis).
                
                💡 **Catatan:** Peralatan yang Anda centang = peralatan yang DIMILIKI, 
                bukan yang sedang nyala sekarang. Sistem menghitung pola pemakaian normal harian.
                """)


            
            # Export PDF button
            st.markdown("---")
            col_export1, col_export2, col_export3 = st.columns([1, 1, 1])
            with col_export2:
                try:
                    # Generate PDF
                    pdf_buffer = generate_pdf_report(
                        va=va,
                        house_type=house_type,
                        selected_appliances=selected_appliances,
                        appliance_dict=appliance_dict,
                        instant_watt=instant_watt,
                        daily_kwh=daily_kwh,
                        current_kw=current_kw,
                        predicted_instant_kw=predicted_instant_kw,
                        utilization_realistic=utilization_realistic,
                        utilization_worst=utilization_worst,
                        risk_level=risk_level,
                        risk_text=risk_text,
                        daily_cost=daily_cost,
                        monthly_cost=monthly_cost,
                        tariff=tariff,
                        breakdown=breakdown,
                        recommendations=recommendations,
                        is_indonesia=is_indonesia,
                        selected_model=selected_model,
                        prediction_method=prediction_method,
                        ml_prediction=ml_prediction,
                        realistic_peak_watt=realistic_peak_watt,
                        va_option_string=va_option,
                        is_weekend_sim=is_weekend_sim
                    )
                    
                    # Download button
                    filename = f"HEMAT_Laporan_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
                    st.download_button(
                        label="📄 Export Laporan PDF",
                        data=pdf_buffer,
                        file_name=filename,
                        mime="application/pdf",
                        use_container_width=True,
                        type="primary"
                    )
                except ImportError:
                    st.error("⚠️ Library ReportLab belum terinstall. Jalankan: pip install reportlab")
                except Exception as e:
                    st.error(f"⚠️ Error membuat PDF: {str(e)}")
                    st.info("Pastikan library reportlab sudah terinstall dengan benar.")
    
    # ========================================================================
    # TAB 2: MODEL ANALYSIS
    # ========================================================================
    with tab2:
        st.markdown("## 🤖 Performa Model Machine Learning")
        st.caption("Metrik berikut diambil dari hasil evaluasi di notebook (test set, bukan training set).")
        
        model_col1, model_col2 = st.columns(2)
        
        with model_col1:
            st.markdown("""
            <div class="metric-card" style="border-top: 4px solid #667eea;">
                <h3 style="color: #667eea;">🔷 Model FR</h3>
                <p style="color: #666;">UCI IHEPC Dataset (Prancis)</p>
                <hr>
                <p><strong>R² Score:</strong> 0.6019</p>
                <p><strong>MAE:</strong> 0.3286 kW</p>
                <p><strong>RMSE:</strong> 0.4779 kW</p>
                <p><strong>Algoritma:</strong> Random Forest Regressor</p>
                <p><strong>Fitur:</strong> 16 (temporal + lag + rolling)</p>
                <p style="color: #667eea;">📌 Dataset benchmark internasional</p>
            </div>
            """, unsafe_allow_html=True)
        
        with model_col2:
            st.markdown("""
            <div class="metric-card" style="border-top: 4px solid #764ba2;">
                <h3 style="color: #764ba2;">🔶 Model ID</h3>
                <p style="color: #666;">PELITA Dataset (Indonesia)</p>
                <hr>
                <p><strong>R² Score:</strong> 0.8156</p>
                <p><strong>MAE:</strong> 0.1119 kW</p>
                <p><strong>RMSE:</strong> 0.1431 kW</p>
                <p><strong>Algoritma:</strong> Random Forest Regressor</p>
                <!-- [FIX] Sebelumnya tertulis "19" -- sudah tidak sinkron
                     setelah is_maghrib_peak ditambahkan ke input_dict di
                     prepare_model_input_for_hour (total sekarang 20 key:
                     Sub_metering x3, Hour, Dayofweek, Month, Lag x3,
                     Rolling x3, is_weekend, VA x3, House x3,
                     is_maghrib_peak). CATATAN PENTING: angka ini mengikuti
                     apa yang DIBANGUN oleh kode. Kalau model_id.joblib yang
                     dipakai belum dilatih ulang dengan is_maghrib_peak
                     sebagai fitur training, baris `input_df[features]`
                     akan diam-diam membuang key ini (tidak error), dan
                     fitur ini efektif tidak berpengaruh ke prediksi
                     walau tampil "20" di sini. Verifikasi len(features)
                     dari bundle model sebelum sidang untuk memastikan. -->
                <p><strong>Fitur:</strong> 20 (sub-metering + temporal + lag + rolling + is_maghrib_peak)</p>
                <p style="color: #764ba2;">📌 Dataset sintetik rumah tangga Indonesia</p>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        st.markdown("### 📝 Catatan Metodologi")
        st.markdown("""
        - **Model FR** menggunakan `closed='left'` pada rolling window untuk mencegah data leakage.
          R² 0.60 adalah performa bersih tanpa kebocoran informasi target.
        - **Model ID** menggunakan temporal split (bukan positional split) untuk validasi yang benar
          pada data multi-household.
        - Lag dan Rolling features di aplikasi ini adalah **aproksimasi** dari daya sesaat,
          bukan data historis nyata. Prediksi di aplikasi bersifat estimasi.
        """)
    
    # ========================================================================
    # TAB 3: ABOUT
    # ========================================================================
    with tab3:
        st.markdown("## ℹ️ Tentang HEMAT")
        st.markdown("""
        **H.E.M.A.T** (Home Energy Management And Recommendation Technology) adalah sistem prediksi 
        konsumsi energi listrik berbasis Machine Learning.
        
        ### 🌟 Fitur Utama
        - Prediksi menggunakan model Machine Learning Random Forest Regressor
        - Preset scenario untuk kemudahan
        - Analisis logika yang transparan
        - Rekomendasi prioritas dengan estimasi penghematan 
        - Breakdown biaya per peralatan
        
        ### 📐 Formula Perhitungan
        - **Utilisasi:** (Daya Konsumsi / (VA × 0.85)) × 100%
        - **Biaya Harian:** Daya (kW) × 24 jam × Tarif (Rp/kWh)
        - **Power Factor:** 0.85 (standar PLN)
        
        ### 👨‍💻 Pengembang
        **Nama:** Nofal Rafif Setiawan  
        **NIM:** 221011402613  
        **Universitas:** Universitas Pamulang  
        **Tahun:** 2026
        """)

if __name__ == "__main__":
    main()