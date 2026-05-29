"""
🏠 H.E.M.A.T - Home Energy Management and Recommendation Technology
========================================================

Enhanced version with:
- Dataset selection (France vs Indonesia)
- Model selection (FR and ID)
- Preset scenarios
- Clear analysis logic
- PDF export
- Analysis button to trigger breakdown

Author: Nofal Rafif
"""

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
        animation: pulse 2s infinite;
    }
    .risk-high h4, .risk-high h2, .risk-high p {
        color: white !important;
    }
    @keyframes pulse {
        0% { box-shadow: 0 0 0 0 rgba(235, 51, 73, 0.7); }
        70% { box-shadow: 0 0 0 10px rgba(235, 51, 73, 0); }
        100% { box-shadow: 0 0 0 0 rgba(235, 51, 73, 0); }
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
    'rice_cooker': ('🍚 Rice Cooker', [('COOK', 350, 0.5), ('WARM', 50, 3.5)]),  # Multi-mode
    'fan': ('🌀 Kipas Angin', 50, 2),  # 2 jam (pas kepanasan)
    'lamp': ('💡 Lampu LED', 20, 12),  # 12 jam (18:00-06:00)
    'water_pump': ('💧 Pompa Air', 250, 1),  # 1 jam
    'dispenser': ('🥤 Dispenser', 350, 24),  # 24 jam standby
    'kettle': ('☕ Teko Listrik', 1500, 0.5),  # 30 menit total
    'refrigerator': ('🧊 Kulkas', 100, 24),  # 24 jam always on
    'tv': ('📺 TV', 100, 6),  # 6 jam
    'ac': ('❄️ AC', 900, 8),  # 8 jam
    'iron': ('👔 Setrika', 300, 1)  # 1 jam
}

PRESET_SCENARIOS = {
    'Pagi Hari (06:00-10:00)': ['refrigerator', 'rice_cooker', 'tv', 'lamp'],
    'Siang Hari (10:00-17:00)': ['refrigerator', 'ac', 'fan', 'water_pump'],
    'Malam Hari (17:00-22:00)': ['refrigerator', 'ac', 'tv', 'lamp', 'rice_cooker'],
    'Malam Hari Prancis': ['refrigerator', 'electric_heater', 'tv', 'computer'],
    'Custom': []
}

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
        trained_models_path = project_root / '2 - Model' / 'trained_models'
        
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

def calculate_realistic_peak(instant_watt, selected_appliances, appliance_dict):
    """Calculate realistic peak power with simultaneity factor"""
    
    # Kategori beban
    high_power = []  # >500W
    medium_power = []  # 200-500W
    low_power = []  # <200W
    
    for app in selected_appliances:
        if app in appliance_dict:
            data = appliance_dict[app]
            
            # Handle multi-mode (get max power)
            if isinstance(data[1], list):
                power = max([mode[1] for mode in data[1]])
            else:
                power = data[1]
            
            if power >= 500:
                high_power.append(power)
            elif power >= 200:
                medium_power.append(power)
            else:
                low_power.append(power)
    
    # Simultaneity factors (tidak semua nyala 100% bersamaan)
    # Berdasarkan IEC 60364 electrical installation standards
    
    # High power: hanya 1 yang nyala full (contoh: AC atau Rice Cooker atau Setrika)
    high_power_peak = max(high_power) if high_power else 0
    if len(high_power) > 1:
        # Tambah 30% dari high power kedua (cycling)
        high_power_peak += sum(high_power[1:]) * 0.3
    
    # Medium power: 70% simultaneity (contoh: Pompa + Dispenser)
    medium_power_peak = sum(medium_power) * 0.7
    
    # Low power: 90% simultaneity (Lampu, kipas kecil - hampir selalu nyala)
    low_power_peak = sum(low_power) * 0.9
    
    realistic_peak = high_power_peak + medium_power_peak + low_power_peak
    
    return realistic_peak

def get_pln_tariff(va):
    """Get PLN tariff per kWh based on VA capacity"""
    tariffs = {450: 415, 900: 1352, 1300: 1444, 2200: 1444, 3500: 1699, 5500: 1699}
    return tariffs.get(va, 1444)

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
                
                # If user provides custom duration, use it for primary mode
                if user_hours is not None:
                    # First mode gets user duration, rest proportional
                    total_kwh = (max_power / 1000) * user_hours
                else:
                    total_kwh = sum([(mode[1] / 1000) * mode[2] for mode in modes])
                
                instant_power += max_power
                daily_kwh += total_kwh
                breakdown.append((name, max_power, user_hours or sum([m[2] for m in modes]), total_kwh, False))
            else:
                # Single-mode appliance
                power = data[1]
                hours = user_hours if user_hours is not None else data[2]
                instant_power += power
                daily_energy = (power / 1000) * hours
                daily_kwh += daily_energy
                breakdown.append((name, power, hours, daily_energy, False))
    
    return instant_power, daily_kwh, breakdown

def prepare_model_input_for_hour(selected_appliances, appliance_dict, va, house_type, is_indonesia, target_hour, target_dayofweek=None, target_month=None, custom_durations=None):
    """
    Prepare input features for ML model prediction at a SPECIFIC hour.
    
    UPGRADED: Now includes time-based activity logic based on custom_durations.
    This makes the 24-hour curve dynamic and realistic.
    
    Args:
        target_hour: The hour (0-23) to prepare input for
        target_dayofweek: Optional override for day of week (0=Mon, 6=Sun)
        target_month: Optional override for month (1-12)
        custom_durations: Dict of {app_key: hours} from user sliders
    """
    current_time = datetime.now()
    hour = target_hour
    dayofweek = target_dayofweek if target_dayofweek is not None else current_time.weekday()
    month = target_month if target_month is not None else current_time.month
    
    # Define time periods for activity logic
    is_night = target_hour >= 18 or target_hour <= 6
    is_morning = 5 <= target_hour <= 9
    is_evening = 17 <= target_hour <= 22
    is_afternoon = 12 <= target_hour <= 17
    is_work_hours = 8 <= target_hour <= 17
    
    # Calculate sub-metering from appliances WITH time-based activity
    sub1_total = 0
    sub2_total = 0
    sub3_total = 0
    
    for app_key in selected_appliances:
        if app_key in appliance_dict:
            data = appliance_dict[app_key]
            if isinstance(data[1], list):
                power = max([mode[1] for mode in data[1]])
            else:
                power = data[1]
            
            power_kw = power / 1000
            
            # Get duration from slider (default 24 = always on)
            duration = 24
            if custom_durations and app_key in custom_durations:
                duration = custom_durations[app_key]
            
            # ================================================================
            # SMART ACTIVITY LOGIC - Makes the curve dynamic
            # ================================================================
            is_active = True
            
            if duration < 24:
                # Apply time-based heuristics based on appliance type
                if app_key in ['ac', 'electric_heater']:
                    # AC/Heater: prioritize evening/night (duration hours)
                    is_active = is_evening or is_night
                    
                elif app_key in ['lamp']:
                    # Lamp: active at night/evening only
                    is_active = is_night or is_evening
                    
                elif app_key in ['tv', 'computer']:
                    # TV/Computer: evening entertainment + some weekend daytime
                    is_active = is_evening or (dayofweek >= 5 and is_afternoon)
                    
                elif app_key in ['rice_cooker', 'kettle']:
                    # Rice cooker/Kettle: meal times (6-8, 11-13, 17-19)
                    is_active = (6 <= target_hour <= 8) or (11 <= target_hour <= 13) or (17 <= target_hour <= 19)
                    
                elif app_key in ['iron']:
                    # Iron: occasional, daytime on weekends
                    is_active = (dayofweek >= 5 and 9 <= target_hour <= 14)
                    
                elif app_key in ['water_pump']:
                    # Water pump: morning and evening peaks
                    is_active = is_morning or (17 <= target_hour <= 19)
                    
                elif app_key in ['fan']:
                    # Fan: daytime and evening when hot
                    is_active = is_afternoon or is_evening
                    
                elif app_key in ['dishwasher', 'laundry']:
                    # Dishwasher/Laundry: usually once a day
                    is_active = (9 <= target_hour <= 11) or (19 <= target_hour <= 21)
                    
                elif app_key in ['refrigerator', 'dispenser']:
                    # Always on appliances - ignore duration for these
                    is_active = True
                    
                else:
                    # Default: active during waking hours
                    is_active = 6 <= target_hour <= 23
            
            # Only add power if appliance is active at this hour
            if is_active:
                if app_key in ['rice_cooker', 'water_pump', 'kettle', 'dispenser']:
                    sub1_total += power_kw
                elif app_key in ['iron', 'laundry', 'dishwasher']:
                    sub2_total += power_kw
                elif app_key in ['ac', 'fan', 'electric_heater']:
                    sub3_total += power_kw
                elif app_key in ['refrigerator', 'tv', 'lamp', 'computer']:
                    sub1_total += power_kw * 0.3
                    sub3_total += power_kw * 0.7
    
    if is_indonesia:
        # Fitur sesuai model_id.joblib (19 fitur)
        # Catatan: Lag dan Rolling adalah APROKSIMASI dari daya sesaat.
        # Tanpa data historis nyata, nilai ini identik — ini limitasi
        # dari deployment tanpa backend time-series.
        total_active = sub1_total + sub2_total + sub3_total
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
        }
    else:
        # Fitur sesuai model_fr.joblib (16 fitur)
        # Catatan: Lag dan Rolling adalah APROKSIMASI dari daya sesaat.
        total_active = sub1_total + sub2_total + sub3_total
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
            'rate_of_change': 0,  # Tidak ada perubahan tanpa data historis
            'rolling_mean_3': total_active,
            'rolling_std_3': 0.1,
            'rolling_mean_24': total_active,
            'rolling_std_24': 0.15,
            'rolling_mean_168': total_active,
            'rolling_std_168': 0.2,
        }
    
    return input_dict

def predict_daily_curve(model, features, selected_appliances, appliance_dict, va, house_type, is_indonesia, is_weekend=False, custom_durations=None):
    """
    Generate 24-hour load predictions using the ML model.
    
    This is the KEY FEATURE that demonstrates the model's predictive power:
    - Shows how predicted consumption varies by hour
    - Proves ML captures temporal patterns (not just static calculation)
    - NOW uses custom_durations for time-based activity simulation
    
    Returns:
        list of 24 predictions (one per hour), or None if model unavailable
    """
    if model is None or features is None:
        return None
    
    predictions = []
    dayofweek = 5 if is_weekend else 1  # Saturday vs Tuesday
    
    for hour in range(24):
        try:
            # Prepare input for this specific hour WITH duration-based activity
            input_dict = prepare_model_input_for_hour(
                selected_appliances, appliance_dict, va, house_type, is_indonesia,
                target_hour=hour,
                target_dayofweek=dayofweek,
                custom_durations=custom_durations  # Pass slider values!
            )
            
            # Create DataFrame
            input_df = pd.DataFrame([input_dict])
            
            # Ensure all features exist
            for feat in features:
                if feat not in input_df.columns:
                    input_df[feat] = 0
            
            # Select features in correct order
            input_df = input_df[features]
            
            # Predict
            pred = float(model.predict(input_df)[0])
            predictions.append(max(pred, 0))  # Ensure non-negative
            
        except Exception as e:
            # Fall back to a simple approximation on error
            predictions.append(0)
    
    return predictions

def prepare_model_input(selected_appliances, appliance_dict, va, house_type, is_indonesia):
    """Prepare input features for ML model prediction"""
    
    current_time = datetime.now()
    hour = current_time.hour
    dayofweek = current_time.weekday()
    month = current_time.month
    
    # Calculate sub-metering based on actual appliances
    sub1_total = 0  # Kitchen appliances
    sub2_total = 0  # Laundry
    sub3_total = 0  # AC/Climate
    
    for app_key in selected_appliances:
        if app_key in appliance_dict:
            data = appliance_dict[app_key]
            
            # Handle multi-mode
            if isinstance(data[1], list):
                power = max([mode[1] for mode in data[1]])  # Use max power
            else:
                power = data[1]
            
            power_kw = power / 1000
            
            # Distribute to sub-meters (based on PELITA generator logic)
            if app_key in ['rice_cooker', 'water_pump', 'kettle', 'dispenser']:
                sub1_total += power_kw
            elif app_key in ['iron']:
                sub2_total += power_kw
            elif app_key in ['ac', 'fan']:
                sub3_total += power_kw
            elif app_key in ['refrigerator', 'tv', 'lamp']:
                # Distributed across sub1 and sub3
                sub1_total += power_kw * 0.3
                sub3_total += power_kw * 0.7
    
    if is_indonesia:
        # ID Model Features (19 fitur, sesuai model_id.joblib)
        total_active = sub1_total + sub2_total + sub3_total
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
        }
    else:
        # FR Model Features (16 fitur, sesuai model_fr.joblib)
        total_active = sub1_total + sub2_total + sub3_total
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
            'rolling_mean_24': total_active,
            'rolling_std_24': 0.15,
            'rolling_mean_168': total_active,
            'rolling_std_168': 0.2,
        }
    
    return input_dict

def get_input_hash(va, house_type, scenario, appliances):
    """Generate hash of current input state"""
    return hash((va, house_type, scenario, tuple(sorted(appliances))))

def generate_recommendations(power_kw, va, house_type, hour, appliances, dataset, realistic_kw):
    """Generate smart recommendations based on load level"""
    recommendations = []
    utilization = calculate_utilization(realistic_kw, va)  # Use realistic peak
    tariff = get_pln_tariff(va)
    
    # Determine risk level
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
    
    # Recommendations based on risk level
    if risk == "OVERLOAD":
        recommendations.append({
            'priority': 'HIGH', 'icon': '🚨',
            'action': 'MATIKAN peralatan daya besar SEKARANG!',
            'reason': f'Beban {utilization:.0f}% - MCB akan trip! Matikan AC/heater atau setrika segera.',
            'saving': f'Rp {realistic_kw * 0.3 * tariff * 24:,.0f}/hari'
        })
        recommendations.append({
            'priority': 'HIGH', 'icon': '🔴',
            'action': f'Pertimbangkan upgrade daya ke {va*2}VA',
            'reason': f'Kapasitas {va}VA tidak cukup untuk kebutuhan Anda.',
            'saving': 'Investasi untuk kenyamanan & keamanan'
        })
    
    if risk in ["OVERLOAD", "CRITICAL", "HEAVY"]:
        # Load reduction recommendations
        if 'ac' in appliances or 'electric_heater' in appliances:
            recommendations.append({
                'priority': 'HIGH', 'icon': '❄️',
                'action': 'Gunakan AC/Heater secara bijak',
                'reason': 'Set AC 24-26°C, gunakan timer, matikan saat tidak di rumah.',
                'saving': f'Rp {0.9 * 0.2 * 8 * tariff:,.0f}/hari'
            })
        
        if 'iron' in appliances:
            recommendations.append({
                'priority': 'MEDIUM', 'icon': '👔',
                'action': 'Setrika saat AC/heater mati',
                'reason': 'Hindari setrika bersamaan dengan peralatan berat lainnya.',
                'saving': f'Rp {0.3 * 1 * tariff:,.0f}/hari'
            })
        
        if 'rice_cooker' in appliances:
            recommendations.append({
                'priority': 'MEDIUM', 'icon': '🍚',
                'action': 'Cabut rice cooker setelah nasi matang',
                'reason': 'Mode warm 50W selama 4 jam = 0.2 kWh. Bisa pakai termos sebagai gantinya.',
                'saving': f'Rp {0.05 * tariff * 30:,.0f}/bulan'
            })
    
    # Time-based recommendations
    if 17 <= hour <= 22:
        recommendations.append({
            'priority': 'MEDIUM', 'icon': '🕐',
            'action': 'Tunda peralatan besar ke jam 22:00+',
            'reason': 'Jam 17:00-22:00 adalah maghrib peak dengan konsumsi tinggi.',
            'saving': f'Rp {power_kw * 0.15 * tariff * 5:,.0f}/hari'
        })
    
    # House type specific
    if house_type == 'Rumah Pekerja':
        recommendations.append({
            'priority': 'LOW', 'icon': '⏰',
            'action': 'Gunakan timer untuk peralatan',
            'reason': 'Rumah kosong siang hari. Timer mencegah pemborosan.',
            'saving': f'Rp {1.0 * tariff * 30:,.0f}/bulan'
        })
    
    # General energy saving tips
    if risk in ["NORMAL", "MEDIUM"]:
        recommendations.append({
            'priority': 'LOW', 'icon': '🔌',
            'action': 'Cabut charger & peralatan standby',
            'reason': 'Phantom load bisa 5-10% dari tagihan. Cabut TV, charger saat tidak dipakai.',
            'saving': f'Rp {0.2 * 30 * tariff:,.0f}/bulan'
        })
        
        recommendations.append({
            'priority': 'LOW', 'icon': '💡',
            'action': 'Ganti ke lampu LED',
            'reason': 'Lampu LED 80% lebih hemat dari lampu biasa.',
            'saving': f'Rp {0.5 * tariff * 30:,.0f}/bulan per lampu'
        })
    
    return recommendations

def generate_pdf_report(
    va, house_type, selected_appliances, appliance_dict,
    instant_watt, daily_kwh, current_kw, predicted_instant_kw,
    utilization_realistic, utilization_worst, risk_level, risk_text, daily_cost, monthly_cost,
    tariff, breakdown, recommendations, is_indonesia, selected_model,
    prediction_method, ml_prediction, realistic_peak_watt
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
    story.append(Paragraph(f"Laporan Analisis Konsumsi Energi - {datetime.now().strftime('%d %B %Y, %H:%M')}", normal_style))
    story.append(Spacer(1, 0.5*cm))
    
    # Garis pemisah
    story.append(Paragraph("<hr/>", normal_style))
    story.append(Spacer(1, 0.3*cm))
    
    # Section 1: Informasi Rumah
    story.append(Paragraph("🏠 INFORMASI RUMAH", heading_style))
    
    house_data = [
        ['Daya Terpasang', f'{va} VA'],
        ['Tipe Hunian', house_type],
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
        ['Prediksi ML Model', f'{predicted_instant_kw:.3f} kW' if ml_prediction else 'N/A', 
         f'{abs(predicted_instant_kw - current_kw):.3f} kW selisih' if ml_prediction else '-'],
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
    
    story.append(Paragraph(f"Laporan dibuat oleh H.E.M.A.T v2.0 - PELITA System", footer_style))
    story.append(Paragraph(f"© 2024 Nofal Rafif - Universitas Pamulang", footer_style))
    story.append(Paragraph(f"Disclaimer: Estimasi biaya berdasarkan asumsi pemakaian rata-rata dengan Simultaneity Factor IEC 60364. Biaya aktual dapat bervariasi.", footer_style))
    
    # Build PDF
    doc.build(story)
    buffer.seek(0)
    return buffer

# ============================================================================
# MAIN APP
# ============================================================================

def main():
    # Initialize session state
    if 'show_analysis' not in st.session_state:
        st.session_state.show_analysis = False
    if 'last_input_hash' not in st.session_state:
        st.session_state.last_input_hash = None
    
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
        
        # Dataset Selection
        st.markdown("### 📍 Pilih Dataset")
        dataset = st.radio(
            "Lokasi",
            options=["🇮🇩 Indonesia (ID)", "🇫🇷 Prancis (FR)"],
            help="Pilih dataset sesuai lokasi untuk peralatan yang relevan"
        )
        
        is_indonesia = "Indonesia" in dataset
        appliance_dict = APPLIANCES_INDONESIA if is_indonesia else APPLIANCES_FRANCE
        
        # Model Selection
        st.markdown("### 🤖 Pilih Model ML")
        if is_indonesia:
            model_options = ["ID"]
            default_model = "ID"
        else:
            model_options = ["FR"]
            default_model = "FR"
        
        selected_model = st.selectbox(
            "Model",
            options=model_options,
            index=0,
            help=f"Model {default_model} dioptimalkan untuk dataset ini"
        )
        
        st.markdown("---")
        
        # House Configuration
        st.markdown("### 🏠 Data Rumah")
        
        va = st.selectbox(
            "💡 Daya Listrik",
            options=[450, 900, 1300, 2200, 3500, 5500],
            index=1,  # Default 900 VA
            format_func=lambda x: f"{x} VA"
        )
        
        house_type = st.radio(
            "🏡 Tipe Hunian",
            options=["Rumah Pekerja", "Rumah Pensiunan", "Rumah Campuran"]
        )
        
        st.markdown("---")
        
        # Preset Scenarios
        st.markdown("### ⚡ Skenario Konsumsi")
        
        # Filter scenarios based on dataset
        if is_indonesia:
            scenario_options = [k for k in PRESET_SCENARIOS.keys() if 'Prancis' not in k]
        else:
            scenario_options = ['Malam Hari Prancis', 'Custom']
        
        selected_scenario = st.selectbox(
            "Pilih Skenario",
            options=scenario_options,
            index=len(scenario_options) - 1  # Default ke 'Custom' (terakhir)
        )
        
        # Get preset appliances
        preset_appliances = PRESET_SCENARIOS.get(selected_scenario, [])
        
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
                    default_hours = sum([m[2] for m in data[1]])
                else:
                    power = data[1]
                    default_hours = data[2]
                
                is_checked = key in preset_appliances if selected_scenario != 'Custom' else False
                if st.checkbox(f"{name} ({power}W)", value=is_checked, key=key):
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
            <p>⚡ H.E.M.A.T v1.0</p>
            <p>Nofal Rafif Setiawan Reserved 2026</p>
        </div>
        """, unsafe_allow_html=True)
    
    # ========================================================================
    # TAB 1: DASHBOARD
    # ========================================================================
    with tab1:
        # Check if input changed - if yes, reset analysis
        current_input_hash = get_input_hash(va, house_type, selected_scenario, selected_appliances)
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
            🤖 <strong style="color: white;">Model Aktif:</strong> {selected_model} | 
            📍 <strong style="color: white;">Dataset:</strong> {'Indonesia (PELITA)' if is_indonesia else 'Prancis (UCI)'}
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Only show metrics if there are selected appliances
        if len(selected_appliances) == 0:
            st.info("👈 Silakan pilih peralatan di sidebar untuk mulai menganalisis konsumsi energi Anda")
            return
        
        # Metrics
        current_hour = datetime.now().hour
        
        # Calculate REALISTIC peak (with simultaneity)
        realistic_peak_watt = calculate_realistic_peak(instant_watt, selected_appliances, appliance_dict)
        realistic_peak_kw = realistic_peak_watt / 1000
        
        utilization_worst = calculate_utilization(current_kw, va)  # Jika SEMUA nyala
        utilization_realistic = calculate_utilization(realistic_peak_kw, va)  # Realistis
        
        risk_level, risk_class, risk_text = get_risk_level(utilization_realistic)
        
        # Calculate cost correctly based on ACTUAL daily usage
        tariff = get_pln_tariff(va)
        daily_cost = daily_kwh * tariff  # kWh harian × Rp/kWh
        monthly_cost = daily_cost * 30
        
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
            # Show loading spinner while analyzing
            with st.spinner('🔍 Menganalisis konsumsi energi dengan ML model...'):
                import time
                
                # Load model
                dataset_type = "indonesia" if is_indonesia else "france"
                model, features = load_model(dataset_type)
                
                # Prepare input for model
                model_input = prepare_model_input(
                    selected_appliances, 
                    appliance_dict, 
                    va, 
                    house_type, 
                    is_indonesia
                )
                
                # Try ML prediction
                ml_prediction = None
                prediction_method = "Perhitungan Manual"
                
                if model is not None and features is not None:
                    try:
                        # Create DataFrame with exact features
                        input_df = pd.DataFrame([model_input])
                        
                        # Ensure all features exist
                        for feat in features:
                            if feat not in input_df.columns:
                                input_df[feat] = 0
                        
                        # Select only model features in correct order
                        input_df = input_df[features]
                        
                        # Predict
                        ml_prediction = float(model.predict(input_df)[0])
                        prediction_method = f"Machine Learning Model ({selected_model})"
                        
                        # Add delay to show processing
                        time.sleep(0.8)
                        
                    except Exception as e:
                        st.warning(f"⚠️ Model prediction error: {str(e)[:100]}")
                        ml_prediction = None
                
                # Use ML prediction if available, else use calculated
                predicted_instant_kw = ml_prediction if ml_prediction is not None else current_kw
            
            # Analysis Logic Section
            st.markdown("### 🔍 Analisis Sistem")
            
            # Show prediction comparison
            if ml_prediction is not None:
                col_pred1, col_pred2 = st.columns(2)
                with col_pred1:
                    st.metric(
                        "🤖 Prediksi ML Model", 
                        f"{predicted_instant_kw:.3f} kW",
                        help="Prediksi konsumsi sesaat dari Machine Learning"
                    )
                with col_pred2:
                    st.metric(
                        "🧮 Perhitungan Manual", 
                        f"{current_kw:.3f} kW",
                        help="Perhitungan dari penjumlahan daya peralatan"
                    )
                
                # Show difference
                diff_percent = abs(predicted_instant_kw - current_kw) / max(current_kw, 0.001) * 100
                if diff_percent < 10:
                    st.success(f"✅ Model dan perhitungan manual sesuai (selisih {diff_percent:.1f}%)")
                else:
                    st.info(f"ℹ️ Selisih {diff_percent:.1f}% — Model memperhitungkan pola konsumsi historis")
                
                st.caption("⚠️ Catatan: Lag dan Rolling features diisi dengan aproksimasi dari daya sesaat, bukan data historis nyata. Prediksi lebih akurat jika tersedia data pemakaian historis.")
            
            # ==================================================================
            # 📈 DAILY LOAD CURVE - The KEY ML Feature
            # ==================================================================
            st.markdown("### 📈 Prediksi Konsumsi 24 Jam (ML)")
            st.caption("Grafik ini menunjukkan bagaimana model ML memprediksi pola konsumsi Anda sepanjang hari")
            
            # Day type selector
            day_type = st.radio(
                "Pilih tipe hari:",
                options=["Hari Kerja (Weekday)", "Akhir Pekan (Weekend)"],
                horizontal=True,
                key="day_type_selector"
            )
            is_weekend_sim = "Weekend" in day_type
            
            # Generate predictions for both ML and static
            if model is not None and features is not None:
                with st.spinner("🔮 Memprediksi pola 24 jam..."):
                    ml_curve = predict_daily_curve(
                        model, features, selected_appliances, appliance_dict,
                        va, house_type, is_indonesia, is_weekend=is_weekend_sim,
                        custom_durations=st.session_state.get('custom_durations', {})
                    )
                
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
                    st.warning("Model mengembalikan prediksi kosong. Menggunakan perhitungan manual.")
            else:
                st.info("📊 Model ML tidak tersedia. Menampilkan perhitungan statis.")
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            # Analysis box dengan native Streamlit (bukan HTML)
            st.markdown("### 🔍 Analisis Sistem")
            
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
                    selected_appliances, dataset, realistic_peak_kw
                )
                
                for rec in recommendations:
                    priority_class = f"rec-{rec['priority'].lower()}"
                    st.markdown(f"""
                    <div class="rec-card {priority_class}">
                        <strong style="color: #2c3e50 !important;">{rec['icon']} {rec['action']}</strong>
                        <p style="margin: 8px 0 5px 0; color: #555 !important; font-size: 0.95rem;">{rec['reason']}</p>
                        <p style="margin: 5px 0 0 0; color: #11998e !important; font-weight: bold; font-size: 1rem;">💰 {rec['saving']}</p>
                    </div>
                    """, unsafe_allow_html=True)
            
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
                Beban di atas 100% akan membuat MCB trip (listrik padam otomatis) untuk 
                melindungi instalasi dari kerusakan. Terus menerus overload bisa merusak 
                kabel dan peralatan listrik!
                """)
            elif risk_level == "KRITIS":
                st.warning(f"""
                🔴 **KRITIS - HAMPIR PENUH!**
                
                **Beban Anda: {utilization_realistic:.1f}% dari kapasitas {va}VA**
                
                ⚠️ **Rekomendasi:**
                - Hindari menyalakan peralatan besar tambahan
                - Lonjakan kecil bisa menyebabkan MCB trip
                - Pertimbangkan upgrade ke {va*1.5:.0f}VA atau {va*2}VA
                
                💡 Sisa margin hanya {100-utilization_realistic:.1f}% - sangat sedikit!
                """)
            elif risk_level == "BERAT":
                st.info(f"""
                ⚡ **BEBAN BERAT - PERLU PERHATIAN**
                
                **Beban Anda: {utilization_realistic:.1f}% dari kapasitas {va}VA**
                
                ✓ **Masih aman, tapi:**
                - Jangan nyalakan AC dan setrika bersamaan
                - Jangan tambah peralatan daya besar (>500W)
                - Gunakan peralatan berat secara bergantian
                
                💡 **Analogi:** Seperti lift dengan kapasitas {va}kg, sekarang isi {utilization_realistic:.0f}%. 
                Masih OK, tapi jangan tambah 2-3 orang lagi!
                
                📈 Untuk kenyamanan lebih, pertimbangkan upgrade ke {va*1.5:.0f}VA
                """)
            elif risk_level == "PERHATIAN":
                st.info(f"""
                🟡 **BEBAN SEDANG**
                
                **Beban Anda: {utilization_realistic:.1f}% dari kapasitas {va}VA**
                
                ✓ Kondisi normal untuk rumah tangga
                ✓ Hindari menyalakan terlalu banyak peralatan besar bersamaan
                ✓ Sisa margin: {100-utilization_realistic:.1f}%
                
                💡 Tips: Gunakan AC dan rice cooker/setrika secara bergantian
                """)
            else:
                st.success(f"""
                ✅ **AMAN - BEBAN NORMAL**
                
                **Beban Anda: {utilization_realistic:.1f}% dari kapasitas {va}VA**
                
                💚 Beban listrik dalam kondisi normal dan aman
                ✓ Masih ada margin {100-utilization_realistic:.1f}% untuk peralatan tambahan
                ✓ MCB tidak akan trip dalam kondisi normal
                
                👍 Kapasitas daya Anda sudah sesuai dengan kebutuhan!
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
                        realistic_peak_watt=realistic_peak_watt
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
                <p><strong>Fitur:</strong> 19 (sub-metering + temporal + lag + rolling)</p>
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
        - 🔮 Prediksi dengan 2 model ML (Random Forest Regressor)
        - 📍 Support 2 dataset (Indonesia & Prancis)
        - ⚡ Preset scenario untuk kemudahan
        - 🔍 Analisis logika yang transparan
        - 💡 Rekomendasi prioritas dengan estimasi hemat
        - 📊 Breakdown biaya per peralatan
        
        ### 📐 Formula Perhitungan
        - **Utilisasi:** (Daya Konsumsi / (VA × 0.85)) × 100%
        - **Biaya Harian:** Daya (kW) × 24 jam × Tarif (Rp/kWh)
        - **Power Factor:** 0.85 (standar PLN)
        
        ### 👨‍💻 Pengembang
        **Nama:** Nofal Rafif  
        **NIM:** 221011402613  
        **Universitas:** Universitas Pamulang  
        **Tahun:** 2024
        """)

if __name__ == "__main__":
    main()