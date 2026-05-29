import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

print("="*80)
print("DATASET GENERATOR - PELITA")
print("="*80)

# ============================================================================
# PARAMETER
# ============================================================================
num_houses = 100
hours_per_house = 168 * 12  # ~3 bulan
start_date_base = datetime(2025, 4, 1)  # April: mencakup season 1 (Mei-Jun) dan 2 (Apr)

print(f"Rumah: {num_houses}")
print(f"Durasi: {hours_per_house} jam")
print(f"Start: {start_date_base.date()}")
print(f"Total: {num_houses * hours_per_house:,} baris")

# ============================================================================
# CLASS FSM dengan VA-Aware
# ============================================================================
class ApplianceFSM:
    def __init__(self, appliance_type, house_va):
        self.appliance_type = appliance_type
        self.state = 'OFF'
        self.power = 0
        self.cycle_time = 0
        self.house_va = house_va
        self.duty_ratio = np.random.uniform(0.4, 0.7) if appliance_type == 'AC' else 1
        
    def update(self, hour, season, dayofweek, demand_trigger=False):
        if self.appliance_type == 'AC':
            if season == 1 or hour in [12,13,14,18,19,20,21]:
                self.state = 'COOL' if np.random.rand() < self.duty_ratio else 'IDLE'
                self.power = 800 if self.state == 'COOL' else 30
            else:
                self.state = 'OFF'
                self.power = 0
        elif self.appliance_type == 'Rice Cooker':
            if hour in [6,7,18,19] and np.random.rand() > 0.4:
                if self.cycle_time < 1:  # 1 jam masak (resolusi per jam)
                    self.state = 'COOK'
                    self.power = 350
                else:
                    self.state = 'WARM'
                    self.power = 50
                self.cycle_time += 1
            else:
                self.state = 'OFF'
                self.power = 0
                self.cycle_time = 0
        elif self.appliance_type == 'Pump':
            # Perhitungan dalam Watt, konversi ke kW hanya di return
            max_power_w = self.house_va * 0.8            # 720W untuk 900VA
            min_power_w = min(400, max_power_w * 0.7)    # maks 400W
            if (hour in [6,7,17,18,19]) and demand_trigger:
                if self.cycle_time < 1:  # 1 jam operasi (resolusi per jam)
                    self.state = 'ON'
                    self.power = np.random.uniform(min_power_w, max_power_w)
                    self.cycle_time += 1
                else:
                    self.state = 'OFF'
                    self.power = 0
                    self.cycle_time = 0
            else:
                self.state = 'OFF'
                self.power = 0
        return self.power / 1000  # Watt -> kW

# ============================================================================
# FUNGSI GENERATE DATA
# ============================================================================
def generate_house_data(house_id, start_time):
    data = []
    
    va = np.random.choice([900, 1300, 2200], p=[0.3, 0.5, 0.2])
    max_power = va / 1000  # kW strict
    pf = np.random.uniform(0.85, 0.95)
    
    ac_fsm = ApplianceFSM('AC', va)
    rice_fsm = ApplianceFSM('Rice Cooker', va)
    pump_fsm = ApplianceFSM('Pump', va)
    
    occupancy = np.random.randint(2, 6)
    house_type = np.random.choice(['working_class', 'retired', 'mixed'], p=[0.6, 0.2, 0.2])
    
    # FIX #1: MCB STATE MACHINE
    mcb_is_tripped = False
    mcb_trip_time = None
    mcb_reset_delay = 3  # Reset setelah 3 jam (simulasi reset manual realistis)
    
    for h in range(hours_per_house):
        current_time = start_time + timedelta(hours=h)
        hour = current_time.hour
        dayofweek = current_time.weekday()
        month = current_time.month
        season = 1 if month in [5,6,7,8,9,10] else 2
        
        demand_trigger = np.random.poisson(0.125) > 0
        
        # ====================================================================
        # MCB STATE CHECK - CRITICAL
        # ====================================================================
        if mcb_is_tripped:
            # Saat MCB trip, SEMUA power = 0 (circuit open)
            global_active_power = 0
            sub_metering_1 = 0
            sub_metering_2 = 0
            sub_metering_3 = 0
            sub_metering_4 = 0
            
            # Check if reset time reached (manual reset after 1 hour)
            if (h - mcb_trip_time) >= mcb_reset_delay:
                mcb_is_tripped = False
                mcb_trip_time = None
            
            # Skip ke calculation langsung
            voltage = 220  # Nominal (no load)
            global_reactive_power = 0
            global_intensity = 0
            emisi_co2 = 0
            
            data.append({
                'Datetime': current_time,
                'Global_active_power': 0,
                'Global_reactive_power': 0,
                'Voltage': voltage,
                'Global_intensity': 0,
                'Sub_metering_1': 0,
                'Sub_metering_2': 0,
                'Sub_metering_3': 0,
                'Sub_metering_4': 0,
                'Hour': hour,
                'Dayofweek': dayofweek,
                'Month': month,
                'Season': season,
                'Emisi_CO2': 0,
                'MCB_Tripped': 1,  # FIX: 1 = Tripped (standard convention)
                'Potential_Power': 0,  # Unknown during continued trip
            })
            continue
        
        # ====================================================================
        # NORMAL OPERATION (MCB not tripped)
        # ====================================================================
        ac_power = ac_fsm.update(hour, season, dayofweek) * occupancy * 0.18  # Reduced from 0.2
        rice_power = rice_fsm.update(hour, season, dayofweek)
        pump_power = pump_fsm.update(hour, season, dayofweek, demand_trigger)
        
        # Sub-metering
        sub_metering_1 = rice_power + pump_power + np.random.normal(0.08, 0.03)
        
        if dayofweek >= 5:
            sub_metering_2 = np.random.normal(0.5, 0.15) if np.random.rand() < 0.4 else 0
        else:
            sub_metering_2 = np.random.normal(0.4, 0.1) if np.random.rand() < 0.15 else 0
        sub_metering_2 += np.random.normal(0, 0.03)
        
        sub_metering_3 = ac_power + np.random.normal(0, 0.1)
        if season == 1:
            sub_metering_3 += 0.2
        
        # Maghrib + Weekend
        if 17 <= hour <= 19:
            maghrib_boost = np.random.uniform(0.2, 0.35)
            sub_metering_1 *= (1 + maghrib_boost * 0.5)
        
        if dayofweek >= 5:
            weekend_mult = np.random.uniform(1.1, 1.25)
            sub_metering_3 *= weekend_mult
        
        sub_metering_1 = max(0, sub_metering_1)
        sub_metering_2 = max(0, sub_metering_2)
        sub_metering_3 = max(0, sub_metering_3)
        
        # FIX #3: BASELOAD SCALED BY VA (prevent overload)
        tracked_base = sub_metering_1 + sub_metering_2 + sub_metering_3
        
        # Electronics + baseload SCALED by house capacity
        # More conservative scaling to prevent trips
        va_ratio = va / 1300  # Normalize to 1300VA baseline
        base_electronics = 0.4 * va_ratio  # Reduced from 0.5
        
        if hour >= 18 or hour <= 6:
            base_electronics += 0.25 * va_ratio  # Night boost (reduced from 0.3)
        if house_type == 'retired':
            base_electronics *= 1.15  # More conservative (was 1.2)
        if dayofweek >= 5:
            base_electronics *= np.random.uniform(1.03, 1.12)  # Weekend scaled (reduced)
        
        base_electronics = np.random.normal(base_electronics, 0.12) + np.random.poisson(0.08) * 0.05
        base_electronics = max(0, base_electronics)
        
        # ALWAYS-ON baseload (phantom load)
        phantom_load = np.random.uniform(0.010, 0.025)  # Reduced max from 0.030
        
        total_tracked = tracked_base + base_electronics + phantom_load
        
        # SMART LOAD MANAGEMENT: Prevent overload BEFORE MCB check
        provisional_power = total_tracked * 1.15  # Estimate with untracked
        safety_margin = (va * pf / 1000) * 0.85  # Memperhitungkan power factor per rumah
        
        if provisional_power > safety_margin:
            # Priority shedding (simulasi perilaku penghuni):
            # 1. Matikan AC, 2. Matikan laundry, 3. Kurangi elektronik
            
            if sub_metering_3 > 0:
                sub_metering_3 = 0  # Matikan AC sepenuhnya
                tracked_base = sub_metering_1 + sub_metering_2 + sub_metering_3
                total_tracked = tracked_base + base_electronics + phantom_load
                provisional_power = total_tracked * 1.15
            
            if provisional_power > safety_margin and sub_metering_2 > 0:
                sub_metering_2 = 0  # Matikan laundry
                tracked_base = sub_metering_1 + sub_metering_2 + sub_metering_3
                total_tracked = tracked_base + base_electronics + phantom_load
                provisional_power = total_tracked * 1.15
            
            if provisional_power > safety_margin:
                base_electronics = base_electronics * 0.4  # Kurangi elektronik drastis
                total_tracked = tracked_base + base_electronics + phantom_load
        
        # Untracked power (losses, standby, dll)
        untracked_power = total_tracked * np.random.uniform(0.10, 0.20)
        global_active_power = total_tracked + untracked_power
        
        # Probabilistic safety cap: simulasi bahwa penghuni KADANG berhasil
        # mengurangi beban tepat waktu, tapi kadang tidak (sehingga MCB trip)
        max_safe_active = (va * pf / 1000) * 0.98
        if global_active_power > max_safe_active:
            if np.random.rand() < 0.85:  # 85% berhasil mengurangi beban
                global_active_power = max_safe_active * np.random.uniform(0.88, 0.97)
            # else: 15% gagal → MCB trip terjadi di bawah
        
        # FIX #6: Store POTENTIAL power BEFORE any MCB cutoff
        # This is the load that WOULD have been consumed if MCB didn't trip
        potential_power = global_active_power
        
        # MCB Check BEFORE recording
        apparent_power_va = (global_active_power * 1000) / pf  # Convert to VA
        
        if apparent_power_va > va:  # Overload!
            # MCB TRIPS IMMEDIATELY - CUT POWER NOW
            mcb_is_tripped = True
            mcb_trip_time = h
            mcb_status = 1  # Tripped
            
            # CRITICAL FIX: Cut power IMMEDIATELY at this iteration
            global_active_power = 0
            sub_metering_1 = 0
            sub_metering_2 = 0
            sub_metering_3 = 0
            sub_metering_4 = 0
        else:
            mcb_status = 0  # Normal
            # FIX #5: Sub4 = Residual STRICT (only if not tripped)
            sub_metering_4 = global_active_power - (sub_metering_1 + sub_metering_2 + sub_metering_3)
            sub_metering_4 = max(0, sub_metering_4)
        
        # Missing values (only for non-tripped data)
        if mcb_status == 0 and np.random.rand() < 0.015:
            global_active_power = np.nan
            skip_calc = True
        else:
            skip_calc = False
        
        # ====================================================================
        # FIX #4: VOLTAGE DYNAMICS (Learn from aseli_no_fake)
        # ====================================================================
        if not skip_calc and mcb_status == 0:  # Only calculate if not tripped
            # Base voltage dengan realistic variance
            voltage_base = np.random.normal(220, 2.5)  # σ=2.5V
            
            # Voltage drop berdasarkan beban (Ohm's law: V = V0 - I×Z)
            # Higher load → more drop
            current_estimate = (global_active_power * 1000) / (voltage_base * pf)
            line_impedance = np.random.uniform(0.3, 0.8)  # Ohm (typical residential)
            voltage_drop = current_estimate * line_impedance
            
            voltage = voltage_base - voltage_drop
            
            # Clamp PLN range (+5%/-10%)
            voltage = max(198, min(231, voltage))
            
            # Reactive power
            theta = np.arccos(pf)
            global_reactive_power = global_active_power * np.tan(theta)
            
            # Current (dengan voltage yang sudah adjusted)
            global_intensity = (global_active_power * 1000) / (voltage * pf) if voltage > 0 else 0
            
            # CO2 dynamic (higher at night - coal baseload)
            if 0 <= hour <= 6 or 22 <= hour <= 23:
                co2_factor = 0.90  # Night: coal-heavy
            elif 10 <= hour <= 14:
                co2_factor = 0.80  # Day: some solar (minimal in Indonesia)
            else:
                co2_factor = 0.85  # Average
            emisi_co2 = global_active_power * co2_factor
        elif mcb_status == 1:  # Tripped state
            voltage = 220  # Nominal (no load)
            global_reactive_power = 0
            global_intensity = 0
            emisi_co2 = 0
        else:
            voltage = np.nan
            global_reactive_power = np.nan
            global_intensity = np.nan
            emisi_co2 = np.nan
        
        data.append({
            'Datetime': current_time,
            'Global_active_power': global_active_power,
            'Global_reactive_power': max(0, global_reactive_power) if not np.isnan(global_reactive_power) else np.nan,
            'Voltage': voltage,
            'Global_intensity': max(0, global_intensity) if not np.isnan(global_intensity) else np.nan,
            'Sub_metering_1': sub_metering_1,
            'Sub_metering_2': sub_metering_2,
            'Sub_metering_3': sub_metering_3,
            'Sub_metering_4': sub_metering_4,
            'Hour': hour,
            'Dayofweek': dayofweek,
            'Month': month,
            'Season': season,
            'Emisi_CO2': max(0, emisi_co2) if not np.isnan(emisi_co2) else np.nan,
            'MCB_Tripped': mcb_status,
            'Potential_Power': potential_power,  # FIX #6: Pre-trip load for recommendations
        })
    
    df_house = pd.DataFrame(data)
    df_house.set_index('Datetime', inplace=True)
    
    # Lag features
    df_house['Lag_1'] = df_house['Global_active_power'].shift(1)
    df_house['Lag_24'] = df_house['Global_active_power'].shift(24)
    df_house['Lag_168'] = df_house['Global_active_power'].shift(168)

    # Rolling statistics: dihitung dari nilai yang sudah di-shift
    # agar nilai waktu-t (target) tidak masuk ke window
    shifted = df_house['Global_active_power'].shift(1)
    df_house['Rolling_mean_3'] = shifted.rolling(window=3).mean()
    df_house['Rolling_mean_24'] = shifted.rolling(window=24).mean()
    df_house['Rolling_std_24'] = shifted.rolling(window=24).std()

    # Drop baris dengan NaN di lag (bukan fillna(0) yang memasukkan sinyal palsu)
    # 168 baris pertama per rumah hilang — lebih bersih daripada belajar dari lag=0
    df_house = df_house.dropna(subset=['Lag_1', 'Lag_24', 'Lag_168'])
    
    df_house.reset_index(inplace=True)
    df_house['House_ID'] = house_id
    df_house['VA'] = va
    df_house['House_Type'] = house_type
    
    return df_house

# ============================================================================
# MAIN
# ============================================================================
print(f"\nGenerating data...")
all_houses = []
for house_id in range(1, num_houses + 1):
    df_house = generate_house_data(house_id, start_date_base)
    all_houses.append(df_house)
    if house_id % 10 == 0:
        print(f"  {house_id}/{num_houses} houses done")

df_dataset = pd.concat(all_houses, ignore_index=True)
output_filename = 'dataset_energi_rumah_indonesia_PELITA_Fix.csv'
df_dataset.to_csv(output_filename, index=False)

print(f"\nDataset saved: {output_filename}")
print(f"Total rows: {len(df_dataset):,}")

# ============================================================================
# VALIDATION
# ============================================================================
print("\n" + "="*60)
print("VALIDATION")
print("="*60)

# Test 1: Energy Conservation
print("\n1. Energy Conservation:")
sample = df_dataset[df_dataset['MCB_Tripped']==0].dropna().sample(min(20, len(df_dataset)))
energy_valid = 0
for idx, row in sample.iterrows():
    gp = row['Global_active_power']
    ss = row['Sub_metering_1'] + row['Sub_metering_2'] + row['Sub_metering_3'] + row['Sub_metering_4']
    if abs(gp - ss) < 0.001:
        energy_valid += 1
print(f"   Valid: {energy_valid}/20 — {'PASS' if energy_valid >= 19 else 'FAIL'}")

# Test 2: MCB Behavior
print("\n2. MCB Logic (1=Trip, 0=Normal):")
trip_samples = df_dataset[df_dataset['MCB_Tripped']==1].head(5)
if len(trip_samples) > 0 and (trip_samples['Global_active_power'] == 0).all():
    print("   PASS — MCB trip cuts power to 0")
else:
    print("   FAIL — power flows during trip")

# Test 3: Baseload
print("\n3. Baseload Realism:")
normal_data = df_dataset[df_dataset['MCB_Tripped']==0]
exact_zero = (normal_data['Global_active_power'] == 0.0).sum()
print(f"   Exact 0.0 during normal: {exact_zero} ({exact_zero/len(normal_data)*100:.2f}%)")
print(f"   {'PASS' if exact_zero < len(normal_data)*0.001 else 'Too many zeros'}")

# Test 4: Rolling Leakage
print("\n4. Rolling Leakage Check:")
first_50 = df_dataset.head(50).dropna()
leakage = 0
for idx, row in first_50.iterrows():
    if abs(row['Rolling_mean_3'] - row['Global_active_power']) < 0.0001:
        leakage += 1
print(f"   Exact match Rolling=Global: {leakage}/50 — {'NO LEAKAGE' if leakage < 3 else 'LEAKAGE'}")

# Test 5: Voltage Correlation
print("\n5. Voltage-Load Correlation (should be negative):")
corr_data = df_dataset[df_dataset['MCB_Tripped']==0].dropna()
if len(corr_data) > 100:
    corr = corr_data[['Global_active_power', 'Voltage']].corr().iloc[0,1]
    print(f"   Correlation: {corr:.3f} — {'Realistic' if corr < -0.1 else 'Check needed'}")

# Test 6: MCB Trip Rate
print("\n6. MCB Trip Frequency (by VA class):")
for va_class in [900, 1300, 2200]:
    subset = df_dataset[df_dataset['VA'] == va_class]
    trip_rate = subset['MCB_Tripped'].mean() * 100
    print(f"   {va_class}VA: {100-trip_rate:.1f}% operational (trip: {trip_rate:.1f}%)")

# Test 7: Season Coverage
print("\n7. Season Coverage:")
season_counts = df_dataset.groupby('Season').size()
for s, count in season_counts.items():
    print(f"   Season {s}: {count:,} rows ({count/len(df_dataset)*100:.1f}%)")

print("\n" + "="*60)
print("Generation complete.")
print("="*60)