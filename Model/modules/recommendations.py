"""
Recommendation Engine for Energy Prediction Models
===================================================

Sistem rekomendasi hemat listrik yang user-friendly untuk rumah tangga Indonesia.

Features:
- Model selection strategy (auto-recommend best model)
- MCB risk prediction (simplified as "warning listrik padam")
- Load shedding suggestions with priority
- Rupiah savings estimation with PLN tariff
- Peak/off-peak optimization

Author: Nofal Rafif
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum
import numpy as np
import pandas as pd


class RiskLevel(Enum):
    """Risk level for power consumption"""
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class Priority(Enum):
    """Priority level for recommendations"""
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"


@dataclass
class Recommendation:
    """Single recommendation item"""
    action: str
    reason: str
    priority: Priority
    potential_saving_kwh: float
    potential_saving_idr: float
    category: str  # 'time', 'appliance', 'behavior', 'mcb'


@dataclass
class PredictionResult:
    """Result from energy prediction"""
    current_power_kw: float
    predicted_power_kw: float
    va_capacity: int
    utilization_percent: float
    risk_level: RiskLevel
    estimated_daily_kwh: float
    estimated_daily_idr: float


class PLNTariff:
    """PLN electricity tariff calculator for Indonesia"""
    
    # Tarif PLN 2024 (Rp/kWh) - simplified
    TARIFF_BY_VA = {
        450: 415,    # Subsidi
        900: 1352,   # Subsidi
        1300: 1444,  # Non-subsidi
        2200: 1444,  # Non-subsidi
        3500: 1699,  # Non-subsidi
        5500: 1699,  # Non-subsidi
    }
    
    # Peak hours (17:00-22:00) multiplier
    PEAK_MULTIPLIER = 1.0  # Currently no time-based tariff for residential
    
    @classmethod
    def get_tariff(cls, va: int) -> float:
        """Get tariff per kWh based on VA capacity"""
        if va in cls.TARIFF_BY_VA:
            return cls.TARIFF_BY_VA[va]
        # Default to highest non-subsidi
        return 1699
    
    @classmethod
    def calculate_cost(cls, kwh: float, va: int) -> float:
        """Calculate cost in Rupiah"""
        tariff = cls.get_tariff(va)
        return kwh * tariff
    
    @classmethod
    def estimate_monthly_bill(cls, daily_kwh: float, va: int) -> float:
        """Estimate monthly electricity bill"""
        return cls.calculate_cost(daily_kwh * 30, va)


class ModelSelector:
    """Select best model based on evaluation results"""
    
    def recommend_model(
        self, 
        comparison_results: Dict[str, Dict[str, float]]
    ) -> Dict[str, Any]:
        """
        Recommend best model based on comparison results.
        
        Logic:
        - If difference <5% → Epsilon (auto, simple)
        - If specialized model >10% better → recommend that model
        
        Args:
            comparison_results: Dict with model names and their R2 scores
            
        Returns:
            Recommendation dict with model name and reason
        """
        scores = {name: res.get('R2', 0) for name, res in comparison_results.items()}
        
        if not scores:
            return {
                'recommended': 'Epsilon',
                'reason': 'Default: Epsilon otomatis mendeteksi tipe data',
                'allow_manual': False
            }
        
        best_model = max(scores, key=scores.get)
        best_score = scores[best_model]
        min_score = min(scores.values())
        
        if best_score == 0:
            diff = 0
        else:
            diff = (best_score - min_score) / best_score
        
        if diff < 0.05:  # <5% difference
            return {
                'recommended': 'Epsilon',
                'reason': 'Performa ketiga model hampir sama. Epsilon dipilih karena otomatis mendeteksi tipe data.',
                'allow_manual': False,
                'scores': scores
            }
        else:
            return {
                'recommended': best_model,
                'reason': f'{best_model} lebih akurat {diff*100:.1f}% dari model lain',
                'allow_manual': True,
                'scores': scores,
                'options': {
                    'Delta': 'Untuk data standar internasional (UCI)',
                    'Nadir': 'Untuk rumah Indonesia dengan data VA/MCB',
                    'Epsilon': 'Otomatis menyesuaikan (recommended)'
                }
            }


class RecommendationEngine(ABC):
    """Abstract base class for recommendation engines"""
    
    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.tariff = PLNTariff()
    
    @abstractmethod
    def generate_recommendations(
        self,
        prediction: PredictionResult,
        user_data: Dict[str, Any]
    ) -> List[Recommendation]:
        """Generate recommendations based on prediction and user data"""
        pass
    
    @abstractmethod
    def predict_risk(
        self,
        current_power: float,
        va_capacity: int
    ) -> Tuple[RiskLevel, str]:
        """Predict risk level and generate warning message"""
        pass
    
    def _log(self, message: str):
        if self.verbose:
            print(f"[{self.__class__.__name__}] {message}")


class PELITARecommendationEngine(RecommendationEngine):
    """
    Recommendation engine optimized for PELITA (Indonesia) dataset.
    
    Features:
    - MCB risk prediction (simplified as "warning listrik padam")
    - VA-aware recommendations
    - Rupiah savings estimation
    - House type specific suggestions
    """
    
    # Appliance power consumption estimates (Watt)
    APPLIANCE_POWER = {
        'ac': 900,
        'kulkas': 100,
        'tv': 100,
        'rice_cooker': 400,
        'mesin_cuci': 500,
        'water_heater': 1500,
        'setrika': 300,
        'pompa_air': 250,
        'lampu': 50,
        'charger': 10,
    }
    
    def predict_risk(
        self,
        current_power: float,
        va_capacity: int
    ) -> Tuple[RiskLevel, str]:
        """
        Predict MCB trip risk (simplified for user).
        
        Args:
            current_power: Current power in kW
            va_capacity: VA capacity (900, 1300, 2200)
            
        Returns:
            Tuple of (RiskLevel, warning_message)
        """
        # Convert VA to kW (assuming power factor 0.85)
        max_power_kw = (va_capacity * 0.85) / 1000
        utilization = (current_power / max_power_kw) * 100
        
        if utilization >= 95:
            return (
                RiskLevel.CRITICAL,
                "⚠️ BAHAYA: Listrik bisa padam kapan saja! Segera matikan beberapa peralatan!"
            )
        elif utilization >= 85:
            return (
                RiskLevel.HIGH,
                "⚠️ PERINGATAN: Konsumsi mendekati batas daya! Kurangi beban segera."
            )
        elif utilization >= 70:
            return (
                RiskLevel.MEDIUM,
                "⚡ Perhatian: Konsumsi cukup tinggi. Hindari menyalakan peralatan besar."
            )
        else:
            return (
                RiskLevel.LOW,
                "✅ Konsumsi normal. Listrik aman."
            )
    
    def suggest_load_shedding(
        self,
        current_power: float,
        va_capacity: int,
        active_appliances: List[str]
    ) -> List[Dict[str, Any]]:
        """
        Suggest which appliances to turn off to reduce load.
        
        Priority: Turn off high-power, non-essential appliances first.
        """
        max_power_kw = (va_capacity * 0.85) / 1000
        target_power = max_power_kw * 0.7  # Target 70% utilization
        power_to_reduce = current_power - target_power
        
        if power_to_reduce <= 0:
            return []
        
        # Priority order (turn off first)
        priority_order = [
            'water_heater',  # Highest power, can wait
            'setrika',       # Can postpone
            'ac',            # High power
            'mesin_cuci',    # Can postpone
            'rice_cooker',   # Can postpone if not cooking
            'pompa_air',     # Usually automatic
            'tv',            # Entertainment
        ]
        
        suggestions = []
        reduced = 0
        
        for appliance in priority_order:
            if appliance in active_appliances and reduced < power_to_reduce * 1000:
                power = self.APPLIANCE_POWER.get(appliance, 100)
                reduced += power
                
                suggestions.append({
                    'appliance': appliance,
                    'power_watt': power,
                    'priority': 'HIGH' if power > 500 else 'MEDIUM',
                    'message': self._get_appliance_message(appliance)
                })
        
        return suggestions
    
    def _get_appliance_message(self, appliance: str) -> str:
        """Get user-friendly message for appliance"""
        messages = {
            'ac': "Matikan AC atau naikkan suhu ke 26°C",
            'water_heater': "Matikan water heater sementara",
            'setrika': "Tunda menyetrika ke jam non-peak",
            'mesin_cuci': "Tunda mencuci ke malam hari (setelah jam 22:00)",
            'rice_cooker': "Gunakan mode warm, bukan cook",
            'pompa_air': "Matikan pompa air sementara",
            'tv': "Matikan TV yang tidak ditonton",
        }
        return messages.get(appliance, f"Pertimbangkan mematikan {appliance}")

    
    def estimate_savings(
        self,
        current_daily_kwh: float,
        target_reduction_percent: float,
        va_capacity: int
    ) -> Dict[str, float]:
        """
        Estimate potential savings in kWh and Rupiah.
        
        Args:
            current_daily_kwh: Current daily consumption
            target_reduction_percent: Target reduction (e.g., 0.15 for 15%)
            va_capacity: VA for tariff calculation
            
        Returns:
            Dict with daily, monthly, yearly savings
        """
        daily_saving_kwh = current_daily_kwh * target_reduction_percent
        tariff = PLNTariff.get_tariff(va_capacity)
        
        return {
            'daily_kwh': daily_saving_kwh,
            'daily_idr': daily_saving_kwh * tariff,
            'monthly_kwh': daily_saving_kwh * 30,
            'monthly_idr': daily_saving_kwh * 30 * tariff,
            'yearly_kwh': daily_saving_kwh * 365,
            'yearly_idr': daily_saving_kwh * 365 * tariff,
        }
    
    def generate_recommendations(
        self,
        prediction: PredictionResult,
        user_data: Dict[str, Any]
    ) -> List[Recommendation]:
        """
        Generate personalized recommendations for PELITA users.
        
        Args:
            prediction: Prediction result with power and risk info
            user_data: User input data (VA, house_type, appliances, etc.)
            
        Returns:
            List of prioritized recommendations
        """
        recommendations = []
        va = user_data.get('va', 1300)
        house_type = user_data.get('house_type', 'mixed')
        hour = user_data.get('hour', 12)
        appliances = user_data.get('appliances', [])
        
        tariff = PLNTariff.get_tariff(va)
        
        # 1. MCB Risk Warning (highest priority)
        if prediction.risk_level in [RiskLevel.HIGH, RiskLevel.CRITICAL]:
            saving = prediction.current_power_kw * 0.2 * tariff  # 20% reduction
            recommendations.append(Recommendation(
                action="Kurangi beban listrik segera!",
                reason=f"Konsumsi {prediction.utilization_percent:.0f}% dari kapasitas. Listrik bisa padam!",
                priority=Priority.HIGH,
                potential_saving_kwh=prediction.current_power_kw * 0.2,
                potential_saving_idr=saving,
                category='mcb'
            ))
        
        # 2. Time-based recommendations
        if 17 <= hour <= 22:  # Peak hours
            saving_kwh = prediction.current_power_kw * 0.15
            recommendations.append(Recommendation(
                action="Tunda penggunaan peralatan besar ke jam 22:00+",
                reason="Jam 17:00-22:00 adalah jam sibuk. Konsumsi biasanya lebih tinggi.",
                priority=Priority.MEDIUM,
                potential_saving_kwh=saving_kwh,
                potential_saving_idr=saving_kwh * tariff,
                category='time'
            ))
        
        # 3. AC recommendations (if applicable)
        if 'ac' in appliances:
            ac_saving_kwh = 0.9 * 0.15  # 15% saving from AC optimization
            recommendations.append(Recommendation(
                action="Set AC ke suhu 24-26°C",
                reason="Setiap 1°C lebih rendah = 6% lebih boros. Suhu 24-26°C optimal.",
                priority=Priority.MEDIUM,
                potential_saving_kwh=ac_saving_kwh,
                potential_saving_idr=ac_saving_kwh * 8 * tariff,  # 8 hours/day
                category='appliance'
            ))
        
        # 4. House type specific
        if house_type == 'working_class':
            recommendations.append(Recommendation(
                action="Gunakan timer untuk AC dan water heater",
                reason="Rumah kosong siang hari. Timer mencegah pemborosan.",
                priority=Priority.LOW,
                potential_saving_kwh=1.0,
                potential_saving_idr=1.0 * tariff,
                category='behavior'
            ))
        elif house_type == 'retired':
            recommendations.append(Recommendation(
                action="Manfaatkan cahaya alami siang hari",
                reason="Di rumah seharian? Kurangi lampu dengan buka tirai.",
                priority=Priority.LOW,
                potential_saving_kwh=0.3,
                potential_saving_idr=0.3 * tariff,
                category='behavior'
            ))
        
        # 5. General efficiency tips
        if prediction.utilization_percent > 50:
            recommendations.append(Recommendation(
                action="Cabut charger dan adaptor saat tidak dipakai",
                reason="Phantom load bisa 5-10% dari tagihan listrik.",
                priority=Priority.LOW,
                potential_saving_kwh=0.2,
                potential_saving_idr=0.2 * 30 * tariff,  # Monthly
                category='behavior'
            ))
        
        # Sort by priority
        priority_order = {Priority.HIGH: 0, Priority.MEDIUM: 1, Priority.LOW: 2}
        recommendations.sort(key=lambda x: priority_order[x.priority])
        
        return recommendations


class UCIRecommendationEngine(RecommendationEngine):
    """
    Recommendation engine for UCI (International) dataset.
    
    Features:
    - Time-based recommendations
    - Sub-metering optimization
    - Seasonal adjustments
    """
    
    def predict_risk(
        self,
        current_power: float,
        va_capacity: int
    ) -> Tuple[RiskLevel, str]:
        """
        Predict risk level for UCI dataset.
        UCI doesn't have VA/MCB, so we use general thresholds.
        """
        # UCI average is around 1-2 kW, max around 10 kW
        if current_power > 8:
            return (RiskLevel.HIGH, "⚠️ High power consumption detected!")
        elif current_power > 5:
            return (RiskLevel.MEDIUM, "⚡ Above average consumption")
        else:
            return (RiskLevel.LOW, "✅ Normal consumption")
    
    def generate_recommendations(
        self,
        prediction: PredictionResult,
        user_data: Dict[str, Any]
    ) -> List[Recommendation]:
        """Generate recommendations for UCI users."""
        recommendations = []
        hour = user_data.get('hour', 12)
        month = user_data.get('month', 6)
        
        # Assume standard tariff for international
        tariff = 0.15  # USD/kWh (average)
        
        # 1. Time-based
        if 18 <= hour <= 21:
            recommendations.append(Recommendation(
                action="Shift high-power activities to off-peak hours",
                reason="Evening hours (6-9 PM) typically have highest consumption.",
                priority=Priority.MEDIUM,
                potential_saving_kwh=0.5,
                potential_saving_idr=0.5 * tariff * 15000,  # Convert to IDR
                category='time'
            ))
        
        # 2. Seasonal (winter = high heating)
        if month in [12, 1, 2]:  # Winter
            recommendations.append(Recommendation(
                action="Optimize heating: lower thermostat by 1-2°C",
                reason="Winter months show highest consumption. Small adjustments help.",
                priority=Priority.MEDIUM,
                potential_saving_kwh=1.0,
                potential_saving_idr=1.0 * tariff * 15000,
                category='behavior'
            ))
        elif month in [6, 7, 8]:  # Summer
            recommendations.append(Recommendation(
                action="Use natural ventilation when possible",
                reason="Summer cooling can be reduced with proper ventilation.",
                priority=Priority.LOW,
                potential_saving_kwh=0.5,
                potential_saving_idr=0.5 * tariff * 15000,
                category='behavior'
            ))
        
        # 3. Sub-metering based (if data available)
        sub_metering = user_data.get('sub_metering', {})
        if sub_metering.get('sub_3', 0) > 5:  # High AC/water heater
            recommendations.append(Recommendation(
                action="Review water heater and AC usage",
                reason="Sub-metering 3 (water heater/AC) shows high consumption.",
                priority=Priority.HIGH,
                potential_saving_kwh=1.5,
                potential_saving_idr=1.5 * tariff * 15000,
                category='appliance'
            ))
        
        return recommendations


class AdaptiveRecommendationEngine(RecommendationEngine):
    """
    Adaptive recommendation engine that works with both datasets.
    Automatically selects appropriate strategy based on data type.
    """
    
    def __init__(self, dataset_type: str = 'auto', verbose: bool = True):
        super().__init__(verbose)
        self.dataset_type = dataset_type
        self._PELITA_engine = PELITARecommendationEngine(verbose=False)
        self._uci_engine = UCIRecommendationEngine(verbose=False)
    
    def detect_dataset_type(self, user_data: Dict[str, Any]) -> str:
        """Auto-detect dataset type from user data"""
        if 'va' in user_data or 'mcb_tripped' in user_data:
            return 'PELITA'
        return 'uci'
    
    def predict_risk(
        self,
        current_power: float,
        va_capacity: int
    ) -> Tuple[RiskLevel, str]:
        """Predict risk using appropriate engine"""
        if self.dataset_type == 'PELITA' or va_capacity in [900, 1300, 2200]:
            return self._PELITA_engine.predict_risk(current_power, va_capacity)
        return self._uci_engine.predict_risk(current_power, va_capacity)
    
    def generate_recommendations(
        self,
        prediction: PredictionResult,
        user_data: Dict[str, Any]
    ) -> List[Recommendation]:
        """Generate recommendations using appropriate engine"""
        if self.dataset_type == 'auto':
            self.dataset_type = self.detect_dataset_type(user_data)
        
        self._log(f"Using {self.dataset_type.upper()} recommendation strategy")
        
        if self.dataset_type == 'PELITA':
            return self._PELITA_engine.generate_recommendations(prediction, user_data)
        return self._uci_engine.generate_recommendations(prediction, user_data)
