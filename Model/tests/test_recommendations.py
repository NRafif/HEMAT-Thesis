"""
Unit tests for Recommendation System
"""

import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from modules.recommendations import (
    PELITARecommendationEngine,
    UCIRecommendationEngine,
    AdaptiveRecommendationEngine,
    ModelSelector,
    PLNTariff,
    RiskLevel,
    Priority,
    PredictionResult,
)


class TestPLNTariff:
    """Tests for PLN tariff calculator"""
    
    def test_get_tariff_valid_va(self):
        """Test tariff for valid VA values"""
        assert PLNTariff.get_tariff(900) == 1352
        assert PLNTariff.get_tariff(1300) == 1444
        assert PLNTariff.get_tariff(2200) == 1444
    
    def test_get_tariff_invalid_va(self):
        """Test tariff for invalid VA returns default"""
        assert PLNTariff.get_tariff(999) == 1699  # Default
    
    def test_calculate_cost(self):
        """Test cost calculation"""
        cost = PLNTariff.calculate_cost(10, 1300)  # 10 kWh
        assert cost == 10 * 1444
    
    def test_estimate_monthly_bill(self):
        """Test monthly bill estimation"""
        monthly = PLNTariff.estimate_monthly_bill(20, 900)  # 20 kWh/day
        assert monthly == 20 * 30 * 1352


class TestModelSelector:
    """Tests for model selector"""
    
    def test_recommend_epsilon_when_similar(self):
        """When models have similar performance, recommend Epsilon"""
        selector = ModelSelector()
        results = {
            'Delta': {'R2': 0.99},
            'Nadir': {'R2': 0.98},
            'Epsilon': {'R2': 0.985}
        }
        rec = selector.recommend_model(results)
        assert rec['recommended'] == 'Epsilon'
        assert rec['allow_manual'] == False
    
    def test_recommend_best_when_different(self):
        """When one model is much better, recommend it"""
        selector = ModelSelector()
        results = {
            'Delta': {'R2': 0.95},
            'Nadir': {'R2': 0.70},
            'Epsilon': {'R2': 0.80}
        }
        rec = selector.recommend_model(results)
        assert rec['recommended'] == 'Delta'
        assert rec['allow_manual'] == True


class TestPELITARecommendationEngine:
    """Tests for PELITA recommendation engine"""
    
    @pytest.fixture
    def engine(self):
        return PELITARecommendationEngine(verbose=False)
    
    def test_predict_risk_low(self, engine):
        """Test low risk prediction"""
        risk, msg = engine.predict_risk(0.3, 900)
        assert risk == RiskLevel.LOW
        assert "normal" in msg.lower() or "aman" in msg.lower()
    
    def test_predict_risk_high(self, engine):
        """Test high risk prediction when near capacity"""
        risk, msg = engine.predict_risk(0.7, 900)  # ~92% of 900VA
        assert risk in [RiskLevel.HIGH, RiskLevel.CRITICAL]
    
    def test_predict_risk_critical(self, engine):
        """Test critical risk when over capacity"""
        risk, msg = engine.predict_risk(0.8, 900)  # ~105% of 900VA
        assert risk == RiskLevel.CRITICAL
        assert "padam" in msg.lower() or "bahaya" in msg.lower()
    
    def test_suggest_load_shedding(self, engine):
        """Test load shedding suggestions"""
        active = ['ac', 'water_heater', 'tv']
        suggestions = engine.suggest_load_shedding(0.8, 900, active)
        
        # Should suggest turning off high-power appliances first
        assert len(suggestions) > 0
        assert suggestions[0]['appliance'] == 'water_heater'  # Highest priority
    
    def test_estimate_savings(self, engine):
        """Test savings estimation"""
        savings = engine.estimate_savings(20, 0.10, 1300)  # 10% reduction
        
        assert savings['daily_kwh'] == 2.0
        assert savings['monthly_kwh'] == 60.0
        assert savings['daily_idr'] == 2.0 * 1444
    
    def test_generate_recommendations(self, engine):
        """Test full recommendation generation"""
        prediction = PredictionResult(
            current_power_kw=0.7,
            predicted_power_kw=0.75,
            va_capacity=900,
            utilization_percent=92,
            risk_level=RiskLevel.HIGH,
            estimated_daily_kwh=16,
            estimated_daily_idr=16 * 1352
        )
        
        user_data = {
            'va': 900,
            'house_type': 'working_class',
            'hour': 19,
            'appliances': ['ac', 'kulkas']
        }
        
        recs = engine.generate_recommendations(prediction, user_data)
        
        assert len(recs) > 0
        # First recommendation should be HIGH priority (MCB warning)
        assert recs[0].priority == Priority.HIGH


class TestAdaptiveRecommendationEngine:
    """Tests for adaptive recommendation engine"""
    
    def test_detect_PELITA_data(self):
        """Test detection of PELITA data"""
        engine = AdaptiveRecommendationEngine(verbose=False)
        
        PELITA_data = {'va': 1300, 'hour': 12}
        assert engine.detect_dataset_type(PELITA_data) == 'PELITA'
    
    def test_detect_uci_data(self):
        """Test detection of UCI data"""
        engine = AdaptiveRecommendationEngine(verbose=False)
        
        uci_data = {'hour': 12, 'month': 6}
        assert engine.detect_dataset_type(uci_data) == 'uci'
    
    def test_predict_risk_PELITA(self):
        """Test risk prediction for PELITA-like data"""
        engine = AdaptiveRecommendationEngine(dataset_type='PELITA', verbose=False)
        
        risk, msg = engine.predict_risk(0.7, 900)
        assert risk in [RiskLevel.HIGH, RiskLevel.CRITICAL]


class TestUCIRecommendationEngine:
    """Tests for UCI recommendation engine"""
    
    @pytest.fixture
    def engine(self):
        return UCIRecommendationEngine(verbose=False)
    
    def test_predict_risk_low(self, engine):
        """Test low risk for normal consumption"""
        risk, msg = engine.predict_risk(2.0, 0)
        assert risk == RiskLevel.LOW
    
    def test_predict_risk_high(self, engine):
        """Test high risk for high consumption"""
        risk, msg = engine.predict_risk(9.0, 0)
        assert risk == RiskLevel.HIGH
