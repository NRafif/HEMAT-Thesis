"""
Energy Prediction Models - Core Modules
========================================

This package contains modular components for energy consumption prediction:
- data_loaders: UCI, PELITA, and Universal data loaders
- feature_engineering: Feature engineering for different datasets
- models: Model classes (Delta, Nadir, Epsilon)
- evaluation: Model evaluation and comparison
- recommendations: Recommendation engines

Author: Nofal Rafif
Project: Household Energy Consumption Prediction System
"""

__version__ = '1.0.0'
__author__ = 'Nofal Rafif'

# Import main classes for easy access
from .data_loaders import UCILoader, PELITALoader, UniversalLoader
from .feature_engineering import (
    UCIFeatureEngineer,
    PELITAFeatureEngineer,
    UniversalFeatureEngineer
)
from .models import DeltaModel, NadirModel, EpsilonModel
from .evaluation import ModelEvaluator, ComparativeAnalyzer

# Recommendation engines
from .recommendations import (
    UCIRecommendationEngine,
    PELITARecommendationEngine,
    AdaptiveRecommendationEngine,
    ModelSelector,
    PLNTariff,
    RiskLevel,
    Priority,
    Recommendation,
    PredictionResult,
)

__all__ = [
    # Data Loaders
    'UCILoader',
    'PELITALoader',
    'UniversalLoader',
    # Feature Engineering
    'UCIFeatureEngineer',
    'PELITAFeatureEngineer',
    'UniversalFeatureEngineer',
    # Models
    'DeltaModel',
    'NadirModel',
    'EpsilonModel',
    # Evaluation
    'ModelEvaluator',
    'ComparativeAnalyzer',
    # Recommendations
    'UCIRecommendationEngine',
    'PELITARecommendationEngine',
    'AdaptiveRecommendationEngine',
    'ModelSelector',
    'PLNTariff',
    'RiskLevel',
    'Priority',
    'Recommendation',
    'PredictionResult',
]
