"""
Model training module for energy prediction.

This module provides base and specialized model classes for training
Random Forest and XGBoost models on UCI and PELITA datasets.

Classes:
    - EnergyPredictionModel: Base class for all models
    - DeltaModel: UCI-optimized model
    - NadirModel: PELITA-optimized model
    - EpsilonModel: Universal adaptive model
"""

import joblib
import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from typing import Dict, Any, Optional, Tuple
import warnings

warnings.filterwarnings('ignore')


class EnergyPredictionModel(ABC):
    """
    Base class for energy prediction models.
    
    Provides common functionality for training, prediction, and persistence.
    Subclasses must implement get_param_grid() for hyperparameter tuning.
    
    Attributes:
        model_type (str): Type of model ('rf' for Random Forest, 'xgb' for XGBoost)
        random_state (int): Random seed for reproducibility
        model: Trained model instance
        best_params (dict): Best hyperparameters from GridSearchCV
        cv_results (dict): Cross-validation results
        feature_names (list): Names of features used for training
    """
    
    def __init__(self, model_type: str = 'rf', random_state: int = 42):
        """
        Initialize the model.
        
        Args:
            model_type: Type of model ('rf' or 'xgb')
            random_state: Random seed for reproducibility
        """
        self.model_type = model_type
        self.random_state = random_state
        self.model = None
        self.best_params = None
        self.cv_results = None
        self.feature_names = None
    
    @abstractmethod
    def get_param_grid(self) -> Dict[str, list]:
        """
        Get hyperparameter grid for GridSearchCV.
        
        Returns:
            Dictionary with parameter names as keys and lists of values to try
        """
        pass
    
    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        cv_splits: int = 5,
        n_jobs: int = -1,
        verbose: int = 0
    ) -> 'EnergyPredictionModel':
        """
        Train the model using GridSearchCV with TimeSeriesSplit.
        
        Args:
            X_train: Training features
            y_train: Training target
            cv_splits: Number of cross-validation splits
            n_jobs: Number of parallel jobs (-1 for all cores)
            verbose: Verbosity level for GridSearchCV
        
        Returns:
            Self for method chaining
        """
        # Store feature names
        if isinstance(X_train, pd.DataFrame):
            self.feature_names = X_train.columns.tolist()
        else:
            self.feature_names = [f'feature_{i}' for i in range(X_train.shape[1])]
        
        # Create base model
        if self.model_type == 'rf':
            base_model = RandomForestRegressor(random_state=self.random_state)
        else:
            raise ValueError(f"Unsupported model_type: {self.model_type}")
        
        # Setup cross-validation strategy
        tscv = TimeSeriesSplit(n_splits=cv_splits)
        
        # Get parameter grid
        param_grid = self.get_param_grid()
        
        # Perform grid search
        grid_search = GridSearchCV(
            estimator=base_model,
            param_grid=param_grid,
            cv=tscv,
            scoring='neg_mean_absolute_error',
            n_jobs=n_jobs,
            verbose=verbose,
            return_train_score=True
        )
        
        # Fit the model
        grid_search.fit(X_train, y_train)
        
        # Store results
        self.model = grid_search.best_estimator_
        self.best_params = grid_search.best_params_
        self.cv_results = grid_search.cv_results_
        
        return self
    
    def predict(self, X_test: pd.DataFrame) -> np.ndarray:
        """
        Make predictions on test data.
        
        Args:
            X_test: Test features
        
        Returns:
            Array of predictions
        
        Raises:
            ValueError: If model hasn't been trained yet
        """
        if self.model is None:
            raise ValueError("Model must be trained before making predictions")
        
        return self.model.predict(X_test)
    
    def get_feature_importance(self, top_n: Optional[int] = None) -> pd.DataFrame:
        """
        Get feature importance from trained model.
        
        Args:
            top_n: Number of top features to return (None for all)
        
        Returns:
            DataFrame with features and their importance scores
        
        Raises:
            ValueError: If model hasn't been trained yet
        """
        if self.model is None:
            raise ValueError("Model must be trained before getting feature importance")
        
        if not hasattr(self.model, 'feature_importances_'):
            raise ValueError("Model doesn't support feature importance")
        
        importance_df = pd.DataFrame({
            'Feature': self.feature_names,
            'Importance': self.model.feature_importances_
        }).sort_values('Importance', ascending=False)
        
        if top_n is not None:
            importance_df = importance_df.head(top_n)
        
        return importance_df
    
    def save(self, filepath: str) -> None:
        """
        Save trained model to disk using joblib.
        
        Args:
            filepath: Path to save the model (e.g., 'model.joblib')
        
        Raises:
            ValueError: If model hasn't been trained yet
        """
        if self.model is None:
            raise ValueError("Model must be trained before saving")
        
        model_data = {
            'model': self.model,
            'model_type': self.model_type,
            'best_params': self.best_params,
            'feature_names': self.feature_names,
            'cv_results': self.cv_results
        }
        
        joblib.dump(model_data, filepath)
    
    @classmethod
    def load(cls, filepath: str) -> 'EnergyPredictionModel':
        """
        Load trained model from disk.
        
        Args:
            filepath: Path to the saved model
        
        Returns:
            Loaded model instance
        """
        model_data = joblib.load(filepath)
        
        # Create instance
        instance = cls(model_type=model_data['model_type'])
        instance.model = model_data['model']
        instance.best_params = model_data['best_params']
        instance.feature_names = model_data['feature_names']
        instance.cv_results = model_data.get('cv_results')
        
        return instance


class DeltaModel(EnergyPredictionModel):
    """
    Model Delta - Optimized for UCI dataset.
    
    This model is tuned for the UCI household power consumption dataset
    with features like sub-metering and French household characteristics.
    """
    
    def get_param_grid(self) -> Dict[str, list]:
        """
        Get hyperparameter grid optimized for UCI dataset.
        
        Returns:
            Parameter grid for GridSearchCV
        """
        return {
            'n_estimators': [100, 200],
            'max_depth': [10, 20, None],
            'min_samples_split': [2, 5],
            'min_samples_leaf': [1, 2],
            'max_features': ['sqrt', 'log2']
        }


class NadirModel(EnergyPredictionModel):
    """
    Model Nadir - Optimized for PELITA dataset.
    
    This model is tuned for the PELITA Indonesian household dataset
    with features like VA capacity, MCB trips, and house types.
    """
    
    def get_param_grid(self) -> Dict[str, list]:
        """
        Get hyperparameter grid optimized for PELITA dataset.
        
        Returns:
            Parameter grid for GridSearchCV
        """
        return {
            'n_estimators': [150, 250, 300],
            'max_depth': [15, 25, None],
            'min_samples_split': [2, 3, 5],
            'min_samples_leaf': [1, 2, 3],  # More conservative for MCB events
            'max_features': ['sqrt', 'log2', None]
        }
    
    def train_stratified(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        va_column: str = 'VA',
        cv_splits: int = 5,
        n_jobs: int = -1,
        verbose: int = 0
    ) -> 'NadirModel':
        """
        Train with stratification by VA class for balanced learning.
        
        Args:
            X_train: Training features (must include VA column)
            y_train: Training target
            va_column: Name of VA column for stratification
            cv_splits: Number of cross-validation splits
            n_jobs: Number of parallel jobs
            verbose: Verbosity level
        
        Returns:
            Self for method chaining
        """
        if va_column not in X_train.columns:
            # Fall back to regular training if VA column not found
            return self.train(X_train, y_train, cv_splits, n_jobs, verbose)
        
        # For now, use regular training
        # TODO: Implement proper stratified sampling in future version
        return self.train(X_train, y_train, cv_splits, n_jobs, verbose)


class EpsilonModel(EnergyPredictionModel):
    """
    Model Epsilon - Universal adaptive model.
    
    This model automatically adapts its hyperparameters based on
    the detected dataset type (UCI or PELITA).
    """
    
    def __init__(self, model_type: str = 'rf', random_state: int = 42, dataset_type: str = 'uci'):
        """
        Initialize universal model.
        
        Args:
            model_type: Type of model ('rf' or 'xgb')
            random_state: Random seed
            dataset_type: Type of dataset ('uci' or 'PELITA')
        """
        super().__init__(model_type, random_state)
        self.dataset_type = dataset_type
    
    def get_param_grid(self) -> Dict[str, list]:
        """
        Get adaptive hyperparameter grid based on dataset type.
        
        Returns:
            Parameter grid appropriate for the dataset type
        """
        if self.dataset_type == 'PELITA':
            # Use Nadir's grid for PELITA data
            return NadirModel().get_param_grid()
        else:
            # Use Delta's grid for UCI data
            return DeltaModel().get_param_grid()
    
    def set_dataset_type(self, dataset_type: str) -> None:
        """
        Set the dataset type for adaptive parameter selection.
        
        Args:
            dataset_type: Type of dataset ('uci' or 'PELITA')
        """
        self.dataset_type = dataset_type
