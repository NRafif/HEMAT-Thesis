"""
Feature Engineering for Energy Prediction Models
=================================================

Author: Nofal Rafif
"""

import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
from typing import List, Optional


class FeatureEngineer(ABC):
    """Abstract base class for feature engineering"""
    
    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.feature_names: List[str] = []
    
    @abstractmethod
    def engineer(self, df: pd.DataFrame) -> pd.DataFrame:
        pass
    
    def create_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create time-based features"""
        self._log("Creating temporal features...")
        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError("DataFrame must have DatetimeIndex")
        
        df = df.copy()
        df['hour'] = df.index.hour
        df['dayofweek'] = df.index.dayofweek
        df['day'] = df.index.day
        df['month'] = df.index.month
        df['season'] = (df['month'] % 12 // 3) + 1
        
        self.feature_names.extend(['hour', 'dayofweek', 'day', 'month', 'season'])
        self._log(f"✓ Created 5 temporal features")
        return df
    
    def create_lag_features(self, df: pd.DataFrame, target_col: str, lags: List[int] = [1, 24, 168]) -> pd.DataFrame:
        """Create lag features (NO data leakage)"""
        self._log(f"Creating lag features for '{target_col}'...")
        if target_col not in df.columns:
            raise KeyError(f"Target column '{target_col}' not found")
        
        df = df.copy()
        for lag in lags:
            col_name = f'lag_{lag}'
            df[col_name] = df[target_col].shift(lag)
            self.feature_names.append(col_name)
        
        self._log(f"✓ Created {len(lags)} lag features")
        return df

    def create_rolling_features(self, df: pd.DataFrame, target_col: str, windows: List[int] = [3, 24]) -> pd.DataFrame:
        """Create rolling features (NO data leakage)"""
        self._log(f"Creating rolling features for '{target_col}'...")
        if target_col not in df.columns:
            raise KeyError(f"Target column '{target_col}' not found")
        
        df = df.copy()
        shifted = df[target_col].shift(1)  # CRITICAL: prevent data leakage
        
        for window in windows:
            col_name_mean = f'rolling_mean_{window}'
            df[col_name_mean] = shifted.rolling(window=window, min_periods=1).mean()
            self.feature_names.append(col_name_mean)
            
            if window >= 24:
                col_name_std = f'rolling_std_{window}'
                df[col_name_std] = shifted.rolling(window=window, min_periods=1).std()
                self.feature_names.append(col_name_std)
        
        self._log(f"✓ Created rolling features for {len(windows)} windows")
        return df
    
    def fill_missing_values(self, df: pd.DataFrame, strategy: str = 'forward') -> pd.DataFrame:
        """Fill missing values"""
        self._log(f"Filling missing values (strategy: {strategy})...")
        df = df.copy()
        
        if strategy == 'forward':
            df = df.ffill()
        elif strategy == 'mean':
            df = df.fillna(df.mean())
        elif strategy == 'zero':
            df = df.fillna(0)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
        
        rows_before = len(df)
        df = df.dropna()
        rows_dropped = rows_before - len(df)
        if rows_dropped > 0:
            self._log(f"⚠ Dropped {rows_dropped} rows with remaining NaN")
        
        return df
    
    def get_feature_names(self) -> List[str]:
        """Get list of engineered feature names"""
        return self.feature_names
    
    def _log(self, message: str):
        if self.verbose:
            print(f"[{self.__class__.__name__}] {message}")


class UCIFeatureEngineer(FeatureEngineer):
    """Feature engineer for UCI dataset"""
    
    def engineer(self, df: pd.DataFrame) -> pd.DataFrame:
        self._log("Engineering features for UCI dataset...")
        df = self.create_temporal_features(df)
        df = self.create_lag_features(df, target_col='Global_active_power', lags=[1, 24, 168])
        df = self.create_rolling_features(df, target_col='Global_active_power', windows=[3, 24])
        df = self.fill_missing_values(df, strategy='forward')
        self._log(f"✓ UCI feature engineering complete: {len(self.feature_names)} features")
        return df



class PELITAFeatureEngineer(FeatureEngineer):
    """Feature engineer for PELITA dataset"""
    
    def engineer(self, df: pd.DataFrame) -> pd.DataFrame:
        self._log("Engineering features for PELITA dataset...")
        df = df.copy()
        df = self._rename_PELITA_features(df)
        
        if 'hour' not in df.columns:
            df = self.create_temporal_features(df)
        else:
            self._log("Using existing temporal features")
            rename_map = {'Hour': 'hour', 'Dayofweek': 'dayofweek', 'Month': 'month', 'Season': 'season'}
            df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})
        
        if 'VA' in df.columns:
            df = self.normalize_va(df)
        if 'House_Type' in df.columns:
            df = self.encode_house_type(df)
        
        df = self.fill_missing_values(df, strategy='forward')
        self._log(f"✓ PELITA feature engineering complete")
        return df
    
    def _rename_PELITA_features(self, df: pd.DataFrame) -> pd.DataFrame:
        self._log("Renaming PELITA features to lowercase...")
        rename_map = {
            'Lag_1': 'lag_1', 'Lag_24': 'lag_24', 'Lag_168': 'lag_168',
            'Rolling_mean_3': 'rolling_mean_3', 'Rolling_mean_24': 'rolling_mean_24',
            'Rolling_std_24': 'rolling_std_24'
        }
        rename_map = {k: v for k, v in rename_map.items() if k in df.columns}
        if rename_map:
            df = df.rename(columns=rename_map)
            self._log(f"✓ Renamed {len(rename_map)} features")
        return df
    
    def normalize_va(self, df: pd.DataFrame) -> pd.DataFrame:
        self._log("Normalizing VA capacity...")
        if 'VA' not in df.columns:
            self._log("⚠ VA column not found")
            return df
        df = df.copy()
        df['VA_normalized'] = df['VA'] / 2200.0
        self.feature_names.append('VA_normalized')
        self._log("✓ VA normalized")
        return df
    
    def encode_house_type(self, df: pd.DataFrame) -> pd.DataFrame:
        self._log("Encoding House_Type...")
        if 'House_Type' not in df.columns:
            self._log("⚠ House_Type column not found")
            return df
        df = pd.get_dummies(df, columns=['House_Type'], prefix='HT')
        ht_cols = [col for col in df.columns if col.startswith('HT_')]
        self.feature_names.extend(ht_cols)
        self._log(f"✓ Encoded House_Type into {len(ht_cols)} columns")
        return df


class UniversalFeatureEngineer(FeatureEngineer):
    """Universal feature engineer with auto-detection"""
    
    def __init__(self, dataset_type: Optional[str] = None, verbose: bool = True):
        super().__init__(verbose)
        self.dataset_type = dataset_type
        self.delegate_engineer: Optional[FeatureEngineer] = None
    
    def detect_type(self, df: pd.DataFrame) -> str:
        PELITA_markers = {'MCB_Tripped', 'VA', 'House_Type'}
        if PELITA_markers.issubset(set(df.columns)):
            return 'PELITA'
        return 'uci'
    
    def engineer(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.dataset_type is None:
            self.dataset_type = self.detect_type(df)
            self._log(f"Auto-detected dataset type: {self.dataset_type.upper()}")
        
        if self.dataset_type == 'PELITA':
            self.delegate_engineer = PELITAFeatureEngineer(verbose=self.verbose)
        else:
            self.delegate_engineer = UCIFeatureEngineer(verbose=self.verbose)
        
        df = self.delegate_engineer.engineer(df)
        self.feature_names = self.delegate_engineer.get_feature_names()
        return df
