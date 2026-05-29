"""
Data Loaders for Energy Prediction Models
==========================================

This module provides data loaders for different dataset formats:
- UCILoader: For UCI Individual Household Electric Power Consumption dataset
- PELITALoader: For PELITA Indonesia synthetic dataset
- UniversalLoader: Auto-detect and load both formats

Author: Nofal Rafif
"""

import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any
import warnings


class DatasetLoader(ABC):
    """
    Abstract base class for dataset loaders.
    
    All loaders must implement load() and validate() methods.
    """
    
    def __init__(self, verbose: bool = True):
        """
        Initialize the loader.
        
        Args:
            verbose: If True, print loading progress and warnings
        """
        self.verbose = verbose
        self.dataset_info: Dict[str, Any] = {}
    
    @abstractmethod
    def load(self, path: str) -> pd.DataFrame:
        """
        Load dataset from file.
        
        Args:
            path: Path to the dataset file
            
        Returns:
            DataFrame with Datetime index and processed columns
            
        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If file format is invalid
        """
        pass
    
    @abstractmethod
    def validate(self, df: pd.DataFrame) -> bool:
        """
        Validate that DataFrame has required columns and structure.
        
        Args:
            df: DataFrame to validate
            
        Returns:
            True if valid, raises exception otherwise
            
        Raises:
            KeyError: If required columns are missing
            ValueError: If data structure is invalid
        """
        pass
    
    def _log(self, message: str):
        """Print message if verbose mode is enabled"""
        if self.verbose:
            print(f"[{self.__class__.__name__}] {message}")
    
    def get_info(self) -> Dict[str, Any]:
        """
        Get information about the loaded dataset.
        
        Returns:
            Dictionary with dataset statistics and metadata
        """
        return self.dataset_info


class UCILoader(DatasetLoader):
    """
    Loader for UCI Individual Household Electric Power Consumption dataset.
    
    Expected format:
    - Semicolon-separated values
    - Date and Time columns (need to be merged)
    - Missing values marked as '?'
    - Minute-level data (needs hourly resampling)
    """
    
    REQUIRED_COLUMNS = [
        'Global_active_power',
        'Global_reactive_power',
        'Voltage',
        'Global_intensity',
        'Sub_metering_1',
        'Sub_metering_2',
        'Sub_metering_3'
    ]
    
    def load(self, path: str) -> pd.DataFrame:
        """
        Load UCI dataset with preprocessing.
        
        Steps:
        1. Read semicolon-separated file
        2. Merge Date and Time columns
        3. Convert to numeric
        4. Handle missing values (forward fill)
        5. Apply IQR outlier clipping
        6. Resample to hourly
        
        Args:
            path: Path to household_power_consumption.txt
            
        Returns:
            DataFrame with hourly data and Datetime index
        """
        self._log(f"Loading UCI dataset from {path}")
        
        try:
            # Read with semicolon separator
            df = pd.read_csv(path, sep=';', na_values='?', low_memory=False)
            self._log(f"Loaded {len(df):,} rows")
            
        except FileNotFoundError:
            raise FileNotFoundError(
                f"UCI dataset not found at {path}. "
                "Please download from UCI ML Repository."
            )
        except Exception as e:
            raise ValueError(f"Error reading UCI dataset: {str(e)}")
        
        # Merge Date and Time
        self._log("Merging Date and Time columns...")
        try:
            df['Datetime'] = pd.to_datetime(
                df['Date'] + ' ' + df['Time'],
                format='%d/%m/%Y %H:%M:%S'
            )
        except Exception as e:
            raise ValueError(f"Error parsing datetime: {str(e)}")
        
        df = df.set_index('Datetime').drop(['Date', 'Time'], axis=1)
        
        # Convert to numeric
        self._log("Converting to numeric...")
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Forward fill missing values
        missing_before = df.isnull().sum().sum()
        df = df.ffill()
        self._log(f"Filled {missing_before:,} missing values")
        
        # IQR outlier clipping
        self._log("Applying IQR outlier clipping...")
        df = self._clip_outliers_iqr(df)
        
        # Resample to hourly
        self._log("Resampling to hourly...")
        df = self._resample_hourly(df)
        
        # Validate
        self.validate(df)
        
        # Store info
        self.dataset_info = {
            'type': 'UCI',
            'rows': len(df),
            'start_date': df.index.min(),
            'end_date': df.index.max(),
            'duration_days': (df.index.max() - df.index.min()).days,
            'columns': list(df.columns)
        }
        
        self._log(f"✓ UCI dataset loaded: {len(df):,} hours")
        return df
    
    def _clip_outliers_iqr(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply IQR method to clip outliers"""
        for col in df.select_dtypes(include=[np.number]).columns:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            df[col] = np.clip(df[col], lower_bound, upper_bound)
        return df
    
    def _resample_hourly(self, df: pd.DataFrame) -> pd.DataFrame:
        """Resample minute-level data to hourly"""
        resample_rules = {
            'Global_active_power': 'sum',
            'Global_reactive_power': 'sum',
            'Voltage': 'mean',
            'Global_intensity': 'mean',
            'Sub_metering_1': 'sum',
            'Sub_metering_2': 'sum',
            'Sub_metering_3': 'sum'
        }
        
        # Use 'h' instead of deprecated 'H'
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            df_resampled = df.resample('h').agg(resample_rules)
        
        return df_resampled
    
    def validate(self, df: pd.DataFrame) -> bool:
        """Validate UCI dataset structure"""
        # Check required columns
        missing_cols = set(self.REQUIRED_COLUMNS) - set(df.columns)
        if missing_cols:
            raise KeyError(
                f"Missing required columns: {missing_cols}. "
                f"Expected: {self.REQUIRED_COLUMNS}"
            )
        
        # Check datetime index
        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError("Index must be DatetimeIndex")
        
        # Check for monotonic increasing
        if not df.index.is_monotonic_increasing:
            raise ValueError("Datetime index must be monotonically increasing")
        
        return True


class PELITALoader(DatasetLoader):
    """
    Loader for PELITA Indonesia synthetic dataset.
    
    Expected format:
    - Comma-separated values
    - Datetime column already formatted
    - Hourly data (no resampling needed)
    - PELITA-specific columns: VA, MCB_Tripped, House_Type, Emisi_CO2
    - Pre-generated lag and rolling features
    """
    
    REQUIRED_COLUMNS = [
        'Global_active_power',
        'Global_reactive_power',
        'Voltage',
        'Global_intensity',
        'Sub_metering_1',
        'Sub_metering_2',
        'Sub_metering_3',
        'Sub_metering_4',
        'MCB_Tripped',
        'VA',
        'House_Type'
    ]
    
    PELITA_SPECIFIC = [
        'MCB_Tripped',
        'VA',
        'House_Type',
        'House_ID',
        'Emisi_CO2',
        'Lag_1',
        'Lag_24',
        'Lag_168',
        'Rolling_mean_3',
        'Rolling_mean_24',
        'Rolling_std_24'
    ]
    
    def load(self, path: str) -> pd.DataFrame:
        """
        Load PELITA dataset with minimal preprocessing.
        
        Steps:
        1. Read comma-separated file
        2. Parse existing Datetime column
        3. Convert to numeric
        4. Conditional outlier handling (preserve MCB trip zeros)
        5. NO resampling (already hourly)
        
        Args:
            path: Path to dataset_energi_rumah_indonesia_PELITA_Fix.csv
            
        Returns:
            DataFrame with hourly data and Datetime index
        """
        self._log(f"Loading PELITA dataset from {path}")
        
        try:
            # Read with comma separator
            df = pd.read_csv(path, low_memory=False)
            self._log(f"Loaded {len(df):,} rows")
            
        except FileNotFoundError:
            raise FileNotFoundError(
                f"PELITA dataset not found at {path}. "
                "Please generate using Generate_Dataset_PELITA.py"
            )
        except Exception as e:
            raise ValueError(f"Error reading PELITA dataset: {str(e)}")
        
        # Parse Datetime column (already formatted)
        self._log("Parsing Datetime column...")
        try:
            df['Datetime'] = pd.to_datetime(df['Datetime'])
        except Exception as e:
            raise ValueError(f"Error parsing datetime: {str(e)}")
        
        df = df.set_index('Datetime')
        
        # Convert numeric columns
        self._log("Converting to numeric...")
        numeric_cols = df.select_dtypes(include=['object']).columns
        numeric_cols = [col for col in numeric_cols if col != 'House_Type']
        
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Conditional outlier handling (preserve MCB trip behavior)
        self._log("Applying conditional outlier handling...")
        df = self._conditional_outlier_handling(df)
        
        # NO resampling - already hourly
        self._log("Data is already hourly - skipping resampling")
        
        # Validate
        self.validate(df)
        
        # Store info
        self.dataset_info = {
            'type': 'PELITA',
            'rows': len(df),
            'houses': df['House_ID'].nunique() if 'House_ID' in df.columns else 1,
            'start_date': df.index.min(),
            'end_date': df.index.max(),
            'duration_days': (df.index.max() - df.index.min()).days,
            'va_distribution': df['VA'].value_counts().to_dict() if 'VA' in df.columns else {},
            'mcb_trip_rate': (df['MCB_Tripped'].mean() * 100) if 'MCB_Tripped' in df.columns else 0,
            'columns': list(df.columns)
        }
        
        self._log(f"✓ PELITA dataset loaded: {len(df):,} hours from {self.dataset_info['houses']} houses")
        self._log(f"  MCB trip rate: {self.dataset_info['mcb_trip_rate']:.2f}%")
        
        return df
    
    def _conditional_outlier_handling(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply outlier handling that preserves MCB trip behavior.
        
        CRITICAL: When MCB trips, power=0 is VALID, not an outlier!
        Only clip outliers for normal operation (MCB_Tripped=0)
        """
        if 'MCB_Tripped' not in df.columns:
            self._log("Warning: MCB_Tripped column not found, skipping outlier handling")
            return df
        
        # Separate normal and tripped data
        normal_mask = df['MCB_Tripped'] == 0
        normal_data = df[normal_mask].copy()
        
        # Apply IQR clipping ONLY to normal operation data
        power_cols = [
            'Global_active_power',
            'Global_reactive_power',
            'Global_intensity',
            'Sub_metering_1',
            'Sub_metering_2',
            'Sub_metering_3',
            'Sub_metering_4'
        ]
        
        for col in power_cols:
            if col in normal_data.columns:
                Q1 = normal_data[col].quantile(0.25)
                Q3 = normal_data[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                # Clip only normal operation data
                df.loc[normal_mask, col] = np.clip(
                    df.loc[normal_mask, col],
                    lower_bound,
                    upper_bound
                )
        
        # Voltage can be clipped for all data (PLN range)
        if 'Voltage' in df.columns:
            df['Voltage'] = np.clip(df['Voltage'], 198, 231)  # PLN: +5%/-10%
        
        return df
    
    def validate(self, df: pd.DataFrame) -> bool:
        """Validate PELITA dataset structure"""
        # Check required columns
        missing_cols = set(self.REQUIRED_COLUMNS) - set(df.columns)
        if missing_cols:
            raise KeyError(
                f"Missing required PELITA columns: {missing_cols}. "
                f"Expected: {self.REQUIRED_COLUMNS}"
            )
        
        # Check datetime index
        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError("Index must be DatetimeIndex")
        
        # Note: PELITA dataset has multiple houses, so datetime may not be monotonic
        # We skip the monotonic check for PELITA as it contains data from multiple houses
        # Each house has its own time series that overlaps with others
        
        # Validate VA values
        if 'VA' in df.columns:
            valid_va = {900, 1300, 2200}
            actual_va = set(df['VA'].unique())
            if not actual_va.issubset(valid_va):
                raise ValueError(
                    f"Invalid VA values: {actual_va - valid_va}. "
                    f"Expected: {valid_va}"
                )
        
        # Validate MCB_Tripped values
        if 'MCB_Tripped' in df.columns:
            valid_mcb = {0, 1}
            actual_mcb = set(df['MCB_Tripped'].unique())
            if not actual_mcb.issubset(valid_mcb):
                raise ValueError(
                    f"Invalid MCB_Tripped values: {actual_mcb - valid_mcb}. "
                    f"Expected: {valid_mcb} (0=normal, 1=tripped)"
                )
        
        # Validate House_Type values
        if 'House_Type' in df.columns:
            valid_types = {'working_class', 'retired', 'mixed'}
            actual_types = set(df['House_Type'].unique())
            if not actual_types.issubset(valid_types):
                raise ValueError(
                    f"Invalid House_Type values: {actual_types - valid_types}. "
                    f"Expected: {valid_types}"
                )
        
        return True
    
    def get_va_statistics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Get statistics by VA class.
        
        Args:
            df: PELITA DataFrame
            
        Returns:
            Dictionary with statistics per VA class
        """
        if 'VA' not in df.columns:
            return {}
        
        stats = {}
        for va in [900, 1300, 2200]:
            va_data = df[df['VA'] == va]
            if len(va_data) > 0:
                stats[f'{va}VA'] = {
                    'count': len(va_data),
                    'mean_power': va_data['Global_active_power'].mean(),
                    'max_power': va_data['Global_active_power'].max(),
                    'mcb_trip_rate': (va_data['MCB_Tripped'].mean() * 100) if 'MCB_Tripped' in va_data.columns else 0,
                    'utilization': (va_data['Global_active_power'].mean() * 1000 / va * 100)  # % of capacity
                }
        
        return stats


class UniversalLoader(DatasetLoader):
    """
    Universal loader that auto-detects dataset type and delegates to appropriate loader.
    
    Detection logic:
    - If columns contain 'MCB_Tripped', 'VA', 'House_Type' → PELITA
    - Otherwise → UCI
    
    This allows Model Epsilon to work with both datasets seamlessly.
    """
    
    def __init__(self, verbose: bool = True):
        """Initialize universal loader"""
        super().__init__(verbose)
        self.detected_type: Optional[str] = None
        self.delegate_loader: Optional[DatasetLoader] = None
    
    def detect_type(self, path: str) -> str:
        """
        Detect dataset type by reading first few rows and checking columns.
        
        Args:
            path: Path to dataset file
            
        Returns:
            'PELITA' or 'uci'
        """
        self._log("Detecting dataset type...")
        
        try:
            # Try reading first few rows with different separators
            # Try comma first (PELITA)
            try:
                df_sample = pd.read_csv(path, nrows=5)
                columns = set(df_sample.columns)
            except:
                # Try semicolon (UCI)
                df_sample = pd.read_csv(path, sep=';', nrows=5)
                columns = set(df_sample.columns)
            
            # Check for PELITA-specific columns
            PELITA_markers = {'MCB_Tripped', 'VA', 'House_Type'}
            
            if PELITA_markers.issubset(columns):
                self._log("✓ Detected: PELITA dataset (Indonesia)")
                return 'PELITA'
            else:
                self._log("✓ Detected: UCI dataset (International)")
                return 'uci'
                
        except Exception as e:
            raise ValueError(f"Error detecting dataset type: {str(e)}")
    
    def load(self, path: str) -> pd.DataFrame:
        """
        Auto-detect dataset type and load using appropriate loader.
        
        Args:
            path: Path to dataset file
            
        Returns:
            DataFrame loaded by appropriate loader
        """
        # Detect type
        self.detected_type = self.detect_type(path)
        
        # Delegate to appropriate loader
        if self.detected_type == 'PELITA':
            self.delegate_loader = PELITALoader(verbose=self.verbose)
        else:
            self.delegate_loader = UCILoader(verbose=self.verbose)
        
        # Load using delegate
        df = self.delegate_loader.load(path)
        
        # Copy info from delegate
        self.dataset_info = self.delegate_loader.get_info()
        self.dataset_info['detected_type'] = self.detected_type
        
        return df
    
    def validate(self, df: pd.DataFrame) -> bool:
        """
        Validate using delegate loader's validation.
        
        Args:
            df: DataFrame to validate
            
        Returns:
            True if valid
        """
        if self.delegate_loader is None:
            raise RuntimeError("Must call load() before validate()")
        
        return self.delegate_loader.validate(df)
    
    def get_loader_type(self) -> Optional[str]:
        """
        Get the detected dataset type.
        
        Returns:
            'PELITA', 'uci', or None if not yet detected
        """
        return self.detected_type
