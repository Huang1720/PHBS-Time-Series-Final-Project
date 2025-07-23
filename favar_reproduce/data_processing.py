"""
Data Processing Utilities for FAVAR Model
==========================================

This module provides utilities for loading, preprocessing, and transforming
macroeconomic data for FAVAR analysis.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

class DataProcessor:
    """
    Data preprocessing utilities for FAVAR model
    """
    
    def __init__(self):
        self.transformation_codes = {
            1: 'levels',
            2: 'first_difference',
            3: 'second_difference',
            4: 'log_levels',
            5: 'log_first_difference',
            6: 'log_second_difference'
        }
        
    def load_excel_data(self, file_path, sheet_name=None):
        """
        Load data from Excel file
        
        Parameters:
        -----------
        file_path : str
            Path to Excel file
        sheet_name : str, optional
            Sheet name to load
            
        Returns:
        --------
        pd.DataFrame : Loaded data
        """
        try:
            data = pd.read_excel(file_path, sheet_name=sheet_name, index_col=0)
            # Convert index to datetime if possible
            if not isinstance(data.index, pd.DatetimeIndex):
                try:
                    data.index = pd.to_datetime(data.index)
                except:
                    pass
            
            return data
            
        except Exception as e:
            raise ValueError(f"Error loading data from {file_path}: {e}")
    
    def apply_transformations(self, data, transformation_map):
        """
        Apply transformations to data based on transformation codes
        
        Parameters:
        -----------
        data : pd.DataFrame
            Raw data
        transformation_map : dict
            Dictionary mapping column names to transformation codes
            
        Returns:
        --------
        pd.DataFrame : Transformed data
        """
        transformed_data = data.copy()
        
        for column, trans_code in transformation_map.items():
            if column in data.columns:
                transformed_data[column] = self._apply_single_transformation(
                    data[column], trans_code
                )
        
        return transformed_data
    
    def _apply_single_transformation(self, series, trans_code):
        """
        Apply single transformation to a series
        
        Parameters:
        -----------
        series : pd.Series
            Data series
        trans_code : int
            Transformation code (1-6)
            
        Returns:
        --------
        pd.Series : Transformed series
        """
        if trans_code == 1:  # Levels
            return series
        
        elif trans_code == 2:  # First difference
            return series.diff()
        
        elif trans_code == 3:  # Second difference
            return series.diff().diff()
        
        elif trans_code == 4:  # Log levels
            return np.log(series.replace(0, np.nan))
        
        elif trans_code == 5:  # Log first difference
            log_series = np.log(series.replace(0, np.nan))
            return log_series.diff()
        
        elif trans_code == 6:  # Log second difference
            log_series = np.log(series.replace(0, np.nan))
            return log_series.diff().diff()
        
        else:
            raise ValueError(f"Unknown transformation code: {trans_code}")
    
    def standardize_data(self, data, method='zscore'):
        """
        Standardize data
        
        Parameters:
        -----------
        data : pd.DataFrame
            Input data
        method : str, default='zscore'
            Standardization method ('zscore', 'minmax', 'robust')
            
        Returns:
        --------
        tuple : (standardized_data, scaler)
        """
        if method == 'zscore':
            scaler = StandardScaler()
        elif method == 'minmax':
            from sklearn.preprocessing import MinMaxScaler
            scaler = MinMaxScaler()
        elif method == 'robust':
            from sklearn.preprocessing import RobustScaler
            scaler = RobustScaler()
        else:
            raise ValueError(f"Unknown standardization method: {method}")
        
        # Remove NaN values for fitting
        clean_data = data.dropna()
        
        # Fit scaler and transform
        standardized_values = scaler.fit_transform(clean_data)
        
        standardized_data = pd.DataFrame(
            standardized_values,
            index=clean_data.index,
            columns=clean_data.columns
        )
        
        return standardized_data, scaler
    
    def handle_missing_data(self, data, method='interpolate'):
        """
        Handle missing data
        
        Parameters:
        -----------
        data : pd.DataFrame
            Data with potential missing values
        method : str, default='interpolate'
            Method for handling missing data
            
        Returns:
        --------
        pd.DataFrame : Data with missing values handled
        """
        if method == 'interpolate':
            return data.interpolate(method='linear')
        
        elif method == 'forward_fill':
            return data.fillna(method='ffill')
        
        elif method == 'backward_fill':
            return data.fillna(method='bfill')
        
        elif method == 'drop':
            return data.dropna()
        
        elif method == 'mean':
            return data.fillna(data.mean())
        
        else:
            raise ValueError(f"Unknown missing data method: {method}")
    
    def prepare_favar_data(self, xdata_file, ydata_file, key_variables=None):
        """
        Prepare data for FAVAR analysis
        
        Parameters:
        -----------
        xdata_file : str
            Path to large dataset file
        ydata_file : str
            Path to observable factors file
        key_variables : list, optional
            List of key variables to track
            
        Returns:
        --------
        dict : Prepared data dictionary
        """
        # Load data
        print("Loading data files...")
        xdata = self.load_excel_data(xdata_file, 'Sheet1')
        ydata = self.load_excel_data(ydata_file, 'Sheet1')
        
        print(f"Loaded xdata: {xdata.shape[1]} variables, {xdata.shape[0]} observations")
        print(f"Loaded ydata: {ydata.shape[1]} variables, {ydata.shape[0]} observations")
        
        # Align time periods
        common_index = xdata.index.intersection(ydata.index)
        xdata = xdata.loc[common_index]
        ydata = ydata.loc[common_index]
        
        print(f"Common time period: {len(common_index)} observations")
        print(f"From {common_index[0]} to {common_index[-1]}")
        
        # Handle missing data
        print("Handling missing data...")
        xdata = self.handle_missing_data(xdata, method='interpolate')
        ydata = self.handle_missing_data(ydata, method='interpolate')
        
        # Define key variables if not provided
        if key_variables is None:
            key_variables = self._get_default_key_variables(xdata.columns)
        
        # Identify slow-moving variables (first 70 as in original code)
        slow_vars = xdata.iloc[:, :min(70, xdata.shape[1])]
        
        return {
            'xdata': xdata,
            'ydata': ydata,
            'slow_vars': slow_vars,
            'key_variables': key_variables,
            'common_index': common_index
        }
    
    def _get_default_key_variables(self, column_names):
        """
        Get default key variables based on common naming patterns
        
        Parameters:
        -----------
        column_names : list
            List of column names
            
        Returns:
        --------
        list : Key variable names
        """
        # Common patterns for key macroeconomic variables
        patterns = {
            'ip': 'Industrial Production',
            'cpi': 'Consumer Price Index',
            'employment': 'Employment',
            'unemployment': 'Unemployment',
            'rate': 'Interest Rate',
            'money': 'Money Supply',
            'exchange': 'Exchange Rate',
            'housing': 'Housing',
            'consumption': 'Consumption'
        }
        
        key_vars = []
        for col in column_names[:20]:  # First 20 columns
            col_lower = str(col).lower()
            for pattern in patterns:
                if pattern in col_lower:
                    key_vars.append(col)
                    break
        
        # If no matches found, use first 19 columns
        if len(key_vars) < 19:
            key_vars = list(column_names[:19])
        
        return key_vars[:19]  # Limit to 19 as in original code


class DataValidator:
    """
    Validation utilities for FAVAR data
    """
    
    def __init__(self):
        self.validation_results = {}
    
    def validate_data_quality(self, data, name="data"):
        """
        Validate data quality
        
        Parameters:
        -----------
        data : pd.DataFrame
            Data to validate
        name : str
            Name for reporting
            
        Returns:
        --------
        dict : Validation results
        """
        results = {
            'name': name,
            'shape': data.shape,
            'missing_values': data.isnull().sum().sum(),
            'missing_percentage': (data.isnull().sum().sum() / data.size) * 100,
            'infinite_values': np.isinf(data.select_dtypes(include=[np.number])).sum().sum(),
            'constant_columns': (data.nunique() <= 1).sum(),
            'time_gaps': self._check_time_gaps(data.index),
            'outliers': self._detect_outliers(data)
        }
        
        self.validation_results[name] = results
        return results
    
    def _check_time_gaps(self, index):
        """Check for gaps in time series index"""
        if isinstance(index, pd.DatetimeIndex):
            if len(index) > 1:
                expected_freq = pd.infer_freq(index)
                if expected_freq:
                    expected_range = pd.date_range(
                        start=index[0], end=index[-1], freq=expected_freq
                    )
                    return len(expected_range) - len(index)
        return 0
    
    def _detect_outliers(self, data, threshold=3):
        """Detect outliers using z-score method"""
        numeric_data = data.select_dtypes(include=[np.number])
        z_scores = np.abs((numeric_data - numeric_data.mean()) / numeric_data.std())
        return (z_scores > threshold).sum().sum()
    
    def print_validation_report(self):
        """Print validation report"""
        print("\n" + "="*60)
        print("DATA VALIDATION REPORT")
        print("="*60)
        
        for name, results in self.validation_results.items():
            print(f"\nDataset: {results['name']}")
            print(f"  Shape: {results['shape']}")
            print(f"  Missing values: {results['missing_values']} ({results['missing_percentage']:.2f}%)")
            print(f"  Infinite values: {results['infinite_values']}")
            print(f"  Constant columns: {results['constant_columns']}")
            print(f"  Time gaps: {results['time_gaps']}")
            print(f"  Potential outliers: {results['outliers']}")
            
            # Data quality assessment
            if results['missing_percentage'] < 5:
                print("  ✓ Good data quality")
            elif results['missing_percentage'] < 15:
                print("  ⚠ Moderate data quality - consider imputation")
            else:
                print("  ✗ Poor data quality - significant missing data")


def create_synthetic_data(n_obs=500, n_vars=119, start_date='1959-01'):
    """
    Create synthetic data for testing purposes
    
    Parameters:
    -----------
    n_obs : int
        Number of observations
    n_vars : int
        Number of variables
    start_date : str
        Start date for time series
        
    Returns:
    --------
    tuple : (xdata, ydata) synthetic datasets
    """
    # Create date index
    dates = pd.date_range(start=start_date, periods=n_obs, freq='M')
    
    # Create synthetic xdata
    np.random.seed(42)
    
    # Generate factors
    n_factors = 5
    factors = np.random.randn(n_obs, n_factors)
    
    # Generate factor loadings
    loadings = np.random.randn(n_vars, n_factors) * 0.5
    
    # Generate idiosyncratic errors
    errors = np.random.randn(n_obs, n_vars) * 0.3
    
    # Combine to create data
    xdata_values = factors @ loadings.T + errors
    
    # Add some persistence
    for i in range(1, n_obs):
        xdata_values[i] += 0.8 * xdata_values[i-1]
    
    # Create DataFrame
    xdata = pd.DataFrame(
        xdata_values,
        index=dates,
        columns=[f'var_{i+1}' for i in range(n_vars)]
    )
    
    # Create synthetic ydata (Federal Funds Rate)
    ffr = np.random.randn(n_obs) * 0.5
    for i in range(1, n_obs):
        ffr[i] += 0.9 * ffr[i-1]
    
    ydata = pd.DataFrame(
        ffr.reshape(-1, 1),
        index=dates,
        columns=['FFR']
    )
    
    return xdata, ydata
