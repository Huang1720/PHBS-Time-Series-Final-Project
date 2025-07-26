"""
Data Processing and Analysis Script for Continuous FAVAR Model
Handles CSV data with date column and mixed-frequency variables
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from continuous_favar import ContinuousFAVAR

class FAVARDataProcessor:
    """
    Data processor for FAVAR analysis with mixed-frequency data
    """
    
    def __init__(self):
        self.raw_data = None
        self.processed_data = None
        self.date_column = None
    
    def load_data(self, file_path, date_col=0):
        """
        Load CSV data with specified format
        
        Parameters:
        -----------
        file_path : str
            Path to the CSV file
        date_col : int or str
            Index or name of the date column
        """
        print(f"Loading data from {file_path}...")
        
        try:
            # Load the data
            data = pd.read_csv(file_path)
            print(f"Data loaded successfully. Shape: {data.shape}")
            
            # Handle date column
            if isinstance(date_col, int):
                date_column_name = data.columns[date_col]
            else:
                date_column_name = date_col
            
            # Convert date column to datetime
            data[date_column_name] = pd.to_datetime(data[date_column_name])
            
            # Set date as index
            data.set_index(date_column_name, inplace=True)
            
            self.raw_data = data
            self.date_column = date_column_name
            
            print(f"Date range: {data.index.min()} to {data.index.max()}")
            print(f"Variables: {data.shape[1]} (K+M columns as expected)")
            
            return data
            
        except Exception as e:
            print(f"Error loading data: {e}")
            return None
    
    def preprocess_data(self, interpolate_method='linear', standardize=True):
        """
        Preprocess the data for FAVAR analysis
        
        Parameters:
        -----------
        interpolate_method : str
            Method for handling missing values ('linear', 'forward', 'backward')
        standardize : bool
            Whether to standardize the non-policy variables
        """
        if self.raw_data is None:
            raise ValueError("No data loaded. Call load_data() first.")
        
        print("Preprocessing data...")
        
        data = self.raw_data.copy()
        
        # Handle missing values
        print(f"Missing values before interpolation: {data.isnull().sum().sum()}")
        
        if interpolate_method == 'linear':
            data = data.interpolate(method='linear')
        elif interpolate_method == 'forward':
            data = data.fillna(method='ffill')
        elif interpolate_method == 'backward':
            data = data.fillna(method='bfill')
        
        # Fill any remaining NaN with forward fill and backward fill
        data = data.fillna(method='ffill').fillna(method='bfill')
        
        print(f"Missing values after interpolation: {data.isnull().sum().sum()}")
        
        # Standardize all variables except the last one (interest rate)
        if standardize:
            print("Standardizing variables (except interest rate)...")
            
            # Standardize all columns except the last one
            for col in data.columns[:-1]:
                data[col] = (data[col] - data[col].mean()) / data[col].std()
            
            print("Standardization complete.")
        
        self.processed_data = data
        
        return data
    
    def summary_statistics(self):
        """
        Display summary statistics of the processed data
        """
        if self.processed_data is None:
            raise ValueError("No processed data available. Call preprocess_data() first.")
        
        print("\n" + "="*60)
        print("DATA SUMMARY STATISTICS")
        print("="*60)
        
        data = self.processed_data
        
        print(f"Total observations: {len(data)}")
        print(f"Total variables: {data.shape[1]}")
        print(f"Date range: {data.index.min().strftime('%Y-%m-%d')} to {data.index.max().strftime('%Y-%m-%d')}")
        
        print("\nDescriptive Statistics:")
        print(data.describe())
        
        # Check for stationarity (informal)
        print("\nFirst-order autocorrelations:")
        autocorrs = []
        for col in data.columns:
            autocorr = data[col].autocorr(lag=1)
            autocorrs.append(autocorr)
            print(f"{col}: {autocorr:.3f}")
        
        mean_autocorr = np.mean(autocorrs)
        print(f"\nMean autocorrelation: {mean_autocorr:.3f}")
        
        if mean_autocorr > 0.9:
            print("Warning: High autocorrelation detected. Consider differencing the data.")
        
    def plot_data_overview(self, n_vars=6, figsize=(15, 10)):
        """
        Plot overview of the data
        
        Parameters:
        -----------
        n_vars : int
            Number of variables to plot
        figsize : tuple
            Figure size
        """
        if self.processed_data is None:
            raise ValueError("No processed data available. Call preprocess_data() first.")
        
        data = self.processed_data
        
        # Select variables to plot
        if n_vars > len(data.columns):
            n_vars = len(data.columns)
        
        # Always include the interest rate (last column)
        if n_vars < len(data.columns):
            selected_vars = list(data.columns[:n_vars-1]) + [data.columns[-1]]
        else:
            selected_vars = data.columns
        
        fig, axes = plt.subplots(n_vars, 1, figsize=figsize, sharex=True)
        
        if n_vars == 1:
            axes = [axes]
        
        for i, var in enumerate(selected_vars):
            axes[i].plot(data.index, data[var], linewidth=1)
            axes[i].set_ylabel(var)
            axes[i].grid(True, alpha=0.3)
            
            # Highlight interest rate
            if var == data.columns[-1]:
                axes[i].set_title(f"{var} (Policy Variable)", fontweight='bold')
                axes[i].plot(data.index, data[var], color='red', linewidth=1.5)
        
        axes[-1].set_xlabel('Date')
        plt.tight_layout()
        plt.suptitle('Data Overview', fontsize=16, y=1.02)
        plt.show()
    
    def correlation_analysis(self, method='pearson', figsize=(12, 10)):
        """
        Analyze correlations between variables
        
        Parameters:
        -----------
        method : str
            Correlation method ('pearson', 'spearman')
        figsize : tuple
            Figure size
        """
        if self.processed_data is None:
            raise ValueError("No processed data available. Call preprocess_data() first.")
        
        data = self.processed_data
        
        # Compute correlation matrix
        corr_matrix = data.corr(method=method)
        
        # Plot heatmap
        plt.figure(figsize=figsize)
        
        # Create mask for upper triangle
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        
        # Plot
        sns.heatmap(corr_matrix, mask=mask, cmap='coolwarm', center=0,
                   square=True, linewidths=0.5, cbar_kws={"shrink": .5})
        
        plt.title(f'{method.title()} Correlation Matrix')
        plt.tight_layout()
        plt.show()
        
        # Correlation with interest rate
        interest_rate_corr = corr_matrix.iloc[-1, :-1].abs().sort_values(ascending=False)
        
        print(f"\nTop 10 correlations with interest rate:")
        for i, (var, corr) in enumerate(interest_rate_corr.head(10).items()):
            print(f"{i+1:2d}. {var}: {corr:.3f}")

def run_analysis(csv_file_path, K=3, p=2, delta_t=1/30):
    """
    Complete analysis pipeline for the continuous FAVAR model
    
    Parameters:
    -----------
    csv_file_path : str
        Path to the CSV file
    K : int
        Number of factors to extract
    p : int
        VAR lag order
    delta_t : float
        Time discretization step
    """
    print("CONTINUOUS FAVAR ANALYSIS PIPELINE")
    print("="*50)
    
    # Step 1: Load and preprocess data
    processor = FAVARDataProcessor()
    
    # Load data (assumes first column is date)
    raw_data = processor.load_data(csv_file_path, date_col=0)
    if raw_data is None:
        print("Failed to load data. Exiting.")
        return None
    
    # Preprocess
    processed_data = processor.preprocess_data(
        interpolate_method='linear',
        standardize=True
    )
    
    # Display summary
    processor.summary_statistics()
    
    # Plot data overview
    print("\nGenerating data overview plots...")
    processor.plot_data_overview(n_vars=8)
    
    # Correlation analysis
    print("\nPerforming correlation analysis...")
    processor.correlation_analysis()
    
    # Step 2: Fit FAVAR model
    print("\n" + "="*50)
    print("FITTING CONTINUOUS FAVAR MODEL")
    print("="*50)
    
    # Initialize model
    model = ContinuousFAVAR(K=K, p=p, delta_t=delta_t)
    
    # Fit model (remove date index for fitting)
    model_data = processed_data.reset_index(drop=True)
    model.fit(model_data)
    
    # Step 3: Compute and plot impulse responses
    print("\nComputing impulse response functions...")
    irfs = model.compute_impulse_responses(horizon=12, shock_size=1.0)
    
    print("\nPlotting impulse responses...")
    model.plot_impulse_responses(shock='monetary_policy_shock')
    
    # Step 4: Generate forecasts
    print("\nGenerating forecasts...")
    forecasts = model.forecast(steps=30)
    
    # Plot forecasts
    plot_forecasts(processed_data, forecasts, model)
    
    return {
        'processor': processor,
        'model': model,
        'irfs': irfs,
        'forecasts': forecasts
    }

def plot_forecasts(historical_data, forecasts, model, figsize=(15, 8)):
    """
    Plot historical data and forecasts
    """
    # Get the last date
    last_date = historical_data.index[-1]
    
    # Create future dates (assuming daily frequency)
    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), 
                                periods=len(forecasts), freq='D')
    
    # Plot factors and policy variable
    n_plots = model.K + model.M
    n_cols = 2
    n_rows = (n_plots + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    
    # Historical data for comparison
    historical_tail = historical_data.tail(60)  # Last 60 days
    
    for i in range(n_plots):
        row, col = i // n_cols, i % n_cols
        ax = axes[row, col]
        
        if i < model.K:
            # This is a factor (we need to reconstruct from state)
            title = f"Factor {i+1} (Latent)"
            # Plot historical factors
            ax.plot(historical_tail.index, model.factors[-60:, i], 
                   label='Historical', color='blue', alpha=0.7)
            # Plot forecast
            ax.plot(future_dates, forecasts[:, i], 
                   label='Forecast', color='red', linestyle='--')
        else:
            # This is the policy variable
            policy_idx = i - model.K
            var_name = historical_data.columns[-(model.M - policy_idx)]
            title = f"{var_name} (Policy)"
            
            # Plot historical policy variable
            ax.plot(historical_tail.index, historical_tail.iloc[:, -(model.M - policy_idx)], 
                   label='Historical', color='blue', alpha=0.7)
            # Plot forecast
            ax.plot(future_dates, forecasts[:, i], 
                   label='Forecast', color='red', linestyle='--')
        
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # Remove empty subplots
    for i in range(n_plots, n_rows * n_cols):
        row, col = i // n_cols, i % n_cols
        axes[row, col].remove()
    
    plt.tight_layout()
    plt.suptitle('Historical Data and Forecasts', fontsize=14, y=1.02)
    plt.show()

# Example usage
if __name__ == "__main__":
    print("FAVAR Data Processing and Analysis")
    print("==================================")
    print()
    print("This script is designed to work with your CSV file containing:")
    print("- First column: Date")
    print("- Next K+M-1 columns: Principal component processed variables")
    print("- Last column: Interest rate (policy variable)")
    print()
    print("To run the analysis, use:")
    print("results = run_analysis('your_data.csv', K=3, p=2)")
    print()
    print("The analysis will:")
    print("1. Load and preprocess the data")
    print("2. Display summary statistics and plots")
    print("3. Fit the continuous FAVAR model")
    print("4. Compute impulse response functions")
    print("5. Generate forecasts")
