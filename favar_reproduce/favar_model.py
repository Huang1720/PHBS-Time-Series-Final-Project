"""
FAVAR Model Python Implementation
=================================

This module implements the Factor-Augmented Vector Autoregressive (FAVAR) model
as described in Bernanke et al. (2005).
"""

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.vector_ar.var_model import VAR
from statsmodels.tsa.vector_ar.irf import IRAnalysis
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

class FAVARModel:
    """
    Factor-Augmented Vector Autoregressive (FAVAR) Model
    
    This class implements the two-step estimation procedure for FAVAR models:
    1. Extract factors from a large dataset using PCA
    2. Estimate VAR with extracted factors and observable factors
    """
    
    def __init__(self, n_factors=3, n_lags=7, n_steps=48, n_draws=2000):
        """
        Initialize FAVAR model parameters
        
        Parameters:
        -----------
        n_factors : int, default=3
            Number of latent factors to extract
        n_lags : int, default=7
            Number of lags in VAR model
        n_steps : int, default=48
            Number of steps for impulse response function
        n_draws : int, default=2000
            Number of bootstrap draws for confidence intervals
        """
        self.n_factors = n_factors
        self.n_lags = n_lags
        self.n_steps = n_steps
        self.n_draws = n_draws
        
        # Data storage
        self.xdata = None  # Large dataset for factor extraction
        self.ydata = None  # Observable factors (e.g., Federal Funds Rate)
        self.slow_vars = None  # Slow-moving variables
        
        # Model components
        self.scaler = StandardScaler()
        self.pca_slow = None
        self.pca_all = None
        self.factors = None
        self.rotated_factors = None
        self.var_model = None
        self.var_results = None
        
        # Results storage
        self.irf_results = {}
        self.variance_decomposition = {}
        self.bootstrap_results = {}
        
    def load_data(self, xdata_file, ydata_file):
        """
        Load data from Excel files
        
        Parameters:
        -----------
        xdata_file : str
            Path to Excel file containing large dataset (119 variables)
        ydata_file : str
            Path to Excel file containing observable factors (e.g., FFR)
        """
        print("Loading data...")
        
        # Load large dataset
        self.xdata = pd.read_excel(xdata_file, index_col=0)
        
        # Load observable factors
        self.ydata = pd.read_excel(ydata_file, index_col=0)
        
        # Ensure same time period
        common_index = self.xdata.index.intersection(self.ydata.index)
        self.xdata = self.xdata.loc[common_index]
        self.ydata = self.ydata.loc[common_index]
        
        print(f"Data loaded: {len(self.xdata.columns)} variables, {len(self.xdata)} observations")
        print(f"Time period: {self.xdata.index[0]} to {self.xdata.index[-1]}")
        
    def preprocess_data(self, key_variables=None):
        """
        Preprocess data including standardization and transformation
        
        Parameters:
        -----------
        key_variables : list, optional
            List of key variable names to track separately
        """
        print("Preprocessing data...")
        
        # Define key variables if not provided (matching RATS code)
        if key_variables is None:
            key_dict = {
                'ip': (5, "IP"),
                'punew': (5, "CPI"), 
                'fygm3': (1, "3m TREASURY BILLS"),
                'fygt5': (1, "5y TREASURY BONDS"),
                'fmfba': (5, "MONETARY BASE"),
                'fm2': (5, "M2"),
                'exrjan': (5, "EXCHANGE RATE YEN"),
                'pmcp': (1, "COMMODITY PRICE INDEX"),
                'ipxmca': (1, "CAPACITY UTIL RATE"),
                'gmcq': (5, "PERSONAL CONSUMPTION"),
                'gmcdq': (5, "DURABLE CONS"),
                'gmcnq': (5, "NONDURABLE CONS"),
                'lhur': (1, "UNEMPLOYMENT"),
                'lhem': (5, "EMPLOYMENT"),
                'lehm': (5, "AVG HOURLY EARNINGS"),
                'hsfr': (4, "HOUSING STARTS"),
                'mocmq': (5, "NEW ORDERS"),
                'fsdxp': (1, "DIVIDENDS"),
                'hhsntn': (1, "CONSUMER EXPECTATIONS")
            }
            self.key_variables = [i.upper() for i in list(key_dict.keys())]
        else:
            self.key_variables = key_variables
        self.key_variables_idx = [list(self.xdata.columns).index(var) for var in self.key_variables]
        
        # Identify slow-moving variables (first 70 variables as in RATS code)
        self.slow_vars = self.xdata.iloc[:, :70]
        
        # Standardize all variables (Z-score normalization)
        self.xdata_standardized = pd.DataFrame(
            self.scaler.fit_transform(self.xdata),
            index=self.xdata.index,
            columns=self.xdata.columns
        )
        print(self.xdata_standardized)
        # Standardize observable factors
        self.ydata_standardized = pd.DataFrame(
            StandardScaler().fit_transform(self.ydata),
            index=self.ydata.index,
            columns=self.ydata.columns
        )
        
        print("Data preprocessing completed")
        
    def extract_factors(self):
        """
        Extract factors using Principal Component Analysis (PCA)
        Following the two-step procedure in the RATS code
        """
        print(f"Extracting {self.n_factors} factors...")
        
        # Step 1: PCA on slow-moving variables only (first 70 variables)
        self.pca_slow = PCA(n_components=self.n_factors)
        slow_vars_std = StandardScaler().fit_transform(self.slow_vars)
        factors_slow = self.pca_slow.fit_transform(slow_vars_std)
        
        # Step 2: PCA on all variables
        self.pca_all = PCA(n_components=self.n_factors)
        factors_all = self.pca_all.fit_transform(self.xdata_standardized)
        
        # Store factors as DataFrame
        self.factors = pd.DataFrame(
            factors_all,
            index=self.xdata.index,
            columns=[f'PC_{i+1}' for i in range(self.n_factors)]
        )
        
        self.factors_slow = pd.DataFrame(
            factors_slow,
            index=self.xdata.index,
            columns=[f'PC_slow_{i+1}' for i in range(self.n_factors)]
        )
        
        print(f"Factors extracted. Explained variance ratio: {self.pca_all.explained_variance_ratio_}")
        
    def rotate_factors(self):
        """
        Rotate factors to be orthogonal to observable factors (Federal Funds Rate)
        This implements the rotation procedure from the RATS code
        """
        print("Rotating factors...")
        
        # Get Federal Funds Rate (first column of ydata)
        ffr = self.ydata_standardized.iloc[:, 0]
        
        # Rotate each factor
        self.rotated_factors = pd.DataFrame(
            index=self.factors.index,
            columns=[f'PC_new_{i+1}' for i in range(self.n_factors)]
        )
        self.rotate_beta = []
        for i in range(self.n_factors):
            # Regression of factor on constant, FFR, and slow factors
            X = pd.concat([
                pd.Series(1, index=self.factors.index, name='const'),
                ffr,
                self.factors_slow
            ], axis=1)
            
            y = self.factors.iloc[:, i]
            
            # OLS regression
            from sklearn.linear_model import LinearRegression
            reg = LinearRegression()
            reg.fit(X, y)
            
            # Rotate factor: PC_new = PC - beta_ffr * FFR
            beta_ffr = reg.coef_[1]  # Coefficient on FFR
            self.rotated_factors.iloc[:, i] = y - beta_ffr * ffr
            self.rotate_beta.append(beta_ffr)
        print("Factor rotation completed")
        
    def estimate_var(self):
        """
        Estimate VAR model with rotated factors and observable factors
        """
        print(f"Estimating VAR model with {self.n_lags} lags...")
        
        # Combine rotated factors with observable factors
        var_data = pd.concat([
            self.rotated_factors,
            self.ydata_standardized
        ], axis=1)
        
        # Remove NaN values
        var_data = var_data.dropna()

        # Estimate VAR
        self.var_model = VAR(var_data)
        self.var_results = self.var_model.fit(maxlags=self.n_lags, ic='aic')
        
        print(f"VAR model estimated. Selected lags: {self.var_results.k_ar}")
        print(f"AIC: {self.var_results.aic}")
        
    def compute_impulse_responses(self, shock_var=None):
        """
        Compute impulse response functions using Kilian bootstrap method
        
        Parameters:
        -----------
        shock_var : str, optional
            Name of shock variable. If None, uses last variable (typically FFR)
        """
        print("Computing impulse response functions...")
        
        if shock_var is None:
            shock_var = self.var_results.names[-1]  # Last variable (FFR)
        print(f"Shock variable: {shock_var}")

        # Compute baseline IRF
        irf = self.var_results.irf(periods=self.n_steps)
        # Store baseline results
        var_names = self.var_results.names
        shock_idx = var_names.index(shock_var)
        self.irf_baseline = {}
        for i, var in enumerate(var_names):
            self.irf_baseline[var] = irf.irfs[:, i, shock_idx]
        
        # Bootstrap for confidence intervals
        self._bootstrap_irf(shock_var)
        
        print("Impulse response functions computed")
        
    def _bootstrap_irf(self, shock_var):
        """
        Perform Kilian bootstrap for IRF confidence intervals
        """
        print(f"Performing bootstrap with {self.n_draws} draws...")
        
        var_names = self.var_results.names
        shock_idx = var_names.index(shock_var)
        n_vars = len(var_names)
        
        # Storage for bootstrap results
        bootstrap_irfs = np.zeros((self.n_draws, self.n_steps+1, n_vars))
        
        # Get residuals
        residuals = self.var_results.resid
        
        for draw in tqdm(range(self.n_draws), desc="Bootstrap draws"):
            # Resample residuals
            n_obs = len(residuals)
            bootstrap_indices = np.random.choice(n_obs, n_obs, replace=True)
            bootstrap_resids = residuals.iloc[bootstrap_indices]
            
            # Generate bootstrap data
            bootstrap_data = self._generate_bootstrap_data(bootstrap_resids)
            # Estimate VAR on bootstrap data
            try:
                bootstrap_var = VAR(bootstrap_data)
                bootstrap_results = bootstrap_var.fit(maxlags=self.var_results.k_ar, ic=None)
                bootstrap_irf = bootstrap_results.irf(periods=self.n_steps)
                # Store IRF for this draw
                bootstrap_irfs[draw] = bootstrap_irf.irfs[:, :, shock_idx]
                
            except Exception as e:
                # Print error message for debugging
                print(f"Bootstrap draw {draw} failed: {e}")
                continue
        
        # Compute confidence intervals
        self.irf_results = {}
        for i, var in enumerate(var_names):
            responses = bootstrap_irfs[:, :, i]
            
            self.irf_results[var] = {
                'median': np.median(responses, axis=0),
                'lower_5': np.percentile(responses, 5, axis=0),
                'upper_95': np.percentile(responses, 95, axis=0),
                'baseline': self.irf_baseline[var]
            }
            
        print("Bootstrap completed")
        
    def _generate_bootstrap_data(self, bootstrap_resids):
        """
        Generate bootstrap data for VAR estimation
        """
        # This is a simplified version - in practice, you'd want to implement
        # the full Kilian bias-correction procedure
        
        # Get VAR data
        var_data = pd.concat([
            self.rotated_factors,
            self.ydata_standardized
        ], axis=1).dropna()
        
        # Generate new data using VAR coefficients and bootstrap residuals
        n_obs = len(var_data)
        n_vars = var_data.shape[1]
        
        # Initialize with actual initial values
        bootstrap_data = var_data.copy()
        
        # Generate new observations
        for t in range(self.var_results.k_ar, n_obs):
            # Get lagged values
            lagged_values = []
            for lag in range(1, self.var_results.k_ar + 1):
                lagged_values.extend(bootstrap_data.iloc[t-lag].values)
            # Add constant
            X = np.concatenate([[1], lagged_values])
            # Predict using VAR coefficients
            prediction = self.var_results.params.T @ X
            
            # Add bootstrap residual
            if t < len(bootstrap_resids):
                prediction += bootstrap_resids.iloc[t].values
            
            # Update bootstrap data
            bootstrap_data.iloc[t] = prediction
        
        return bootstrap_data
        
    def define_key_variables_with_transforms(self):
        """
        Define key variables and their transformation codes as in the RATS code.
        
        Transformation codes (from RATS code):
        1 = none (levels)
        2 = first difference  
        4 = log levels
        5 = log difference
        
        Returns:
        --------
        dict : Dictionary mapping variable names to (transform_code, label)
        """
        return {
            'ip': (5, "IP"),
            'punew': (5, "CPI"), 
            'fygm3': (1, "3m TREASURY BILLS"),
            'fygt5': (1, "5y TREASURY BONDS"),
            'fmfba': (5, "MONETARY BASE"),
            'fm2': (5, "M2"),
            'exrjan': (5, "EXCHANGE RATE YEN"),
            'pmcp': (1, "COMMODITY PRICE INDEX"),
            'ipxmca': (1, "CAPACITY UTIL RATE"),
            'gmcq': (5, "PERSONAL CONSUMPTION"),
            'gmcdq': (5, "DURABLE CONS"),
            'gmcnq': (5, "NONDURABLE CONS"),
            'lhur': (1, "UNEMPLOYMENT"),
            'lhem': (5, "EMPLOYMENT"),
            'lehm': (5, "AVG HOURLY EARNINGS"),
            'hsfr': (4, "HOUSING STARTS"),
            'mocmq': (5, "NEW ORDERS"),
            'fsdxp': (1, "DIVIDENDS"),
            'hhsntn': (1, "CONSUMER EXPECTATIONS")
        }

    def apply_rats_transform_logic(self, irf_median, irf_lower, irf_upper, transform_code):
        """
        Apply RATS transformation logic to IRF results.
        
        From RATS code line 218-226:
        if keylooks(i_,2)==2.or.keylooks(i_,2)==5 {
           acc s1 1 nsteps ss1
           acc s2 1 nsteps ss2  
           acc s3 1 nsteps ss3
        }
        
        This means for transform codes 2 or 5 (differences), we need to accumulate (integrate)
        the IRF to get the level response.
        
        Parameters:
        -----------
        irf_median, irf_lower, irf_upper : numpy arrays
            IRF values
        transform_code : int
            Variable transformation code
            
        Returns:
        --------
        tuple : (transformed_median, transformed_lower, transformed_upper)
        """
        if transform_code in [2, 5]:  # First difference or log difference
            # Accumulate (cumulative sum) to get level response
            return (np.cumsum(irf_median), 
                   np.cumsum(irf_lower), 
                   np.cumsum(irf_upper))
        else:
            # No transformation needed for levels (codes 1, 4)
            return irf_median, irf_lower, irf_upper

    def transform_to_variables(self, variable_mapping=None):
        """
        Transform factor-level IRFs back to original variable IRFs
        Now includes proper handling of RATS transformation codes.
        
        Correct methodology:
        1. Regress each standardized original variable on rotated factors + observable factors
        2. Use regression coefficients to transform rotated factor IRFs to variable IRFs
        3. Apply RATS transformation logic based on variable codes
        
        Parameters:
        -----------
        variable_mapping : dict, optional
            Mapping of variable names to column indices in original data
        """
        print("Transforming IRFs to original variables...")
        
        # Get the transformation codes for key variables
        key_vars_with_transforms = self.define_key_variables_with_transforms()
        
        # Step 1: Compute regression coefficients for each standardized variable
        print("Computing factor loadings for rotated factors...")
        
        # Combine rotated factors with observable factors (same as VAR data)
        var_data = pd.concat([
            self.rotated_factors,
            self.ydata_standardized
        ], axis=1).dropna()
        
        # Storage for regression coefficients
        self.variable_loadings = {}
        
        # For each variable in the standardized dataset
        for i, col_name in enumerate(self.xdata_standardized.columns):
            if i in self.key_variables_idx:  # Only process key variables
                var_name = col_name
                
                # Get the standardized variable data (aligned with VAR data)
                y_var = self.xdata_standardized[col_name].loc[var_data.index]
                
                # Regression: X_i = alpha + beta_1 * F1_rot + ... + beta_K * FK_rot + gamma * FFR + epsilon
                from sklearn.linear_model import LinearRegression
                reg = LinearRegression()
                reg.fit(var_data, y_var)
                
                # Store coefficients
                self.variable_loadings[var_name] = {
                    'coefficients': reg.coef_,
                    'intercept': reg.intercept_,
                    'r_squared': reg.score(var_data, y_var)
                }
        
        print(f"Computed loadings for {len(self.variable_loadings)} variables")
        
        # Step 2: Transform IRFs using regression coefficients
        self.variable_irfs = {}
        
        # Get factor IRFs
        var_names = self.var_results.names
        
        for var_name, loading_info in self.variable_loadings.items():
            coefficients = loading_info['coefficients']
            
            # Transform IRFs: Variable_IRF = sum(coeff_j * Factor_j_IRF)
            var_irf_median = np.zeros(self.n_steps + 1)
            var_irf_lower = np.zeros(self.n_steps + 1)
            var_irf_upper = np.zeros(self.n_steps + 1)
            
            for j, factor_name in enumerate(var_names):
                if factor_name in self.irf_results:
                    coeff = coefficients[j]
                    
                    var_irf_median += coeff * self.irf_results[factor_name]['median']
                    var_irf_lower += coeff * self.irf_results[factor_name]['lower_5']
                    var_irf_upper += coeff * self.irf_results[factor_name]['upper_95']
            # Add intercept (constant term)
            var_irf_median += loading_info['intercept']
            var_irf_lower += loading_info['intercept']
            var_irf_upper += loading_info['intercept']
            
            # Step 3: Apply RATS transformation logic based on variable code
            if var_name in key_vars_with_transforms:
                transform_code = key_vars_with_transforms[var_name.lower()][0]
                var_irf_median, var_irf_lower, var_irf_upper = self.apply_rats_transform_logic(
                    var_irf_median, var_irf_lower, var_irf_upper, transform_code
                )
                
                print(f"Applied RATS transform logic to {var_name} (code {transform_code})")
            
            # Store results
            self.variable_irfs[var_name] = {
                'median': var_irf_median,
                'lower_5': var_irf_lower,
                'upper_95': var_irf_upper,
                'r_squared': loading_info['r_squared'],
                'transform_code': key_vars_with_transforms.get(var_name.lower(), (1, ''))[0] if var_name.lower() in key_vars_with_transforms else 1
            }
        
        # Print diagnostic information
        print(f"IRFs computed for {len(self.variable_irfs)} variables")
        print("Variable transformation codes applied:")
        for var_name, var_info in self.variable_irfs.items():
            code = var_info['transform_code']
            code_desc = {1: 'levels', 2: 'diff', 4: 'log', 5: 'log_diff'}.get(code, 'unknown')
            print(f"  {var_name}: code {code} ({code_desc}), R² = {var_info['r_squared']:.3f}")

    def plot_impulse_responses(self, variables=None, save_path=None):
        """
        Plot impulse response functions with confidence bands
        
        Parameters:
        -----------
        variables : list, optional
            List of variables to plot. If None, plots all available
        save_path : str, optional
            Path to save the plot
        """
        if variables is None:
            variables = list(self.variable_irfs.keys())[:9]  # Plot first 9
        
        n_vars = len(variables)
        n_cols = 3
        n_rows = (n_vars + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4*n_rows))
        if n_rows == 1:
            axes = axes.reshape(1, -1)
        
        for i, var in enumerate(variables):
            row = i // n_cols
            col = i % n_cols
            ax = axes[row, col]
            
            if var in self.variable_irfs:
                data = self.variable_irfs[var]
                periods = range(self.n_steps)
                
                # Plot median response
                ax.plot(periods, data['median'], 'b-', linewidth=2, label='Median')
                
                # Plot confidence bands
                ax.fill_between(periods, data['lower_5'], data['upper_95'], 
                               alpha=0.3, color='blue', label='90% CI')
                
                # Add zero line
                ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)
                
                ax.set_title(f'{var}')
                ax.set_xlabel('Periods')
                ax.set_ylabel('Response')
                ax.grid(True, alpha=0.3)
                
        # Remove empty subplots
        for i in range(n_vars, n_rows * n_cols):
            row = i // n_cols
            col = i % n_cols
            axes[row, col].remove()
        
        plt.tight_layout()
        plt.suptitle('Impulse Response Functions to Monetary Policy Shock', 
                    fontsize=16, y=1.02)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
        
    def compute_variance_decomposition(self, horizons=[1, 4, 8, 12, 20, 40]):
        """
        Compute variance decomposition
        
        Parameters:
        -----------
        horizons : list
            Forecast horizons for variance decomposition
        """
        print("Computing variance decomposition...")
        
        # Use statsmodels FEVD
        fevd = self.var_results.fevd(maxn=max(horizons))
        
        self.variance_decomposition = {}
        
        # Extract decomposition for specified horizons
        for var in self.var_results.names:
            self.variance_decomposition[var] = {}
            for horizon in horizons:
                decomp = fevd.decomp[horizon-1, :, self.var_results.names.index(var)]
                self.variance_decomposition[var][horizon] = {
                    shock: decomp[i] for i, shock in enumerate(self.var_results.names)
                }
        
        print("Variance decomposition completed")
        
    def save_results(self, output_file='IRF_FAVAR_python.xlsx'):
        """
        Save results to Excel file
        
        Parameters:
        -----------
        output_file : str
            Output Excel file name
        """
        print(f"Saving results to {output_file}...")
        
        with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
            # Save factor-level IRFs
            factor_irf_data = {}
            for var in self.irf_results:
                factor_irf_data[f'{var}_median'] = self.irf_results[var]['median']
                factor_irf_data[f'{var}_lower'] = self.irf_results[var]['lower_5']
                factor_irf_data[f'{var}_upper'] = self.irf_results[var]['upper_95']
            
            factor_df = pd.DataFrame(factor_irf_data)
            factor_df.to_excel(writer, sheet_name='Factor_IRFs')
            
            # Save variable-level IRFs
            if hasattr(self, 'variable_irfs'):
                var_irf_data = {}
                for var in self.variable_irfs:
                    var_irf_data[f'{var}_median'] = self.variable_irfs[var]['median']
                    var_irf_data[f'{var}_lower'] = self.variable_irfs[var]['lower_5']
                    var_irf_data[f'{var}_upper'] = self.variable_irfs[var]['upper_95']
                
                var_df = pd.DataFrame(var_irf_data)
                var_df.to_excel(writer, sheet_name='Variable_IRFs')
            
            # Save variance decomposition
            if hasattr(self, 'variance_decomposition'):
                vd_data = []
                for var in self.variance_decomposition:
                    for horizon in self.variance_decomposition[var]:
                        for shock in self.variance_decomposition[var][horizon]:
                            vd_data.append({
                                'Variable': var,
                                'Horizon': horizon,
                                'Shock': shock,
                                'Contribution': self.variance_decomposition[var][horizon][shock]
                            })
                
                vd_df = pd.DataFrame(vd_data)
                vd_df.to_excel(writer, sheet_name='Variance_Decomposition', index=False)
        
        print("Results saved successfully")
        
    def run_full_analysis(self, xdata_file, ydata_file, output_file='IRF_FAVAR_python.xlsx'):
        """
        Run complete FAVAR analysis
        
        Parameters:
        -----------
        xdata_file : str
            Path to xdata Excel file
        ydata_file : str
            Path to ydata Excel file
        output_file : str
            Output file name
        """
        print("=" * 60)
        print("FAVAR Model Analysis - Python Implementation")
        print("Based on Bernanke et al. (2005)")
        print("=" * 60)
        
        # Load and preprocess data
        self.load_data(xdata_file, ydata_file)
        self.preprocess_data()
        
        # Extract and rotate factors
        self.extract_factors()
        self.rotate_factors()
        
        # Estimate VAR and compute IRFs
        self.estimate_var()
        self.compute_impulse_responses()
        
        # Transform to variable level
        self.transform_to_variables()
        
        # Compute variance decomposition
        self.compute_variance_decomposition()
        
        # Save results
        self.save_results(output_file)
        
        # Plot results
        self.plot_impulse_responses()
        
        print("=" * 60)
        print("Analysis completed successfully!")
        print("=" * 60)
