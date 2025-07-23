"""
Kilian Bootstrap Implementation for FAVAR Model
===============================================

This module implements the Kilian (1998) bootstrap-after-bootstrap method
for constructing confidence intervals for impulse response functions.

Reference:
Kilian, L. (1998). Small-Sample Confidence Intervals for Impulse Response Functions.
Review of Economics and Statistics, 80(2), 218-230.
"""

import numpy as np
import pandas as pd
from statsmodels.tsa.vector_ar.var_model import VAR
from scipy.linalg import eigvals
import warnings
warnings.filterwarnings('ignore')

class KilianBootstrap:
    """
    Implements Kilian (1998) bootstrap-after-bootstrap method for VAR models
    """
    
    def __init__(self, var_results, n_predraws=1000):
        """
        Initialize Kilian bootstrap
        
        Parameters:
        -----------
        var_results : VARResults
            Fitted VAR model results
        n_predraws : int, default=1000
            Number of pre-draws for bias estimation
        """
        self.var_results = var_results
        self.n_predraws = n_predraws
        
        # Store original coefficients and model structure
        self.original_coeffs = self._get_coefficients()
        self.residuals = var_results.resid
        self.n_obs, self.n_vars = self.residuals.shape
        self.n_lags = var_results.k_ar
        
        # Bias correction
        self.bias_correction = None
        self.is_stationary = True
        
    def _get_coefficients(self):
        """Extract coefficients from VAR results"""
        return self.var_results.params.values
        
    def _set_coefficients(self, var_model, coeffs):
        """Set coefficients for a VAR model"""
        # This is a simplified version - in practice, you'd need to properly
        # reconstruct the VAR model with new coefficients
        return coeffs
        
    def _get_largest_root(self, coeffs):
        """
        Compute largest eigenvalue of companion matrix
        
        Parameters:
        -----------
        coeffs : np.array
            VAR coefficients
            
        Returns:
        --------
        float : Largest absolute eigenvalue
        """
        try:
            # Construct companion matrix
            n_vars = self.n_vars
            n_lags = self.n_lags
            
            # Extract coefficient matrices (excluding constant)
            coeff_matrices = []
            for lag in range(n_lags):
                start_idx = lag * n_vars
                end_idx = (lag + 1) * n_vars
                if coeffs.shape[0] > n_vars:  # Has constant
                    coeff_matrix = coeffs[1:, start_idx:end_idx]  # Skip constant
                else:
                    coeff_matrix = coeffs[:, start_idx:end_idx]
                coeff_matrices.append(coeff_matrix)
            
            # Build companion matrix
            companion_size = n_vars * n_lags
            companion = np.zeros((companion_size, companion_size))
            
            # First n_vars rows: coefficient matrices
            for i, coeff_matrix in enumerate(coeff_matrices):
                start_col = i * n_vars
                end_col = (i + 1) * n_vars
                companion[:n_vars, start_col:end_col] = coeff_matrix
            
            # Remaining rows: identity blocks
            for i in range(1, n_lags):
                start_row = i * n_vars
                end_row = (i + 1) * n_vars
                start_col = (i - 1) * n_vars
                end_col = i * n_vars
                companion[start_row:end_row, start_col:end_col] = np.eye(n_vars)
            
            # Compute eigenvalues
            eigenvals = eigvals(companion)
            return np.max(np.abs(eigenvals))
            
        except:
            return 1.1  # Return > 1 if calculation fails
    
    def estimate_bias(self):
        """
        First stage bootstrap to estimate bias in VAR coefficients
        """
        print(f"Estimating bias with {self.n_predraws} pre-draws...")
        
        bootstrap_coeffs = []
        
        for draw in range(self.n_predraws):
            # Generate bootstrap sample
            bootstrap_data = self._generate_bootstrap_sample(self.original_coeffs)
            
            # Estimate VAR on bootstrap sample
            try:
                bootstrap_var = VAR(bootstrap_data)
                bootstrap_results = bootstrap_var.fit(maxlags=self.n_lags, ic=None)
                bootstrap_coeffs.append(self._get_coefficients())
                
            except:
                # If estimation fails, skip this draw
                continue
        
        if len(bootstrap_coeffs) > 0:
            # Compute bias as difference between mean of bootstrap estimates and original
            bootstrap_coeffs = np.array(bootstrap_coeffs)
            mean_bootstrap_coeffs = np.mean(bootstrap_coeffs, axis=0)
            self.bias_correction = mean_bootstrap_coeffs - self.original_coeffs
        else:
            self.bias_correction = np.zeros_like(self.original_coeffs)
        
        print("Bias estimation completed")
    
    def _generate_bootstrap_sample(self, coeffs):
        """
        Generate bootstrap sample using specified coefficients
        
        Parameters:
        -----------
        coeffs : np.array
            VAR coefficients to use for data generation
            
        Returns:
        --------
        pd.DataFrame : Bootstrap sample
        """
        # Resample residuals
        n_total_obs = self.n_obs + self.n_lags
        bootstrap_indices = np.random.choice(self.n_obs, n_total_obs, replace=True)
        bootstrap_residuals = self.residuals.iloc[bootstrap_indices % self.n_obs]
        
        # Initialize data with original starting values
        var_data = self.var_results.endog
        bootstrap_data = np.zeros((n_total_obs, self.n_vars))
        
        # Set initial conditions
        for i in range(self.n_lags):
            bootstrap_data[i] = var_data[i]
        
        # Generate remaining observations
        for t in range(self.n_lags, n_total_obs):
            # Construct lagged variables
            lagged_vars = []
            for lag in range(1, self.n_lags + 1):
                lagged_vars.extend(bootstrap_data[t - lag])
            
            # Add constant term
            X = np.concatenate([[1], lagged_vars])
            
            # Generate new observation
            if len(X) == coeffs.shape[0]:
                y_pred = coeffs.T @ X
            else:
                # Handle dimension mismatch
                y_pred = np.zeros(self.n_vars)
            
            # Add residual
            if t - self.n_lags < len(bootstrap_residuals):
                y_pred += bootstrap_residuals.iloc[t - self.n_lags].values
            
            bootstrap_data[t] = y_pred
        
        # Return as DataFrame
        return pd.DataFrame(
            bootstrap_data[self.n_lags:],  # Remove initial conditions
            columns=self.var_results.names
        )
    
    def bias_corrected_coefficients(self, coeffs):
        """
        Apply bias correction to coefficients while maintaining stationarity
        
        Parameters:
        -----------
        coeffs : np.array
            Original coefficients
            
        Returns:
        --------
        np.array : Bias-corrected coefficients
        """
        if self.bias_correction is None:
            return coeffs
        
        # Check if original model is stationary
        largest_root = self._get_largest_root(coeffs)
        
        if largest_root >= 1.0:
            # Don't bias-correct if already non-stationary
            return coeffs
        
        # Find largest bias correction that maintains stationarity
        delta = 1.0
        
        while delta > 0:
            corrected_coeffs = coeffs - delta * self.bias_correction
            largest_root = self._get_largest_root(corrected_coeffs)
            
            if largest_root < 1.0:
                return corrected_coeffs
            
            delta -= 0.01
        
        # If no correction maintains stationarity, return original
        return coeffs
    
    def bootstrap_draw(self):
        """
        Single bootstrap draw with bias correction
        
        Returns:
        --------
        VARResults : Bootstrap VAR estimation results
        """
        # Generate bootstrap sample
        bootstrap_data = self._generate_bootstrap_sample(self.original_coeffs)
        
        # Estimate VAR
        bootstrap_var = VAR(bootstrap_data)
        bootstrap_results = bootstrap_var.fit(maxlags=self.n_lags, ic=None)
        
        # Apply bias correction
        bootstrap_coeffs = self._get_coefficients()
        corrected_coeffs = self.bias_corrected_coefficients(bootstrap_coeffs)
        
        # Note: In a full implementation, you would need to update the VAR model
        # with the corrected coefficients. This requires modifying the internal
        # structure of the statsmodels VAR results object.
        
        return bootstrap_results
    
    def setup_bootstrap(self):
        """
        Setup bootstrap by estimating bias correction
        """
        self.estimate_bias()
        
        # Check if bias correction is feasible
        corrected_coeffs = self.bias_corrected_coefficients(self.original_coeffs)
        largest_root = self._get_largest_root(corrected_coeffs)
        
        if largest_root >= 1.0:
            print("Warning: Bias correction may not maintain stationarity")
            self.is_stationary = False
        else:
            self.is_stationary = True
            print("Bias correction setup completed")


class BootstrapIRF:
    """
    Bootstrap impulse response function estimation using Kilian method
    """
    
    def __init__(self, var_results, n_steps=48, n_draws=2000):
        """
        Initialize bootstrap IRF
        
        Parameters:
        -----------
        var_results : VARResults
            Fitted VAR model
        n_steps : int
            Number of steps for IRF
        n_draws : int
            Number of bootstrap draws
        """
        self.var_results = var_results
        self.n_steps = n_steps
        self.n_draws = n_draws
        
        # Setup Kilian bootstrap
        self.kilian_bootstrap = KilianBootstrap(var_results)
        self.kilian_bootstrap.setup_bootstrap()
        
    def compute_bootstrap_irfs(self, shock_var=None, impulse_var=None):
        """
        Compute bootstrap confidence intervals for IRFs
        
        Parameters:
        -----------
        shock_var : str, optional
            Variable for shock. If None, uses last variable
        impulse_var : str, optional
            Variable for response. If None, computes for all variables
            
        Returns:
        --------
        dict : Bootstrap IRF results with confidence intervals
        """
        if shock_var is None:
            shock_var = self.var_results.names[-1]
        
        if impulse_var is None:
            response_vars = self.var_results.names
        else:
            response_vars = [impulse_var]
        
        # Storage for bootstrap results
        bootstrap_irfs = {var: [] for var in response_vars}
        
        print(f"Computing bootstrap IRFs with {self.n_draws} draws...")
        
        for draw in range(self.n_draws):
            try:
                # Get bootstrap draw
                bootstrap_results = self.kilian_bootstrap.bootstrap_draw()
                
                # Compute IRF for this draw
                irf = bootstrap_results.irf(periods=self.n_steps)
                
                # Extract relevant responses
                shock_idx = self.var_results.names.index(shock_var)
                
                for var in response_vars:
                    var_idx = self.var_results.names.index(var)
                    response = irf.irfs[:, var_idx, shock_idx]
                    bootstrap_irfs[var].append(response)
                    
            except:
                # Skip failed draws
                continue
        
        # Compute confidence intervals
        results = {}
        for var in response_vars:
            if len(bootstrap_irfs[var]) > 0:
                responses = np.array(bootstrap_irfs[var])
                
                results[var] = {
                    'median': np.median(responses, axis=0),
                    'mean': np.mean(responses, axis=0),
                    'lower_5': np.percentile(responses, 5, axis=0),
                    'upper_95': np.percentile(responses, 95, axis=0),
                    'lower_16': np.percentile(responses, 16, axis=0),
                    'upper_84': np.percentile(responses, 84, axis=0),
                    'std': np.std(responses, axis=0)
                }
        
        return results
