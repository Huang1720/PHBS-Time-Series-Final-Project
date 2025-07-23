"""
Variance Decomposition for FAVAR Model
======================================

This module implements variance decomposition for FAVAR models,
including both orthogonalized and generalized variance decomposition.
"""

import numpy as np
import pandas as pd
from scipy.linalg import cholesky, solve

class VarianceDecomposition:
    """
    Implements variance decomposition for VAR models
    """
    
    def __init__(self, var_results, use_generalized=True):
        """
        Initialize variance decomposition
        
        Parameters:
        -----------
        var_results : VARResults
            Fitted VAR model results
        use_generalized : bool, default=True
            Whether to use generalized variance decomposition
        """
        self.var_results = var_results
        self.use_generalized = use_generalized
        self.n_vars = len(var_results.names)
        self.var_names = var_results.names
        
        # Get coefficient matrices and covariance matrix
        self.coeffs = self._get_coefficient_matrices()
        self.sigma = var_results.sigma_u
        
    def _get_coefficient_matrices(self):
        """Extract coefficient matrices from VAR results"""
        # Use the coefs property which gives us the correct format
        # coefs is (n_lags, n_vars, n_vars)
        try:
            # First try to use the coefs property if available
            if hasattr(self.var_results, 'coefs'):
                coeffs = [self.var_results.coefs[i] for i in range(self.var_results.k_ar)]
                print(f"Using coefs property: {len(coeffs)} lag matrices of shape {coeffs[0].shape if coeffs else 'None'}")
                return coeffs
        except:
            pass
        
        # Fallback to manual extraction
        coeffs = []
        n_vars = self.n_vars
        n_lags = self.var_results.k_ar
        
        # Use statsmodels coefficients directly
        # params is organized as (n_vars, n_vars * n_lags + const)
        params = self.var_results.params.values
        print(f"Manual extraction: params shape = {params.shape}, n_vars = {n_vars}, n_lags = {n_lags}")
        
        # Check if there's a constant term
        has_const = params.shape[1] > n_vars * n_lags
        const_offset = 1 if has_const else 0
        
        for lag in range(n_lags):
            start_col = const_offset + lag * n_vars
            end_col = const_offset + (lag + 1) * n_vars
            
            if end_col <= params.shape[1]:
                # Extract coefficient matrix for this lag
                coeff_matrix = params[:, start_col:end_col]
                coeffs.append(coeff_matrix)
                print(f"Lag {lag}: coeff_matrix shape = {coeff_matrix.shape}")
            else:
                print(f"Warning: Cannot extract lag {lag}, insufficient columns")
                break
        
        return coeffs
    
    def _get_moving_average_matrices(self, horizon):
        """
        Compute moving average representation matrices
        
        Parameters:
        -----------
        horizon : int
            Forecast horizon
            
        Returns:
        --------
        list : Moving average matrices for each period
        """
        n_vars = self.n_vars
        n_lags = len(self.coeffs)
        
        # Initialize MA matrices
        ma_matrices = [np.eye(n_vars)]  # Period 0: identity matrix
        
        # Compute MA matrices recursively
        for h in range(1, horizon + 1):
            ma_h = np.zeros((n_vars, n_vars))
            
            for j in range(min(h, n_lags)):
                if h - j - 1 < len(ma_matrices):
                    ma_h += self.coeffs[j] @ ma_matrices[h - j - 1]
            
            ma_matrices.append(ma_h)
        
        return ma_matrices
    
    def _cholesky_decomposition(self):
        """
        Perform Cholesky decomposition of covariance matrix
        
        Returns:
        --------
        np.array : Lower triangular Cholesky factor
        """
        try:
            return cholesky(self.sigma, lower=True)
        except np.linalg.LinAlgError:
            # If Cholesky fails, use regularized version
            reg_sigma = self.sigma + 1e-8 * np.eye(self.n_vars)
            return cholesky(reg_sigma, lower=True)
    
    def orthogonalized_variance_decomposition(self, horizon):
        """
        Compute orthogonalized variance decomposition (Cholesky)
        
        Parameters:
        -----------
        horizon : int
            Forecast horizon
            
        Returns:
        --------
        dict : Variance decomposition results
        """
        # Get MA representation
        ma_matrices = self._get_moving_average_matrices(horizon)
        
        # Cholesky decomposition
        P = self._cholesky_decomposition()
        
        # Compute variance contributions
        results = {}
        
        for h in range(1, horizon + 1):
            # Compute cumulative forecast error variance
            fevd_h = {}
            total_variance = np.zeros(self.n_vars)
            
            # Sum of squared MA coefficients times P
            for j in range(h):
                ma_p = ma_matrices[j] @ P
                total_variance += np.sum(ma_p**2, axis=1)
            
            # Compute contribution of each shock to each variable
            for i, var in enumerate(self.var_names):
                fevd_h[var] = {}
                
                for k, shock in enumerate(self.var_names):
                    # Contribution of shock k to variable i
                    contribution = 0
                    for j in range(h):
                        contribution += (ma_matrices[j][i, :] @ P[:, k])**2
                    
                    fevd_h[var][shock] = contribution / total_variance[i] if total_variance[i] > 0 else 0
            
            results[h] = fevd_h
        
        return results
    
    def generalized_variance_decomposition(self, horizon):
        """
        Compute generalized variance decomposition (Pesaran & Shin, 1998)
        
        Parameters:
        -----------
        horizon : int
            Forecast horizon
            
        Returns:
        --------
        dict : Variance decomposition results
        """
        try:
            # Get MA representation
            ma_matrices = self._get_moving_average_matrices(horizon)
            
            # Standard deviations of shocks
            shock_std = np.sqrt(np.diag(self.sigma))
            
            # Compute variance contributions
            results = {}
            
            for h in range(1, min(horizon + 1, len(ma_matrices))):
                fevd_h = {}
                
                # Compute total forecast error variance for each variable
                total_variance = np.zeros(self.n_vars)
                for j in range(h):
                    if j < len(ma_matrices):
                        total_variance += np.diag(ma_matrices[j] @ self.sigma @ ma_matrices[j].T)
                
                # Compute contribution of each shock to each variable
                for i, var in enumerate(self.var_names):
                    fevd_h[var] = {}
                    
                    for k, shock in enumerate(self.var_names):
                        # Contribution of shock k to variable i
                        contribution = 0
                        for j in range(h):
                            if j < len(ma_matrices) and i < ma_matrices[j].shape[0] and k < ma_matrices[j].shape[1]:
                                # Individual shock contribution
                                shock_contrib = (ma_matrices[j][i, k] * shock_std[k])**2
                                contribution += shock_contrib
                        
                        fevd_h[var][shock] = contribution / total_variance[i] if total_variance[i] > 0 else 0
                    
                    # Normalize so contributions sum to 1
                    total_contrib = sum(fevd_h[var].values())
                    if total_contrib > 0:
                        for shock in fevd_h[var]:
                            fevd_h[var][shock] /= total_contrib
                
                results[h] = fevd_h
            
            return results
            
        except Exception as e:
            print(f"Error in generalized variance decomposition: {e}")
            # Return empty results
            return {}
    
    def compute_variance_decomposition(self, horizon=40):
        """
        Compute variance decomposition using specified method
        
        Parameters:
        -----------
        horizon : int, default=40
            Maximum forecast horizon
            
        Returns:
        --------
        dict : Variance decomposition results
        """
        try:
            # Use statsmodels built-in FEVD for more robust calculation
            print("Computing variance decomposition using statsmodels FEVD...")
            fevd = self.var_results.fevd(maxn=horizon)
            
            # Convert to our format
            results = {}
            
            for h in range(1, min(horizon + 1, fevd.decomp.shape[0] + 1)):
                results[h] = {}
                
                for i, var in enumerate(self.var_names):
                    results[h][var] = {}
                    
                    for j, shock in enumerate(self.var_names):
                        if h - 1 < fevd.decomp.shape[0]:
                            contribution = fevd.decomp[h - 1, j, i]
                        else:
                            contribution = 0.0
                        results[h][var][shock] = contribution
            
            return results
            
        except Exception as e:
            print(f"Warning: statsmodels FEVD failed ({e}), using custom implementation...")
            
            # Fallback to custom implementation
            if self.use_generalized:
                print("Computing generalized variance decomposition...")
                return self.generalized_variance_decomposition(horizon)
            else:
                print("Computing orthogonalized variance decomposition...")
                return self.orthogonalized_variance_decomposition(horizon)
    
    def format_results(self, vd_results, horizons=None):
        """
        Format variance decomposition results for output
        
        Parameters:
        -----------
        vd_results : dict
            Variance decomposition results
        horizons : list, optional
            Specific horizons to include
            
        Returns:
        --------
        pd.DataFrame : Formatted results
        """
        if horizons is None:
            horizons = [1, 4, 8, 12, 20, 40]
        
        # Create formatted output
        formatted_data = []
        
        for horizon in horizons:
            if horizon in vd_results:
                for var in self.var_names:
                    for shock in self.var_names:
                        contribution = vd_results[horizon][var][shock]
                        formatted_data.append({
                            'Horizon': horizon,
                            'Variable': var,
                            'Shock': shock,
                            'Contribution': contribution * 100  # Convert to percentage
                        })
        
        return pd.DataFrame(formatted_data)


class FAVARVarianceDecomposition:
    """
    Variance decomposition specifically for FAVAR models
    """
    
    def __init__(self, favar_model, use_generalized=True):
        """
        Initialize FAVAR variance decomposition
        
        Parameters:
        -----------
        favar_model : FAVARModel
            Fitted FAVAR model
        use_generalized : bool, default=True
            Whether to use generalized variance decomposition
        """
        self.favar_model = favar_model
        self.use_generalized = use_generalized
        
        # Initialize standard variance decomposition
        self.vd = VarianceDecomposition(
            favar_model.var_results, 
            use_generalized=use_generalized
        )
    
    def compute_factor_variance_decomposition(self, horizon=40):
        """
        Compute variance decomposition at factor level
        
        Parameters:
        -----------
        horizon : int, default=40
            Maximum forecast horizon
            
        Returns:
        --------
        dict : Factor-level variance decomposition
        """
        return self.vd.compute_variance_decomposition(horizon)
    
    def transform_to_variable_level(self, factor_vd, variable_mapping=None):
        """
        Transform factor-level variance decomposition to variable level
        
        Parameters:
        -----------
        factor_vd : dict
            Factor-level variance decomposition
        variable_mapping : dict, optional
            Mapping of variables to factor loadings
            
        Returns:
        --------
        dict : Variable-level variance decomposition
        """
        if not hasattr(self.favar_model, 'pca_all'):
            raise ValueError("PCA model not available for transformation")
        
        # Get factor loadings
        factor_loadings = self.favar_model.pca_all.components_.T
        
        # Transform variance decomposition for key variables
        variable_vd = {}
        
        for horizon in factor_vd:
            variable_vd[horizon] = {}
            
            for var_name in self.favar_model.key_variables:
                if var_name in self.favar_model.xdata.columns:
                    var_idx = self.favar_model.xdata.columns.get_loc(var_name)
                    loadings = factor_loadings[var_idx, :self.favar_model.n_factors]
                    
                    variable_vd[horizon][var_name] = {}
                    
                    # For each shock, compute weighted contribution
                    for shock in self.vd.var_names:
                        contribution = 0
                        
                        # Weight by factor loadings
                        for i, factor in enumerate([f'PC_new_{j+1}' for j in range(self.favar_model.n_factors)]):
                            if factor in factor_vd[horizon]:
                                factor_contrib = factor_vd[horizon][factor].get(shock, 0)
                                contribution += (loadings[i]**2) * factor_contrib
                        
                        variable_vd[horizon][var_name][shock] = contribution
                    
                    # Normalize contributions
                    total_contrib = sum(variable_vd[horizon][var_name].values())
                    if total_contrib > 0:
                        for shock in variable_vd[horizon][var_name]:
                            variable_vd[horizon][var_name][shock] /= total_contrib
        
        return variable_vd
    
    def compute_full_variance_decomposition(self, horizon=40):
        """
        Compute complete variance decomposition for FAVAR model
        
        Parameters:
        -----------
        horizon : int, default=40
            Maximum forecast horizon
            
        Returns:
        --------
        tuple : (factor_vd, variable_vd) variance decomposition results
        """
        # Factor-level variance decomposition
        factor_vd = self.compute_factor_variance_decomposition(horizon)
        
        # Variable-level variance decomposition
        variable_vd = self.transform_to_variable_level(factor_vd)
        
        return factor_vd, variable_vd
    
    def format_variable_results(self, variable_vd, horizons=None):
        """
        Format variable-level variance decomposition results
        
        Parameters:
        -----------
        variable_vd : dict
            Variable-level variance decomposition
        horizons : list, optional
            Specific horizons to include
            
        Returns:
        --------
        pd.DataFrame : Formatted results
        """
        if horizons is None:
            horizons = [1, 4, 8, 12, 20, 40]
        
        formatted_data = []
        
        for horizon in horizons:
            if horizon in variable_vd:
                for var in variable_vd[horizon]:
                    for shock in variable_vd[horizon][var]:
                        contribution = variable_vd[horizon][var][shock]
                        formatted_data.append({
                            'Horizon': horizon,
                            'Variable': var,
                            'Shock': shock,
                            'Contribution': contribution * 100
                        })
        
        return pd.DataFrame(formatted_data)
