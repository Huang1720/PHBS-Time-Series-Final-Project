"""
Continuous-Time FAVAR Model Implementation
Based on the paper's extension of FAVAR to continuous-time differential equations

This implementation follows the SDE framework:
dZ(t) = A*Z(t)*dt + S*dW(t)

Where Z(t) contains both latent factors F(t) and observed policy variables Y(t)
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.linalg as la
from scipy.optimize import minimize
from sklearn.decomposition import PCA
from tqdm import tqdm
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

class ContinuousFAVAR:
    """
    Continuous-Time Factor-Augmented Vector Autoregression Model
    
    This class implements the continuous-time SDE extension of FAVAR as described in the paper:
    dZ(t) = A*Z(t)*dt + S*dW(t)
    
    where Z(t) = [F(t), Y(t)]' contains both latent factors and policy variables
    """
    
    def __init__(self, K=3, p=2, delta_t=1/30):
        """
        Initialize the Continuous FAVAR model
        
        Parameters:
        -----------
        K : int
            Number of latent factors to extract
        p : int
            VAR lag order for constructing the companion form
        delta_t : float
            Time discretization step (default: 1/30 for daily data in monthly scale)
        """
        self.K = K  # Number of factors
        self.p = p  # VAR lag order
        self.delta_t = delta_t  # Time step for discretization
        self.M = None  # Number of policy variables (to be determined from data)
        
        # Model parameters (to be estimated)
        self.A = None  # Drift matrix in SDE
        self.S = None  # Structural impact matrix
        self.Sigma = None  # Covariance matrix
        
        # Data storage
        self.factors = None
        self.policy_vars = None
        self.Z = None  # Combined state vector [F, Y]
        self.Z_augmented = None  # Augmented state for VAR(p) -> VAR(1) conversion
        
    def extract_factors(self, X):
        """
        Extract latent factors using PCA
        
        Parameters:
        -----------
        X : np.array
            Observed macroeconomic indicators (T x N)
            
        Returns:
        --------
        factors : np.array
            Extracted factors (T x K)
        """
        print(f"Extracting {self.K} factors from {X.shape[1]} variables...")
        
        # Store original data information for later use
        self.X_original = X
        self.X_mean = np.mean(X, axis=0)
        self.X_std = np.std(X, axis=0)
        
        # Standardize the data
        X_std = (X - self.X_mean) / self.X_std
        
        # Apply PCA
        self.pca = PCA(n_components=self.K)
        factors = self.pca.fit_transform(X_std)
        
        # Store PCA transformation matrix (components)
        self.pca_components = self.pca.components_  # Shape: (K, N)

        # # plot factors
        # plt.figure(figsize=(10, 6))
        # for k in range(self.K):
        #     plt.plot(factors[:, k], label=f'Factor {k+1}')
        # plt.title('Extracted Factors from PCA')
        # plt.xlabel('Time')
        # plt.ylabel('Factor Value')
        # plt.legend()
        # plt.show()

        # Store explained variance ratio
        self.explained_variance_ratio = self.pca.explained_variance_ratio_
        print(f"Cumulative explained variance: {np.cumsum(self.explained_variance_ratio)[-1]:.3f}")
        
        return factors
    
    def estimate_factor_loadings(self, original_var_names=None):
        """
        Estimate factor loadings by regressing original variables on rotated factors
        
        This method estimates the relationship: X_i = beta_0 + beta_1*F_1 + ... + beta_K*F_K + epsilon_i
        where X_i is the i-th original variable and F_j are the rotated factors
        
        Parameters:
        -----------
        original_var_names : list
            Names of original variables to estimate loadings for
            
        Returns:
        --------
        factor_loadings : dict
            Dictionary containing regression coefficients for each original variable
        """
        if not hasattr(self, 'rotated_factors'):
            raise ValueError("Rotated factors not found. Please run fit() first.")
        
        if original_var_names is None:
            original_var_names = self.original_factor_names
        
        print(f"Estimating factor loadings for {len(original_var_names)} original variables...")
        
        # Get indices of the requested variables
        original_var_indices = []
        for name in original_var_names:
            if name in self.original_factor_names:
                original_var_indices.append(self.original_factor_names.index(name))
            else:
                print(f"Warning: Variable '{name}' not found in original data. Skipping.")
        
        if len(original_var_indices) == 0:
            raise ValueError("No valid original variable names found.")
        
        factor_loadings = {}
        
        # Prepare data for regression
        Y_factors = self.rotated_factors  # Shape: (T, K)
        X_original_subset = self.X_original[:, original_var_indices]  # Shape: (T, N_subset)
        
        # Standardize original variables for regression
        X_original_std = (X_original_subset - self.X_mean[original_var_indices]) / self.X_std[original_var_indices]
        
        # Add constant term to factors for regression
        Y_factors_with_const = np.column_stack([np.ones(Y_factors.shape[0]), Y_factors])  # Shape: (T, K+1)
        
        # Estimate loadings for each original variable
        for i, var_idx in enumerate(original_var_indices):
            var_name = original_var_names[i]
            
            # Regression: X_i = beta_0 + beta_1*F_1 + ... + beta_K*F_K + epsilon
            y = X_original_subset[:, i]  # Standardized original variable
            X = Y_factors_with_const   # Factors with constant
            
            # OLS estimation
            try:
                beta_hat = np.linalg.lstsq(X, y, rcond=None)[0]
                
                # Store results
                factor_loadings[var_name] = {
                    'intercept': beta_hat[0],
                    'factor_coeffs': beta_hat[1:],  # Coefficients on factors
                    'original_index': var_idx,
                    'standardization': {
                        'mean': self.X_mean[var_idx],
                        'std': self.X_std[var_idx]
                    }
                }
                
                # Calculate R-squared for diagnostic
                y_pred = X @ beta_hat
                ss_res = np.sum((y - y_pred) ** 2)
                ss_tot = np.sum((y - np.mean(y)) ** 2)
                r_squared = 1 - (ss_res / ss_tot)
                factor_loadings[var_name]['r_squared'] = r_squared
                
            except np.linalg.LinAlgError:
                print(f"Warning: Could not estimate loadings for variable '{var_name}'. Skipping.")
                continue
        
        self.factor_loadings = factor_loadings
        
        # Print summary
        print(f"Factor loadings estimated for {len(factor_loadings)} variables:")
        for var_name, loadings in factor_loadings.items():
            print(f"  {var_name}: R² = {loadings['r_squared']:.3f}")
        
        return factor_loadings
    
    def construct_state_vector(self, factors, policy_vars):
        """
        Construct the joint state vector Z(t) = [F(t), Y(t)]
        
        Parameters:
        -----------
        factors : np.array
            Latent factors (T x K)
        policy_vars : np.array
            Policy variables (T x M)
            
        Returns:
        --------
        Z : np.array
            Joint state vector (T x (K+M))
        """
        self.factors = factors
        self.policy_vars = policy_vars
        self.M = policy_vars.shape[1]
        
        # Combine factors and policy variables
        Z = np.hstack([factors, policy_vars])
        self.Z = Z
        
        print(f"State vector dimension: {Z.shape[1]} (K={self.K} factors + M={self.M} policy vars)")
        
        return Z
    
    def create_augmented_state(self, Z):
        """
        Create augmented state vector for VAR(p) to VAR(1) conversion
        
        Parameters:
        -----------
        Z : np.array
            State vector (T x (K+M))
            
        Returns:
        --------
        Z_aug : np.array
            Augmented state vector (T-p x p*(K+M))
        """
        T, dim = Z.shape
        
        # Create lagged versions
        z_aug = []
        for lag in range(self.p):
            z_aug.append(Z[lag*30:T-self.p*30+lag*30+30, :])  # Adjust for daily data
        z_aug = z_aug[::-1]  # Reverse order to match VAR(p) structure
        Z_aug = np.hstack(z_aug)
        self.Z_augmented = Z_aug
        
        print(f"Augmented state dimension: {Z_aug.shape[1]} (p={self.p} lags)")
        
        return Z_aug
    
    def estimate_drift_matrix(self, Z_aug):
        """
        Estimate the drift matrix A using Euler-Maruyama discretization
        
        The discretized SDE is:
        ΔZ(t) = A * Z(t) * Δt + S * u(t)
        
        Parameters:
        -----------
        Z_aug : np.array
            Augmented state vector
            
        Returns:
        --------
        A : np.array
            Estimated drift matrix
        """
        print("Estimating drift matrix A...")
        
        T, dim = Z_aug.shape
        
        # Compute differences: ΔZ(t) = Z(t+1) - Z(t)
        Delta_Z = Z_aug[1:] - Z_aug[:-1]
        Z_lagged = Z_aug[:-1]
        
        # OLS regression: ΔZ(t) = A * Z(t) * Δt + error
        # Rearrange: ΔZ(t) / Δt = A * Z(t) + error/Δt
        Y = Delta_Z / self.delta_t
        X = Z_lagged
        
        # Estimate A using least squares
        A = np.linalg.lstsq(X, Y, rcond=None)[0].T
        
        self.A = A
        print(f"Drift matrix shape: {A.shape}")
        
        return A
    
    def estimate_covariance_matrix(self, Z_aug):
        """
        Estimate the covariance matrix of structural innovations
        
        Parameters:
        -----------
        Z_aug : np.array
            Augmented state vector
            
        Returns:
        --------
        Sigma : np.array
            Estimated covariance matrix
        """
        print("Estimating covariance matrix...")
        
        T, dim = Z_aug.shape
        
        # Compute residuals
        Delta_Z = Z_aug[1:] - Z_aug[:-1]
        Z_lagged = Z_aug[:-1]
        
        # Predicted changes
        Delta_Z_pred = (self.A @ Z_lagged.T).T * self.delta_t
        
        # Residuals
        residuals = Delta_Z - Delta_Z_pred
        
        # Sample covariance
        Sigma_res = np.cov(residuals.T)
        
        # Scale by delta_t to get continuous-time covariance
        Sigma = Sigma_res / self.delta_t
        
        self.Sigma = Sigma
        print(f"Covariance matrix shape: {Sigma.shape}")
        
        return Sigma
    
    def cholesky_identification(self):
        """
        Identify structural shocks using Cholesky decomposition
        
        Assumes recursive structure with monetary policy variable ordered last
        
        Returns:
        --------
        S : np.array
            Structural impact matrix (lower triangular)
        """
        print("Performing Cholesky identification...")
        
        # Cholesky decomposition: Σ = S * S'
        try:
            Q_11 = self.Sigma[:(self.K + self.M), :(self.K + self.M)]
            L = la.cholesky(Q_11, lower=True)
            S = np.zeros((self.Sigma.shape[0], self.K + self.M))
            S[:(self.K + self.M), :] = L
            for i in range(self.K + self.M, self.Sigma.shape[0]):
                Qi1 = self.Sigma[i, :(self.K + self.M)]
                # Solve for S[i, :] such that S[i, :] @ L.T = Qi1
                S[i, :] = la.solve_triangular(L.T, Qi1, lower=False)
            self.S = S
            print("Cholesky decomposition successful, got S shape:", S.shape)
        except np.linalg.LinAlgError:
            raise ValueError("Cholesky decomposition failed. Check covariance matrix.")
        return S
    
    def compute_irf_with_ci(self, horizon=30, shock_size=1.0, n_sim=100, ci=[0.05, 0.95], 
                          include_original_vars=False, original_var_names=None, only_policy_vars=True):
        """
        Compute IRFs with confidence intervals using Monte Carlo simulation.
        
        Parameters:
        -----------
        horizon : int
            Maximum horizon for IRF computation
        shock_size : float
            Size of the structural shock
        n_sim : int
            Number of Monte Carlo simulations
        ci : list
            Confidence interval quantiles [lower, upper]
        include_original_vars : bool
            Whether to compute IRFs for original variables (before PCA)
        original_var_names : list
            Names of original variables (if None, uses generic names)
        
        Returns:
        --------
        irfs : dict with keys:
            - 'horizon': time grid
            - 'responses': (T, n_vars) mean IRFs for factors and policy vars
            - 'lower': (T, n_vars) lower CI
            - 'upper': (T, n_vars) upper CI
            - 'original_responses': (T, N_original) mean IRFs for original vars (if requested)
            - 'original_lower': (T, N_original) lower CI for original vars
            - 'original_upper': (T, N_original) upper CI for original vars
            - 'original_var_names': list of original variable names
        """
        print(f"Computing IRFs with confidence intervals using {n_sim} simulations...")
        # print(include_original_vars)
        if include_original_vars:
            print("Will also compute IRFs for original variables through PCA inverse transform")
        
        h_grid = np.linspace(0, horizon, int(horizon * 30))
        T = len(h_grid)
        n_vars = self.K + self.M
        A_hat = self.A
        S_hat = self.S

        # Check if we can compute original variable IRFs
        if include_original_vars and not hasattr(self, 'pca_components'):
            print("Warning: PCA components not found. Cannot compute original variable IRFs.")
            include_original_vars = False
        
        # Set up original variable names and estimate factor loadings if needed
        if include_original_vars:
            if original_var_names is None:
                original_var_names = self.original_factor_names
            
            # Estimate factor loadings if not already done
            if not hasattr(self, 'factor_loadings') or not all(name in self.factor_loadings for name in original_var_names):
                self.estimate_factor_loadings(original_var_names)
            
            N_original = len(original_var_names)
        
        self.original_var_names = original_var_names
        irfs = {}

        for shock_idx in range(n_vars):
            shock_name = f"shock_{shock_idx}"
            if only_policy_vars and shock_idx < n_vars - 1:
                continue
            if shock_idx == n_vars - 1:
                shock_name = "monetary_policy_shock"
            
            irf_samples = np.zeros((n_sim, T, n_vars))
            if include_original_vars:
                original_irf_samples = np.zeros((n_sim, T, N_original))

            for sim in tqdm(range(n_sim), desc=f"Simulating {shock_name}"):
                # 1. 模拟扰动参数（这里只用正态扰动为例，可以替换成bootstrap）
                A_sim = A_hat + np.random.normal(scale=0.01, size=A_hat.shape)
                S_sim = S_hat + np.random.normal(scale=0.01, size=S_hat.shape)
                
                for i, h in enumerate(h_grid):
                    expAh = la.expm(A_sim * h)
                    irf_h = expAh @ (S_sim[:, shock_idx] * shock_size)
                    irf_samples[sim, i, :] = irf_h[:n_vars]
                    
                    # Compute original variable IRFs if requested
                    if include_original_vars:
                        # Extract factor responses (first K components)
                        factor_responses = irf_h[:self.K]
                        
                        # Compute original variable responses using regression coefficients
                        original_responses_at_h = []
                        for var_name in original_var_names:
                            if var_name in self.factor_loadings:
                                loadings = self.factor_loadings[var_name]
                                # Response = beta_1 * dF_1 + beta_2 * dF_2 + ... + beta_K * dF_K
                                # (no intercept since we're computing changes)
                                var_response = np.dot(loadings['factor_coeffs'], factor_responses)
                                # Convert back to original scale
                                # var_response *= loadings['standardization']['std']
                                original_responses_at_h.append(var_response)
                            else:
                                original_responses_at_h.append(0.0)  # Default if not found
                        
                        original_irf_samples[sim, i, :] = original_responses_at_h

            # 计算均值、置信区间
            irf_mean = np.mean(irf_samples, axis=0)
            irf_lower = np.quantile(irf_samples, ci[0], axis=0)
            irf_upper = np.quantile(irf_samples, ci[1], axis=0)

            irf_result = {
                'horizon': h_grid,
                'responses': irf_mean,
                'lower': irf_lower,
                'upper': irf_upper
            }
            
            # Add original variable IRFs if computed
            if include_original_vars:
                original_irf_mean = np.mean(original_irf_samples, axis=0)
                original_irf_lower = np.quantile(original_irf_samples, ci[0], axis=0)
                original_irf_upper = np.quantile(original_irf_samples, ci[1], axis=0)
                
                irf_result.update({
                    'original_responses': original_irf_mean,
                    'original_lower': original_irf_lower,
                    'original_upper': original_irf_upper,
                    'original_var_names': original_var_names
                })
            
            irfs[shock_name] = irf_result

        self.irfs = irfs
        return irfs
    
    def cal_rotated_factors(self, factors, ffr):
        factors = factors.copy()
        # Rotate factor: PC_new = PC - beta_ffr * FFR
        print("Calculating rotated factors...")
        for pc in range(factors.shape[1]):
            # Calculate beta_ffr for each factor
            beta_ffr = np.corrcoef(factors[:, pc], ffr[:, 0])[0, 1]
            factors[:, pc] -= beta_ffr * ffr[:, 0]
        self.rotated_factors = factors
        print("Rotated factors calculated.")
        return factors


    def fit(self, data):
        """
        Fit the continuous FAVAR model to data
        
        Parameters:
        -----------
        data : pd.DataFrame or np.array
            Data with columns: [factor_vars..., policy_var]
            Last column should be the policy variable (interest rate)
        """
        print("=" * 50)
        print("FITTING CONTINUOUS FAVAR MODEL")
        print("=" * 50)

        if isinstance(data, pd.DataFrame):
            data_array = data.values
        else:
            data_array = data

        self.original_factor_names = data.columns[:-1].tolist() if isinstance(data, pd.DataFrame) else [f'var_{i}' for i in range(data_array.shape[1] - 1)]

        # Split data into factor variables and policy variable
        X = data_array[:, :-1]  # All columns except last
        Y = data_array[:, -1:] # Last column (policy variable)
        
        print(f"Data shape: {data_array.shape}")
        print(f"Factor variables: {X.shape[1]}")
        print(f"Policy variables: {Y.shape[1]}")
        
        # Step 1: Extract factors using PCA
        factors = self.extract_factors(X)

        rotated_factors = self.cal_rotated_factors(factors, Y)

        # Step 1.5: Estimate factor loadings using regression after rotation
        print("Estimating factor loadings using regression...")
        self.estimate_factor_loadings()

        # Step 2: Construct state vector Z(t) = [F(t), Y(t)]
        Z = self.construct_state_vector(rotated_factors, Y)
        
        # Step 3: Create augmented state for VAR(p) -> VAR(1)
        Z_aug = self.create_augmented_state(Z)
        
        # Step 4: Estimate drift matrix A
        A = self.estimate_drift_matrix(Z_aug)
        
        # Step 5: Estimate covariance matrix
        Sigma = self.estimate_covariance_matrix(Z_aug)
        
        # Step 6: Structural identification
        S = self.cholesky_identification()
        
        print("=" * 50)
        print("MODEL ESTIMATION COMPLETE")
        print("=" * 50)
        
        return self
    
    def plot_impulse_responses(self, variables=None, shock='monetary_policy_shock', 
                             figsize=(4, 4), save_path=None, show_ci=True, 
                             plot_original_vars=False, max_original_vars=12):
        """
        Plot impulse response functions
        
        Parameters:
        -----------
        variables : list
            List of variable indices to plot for factors/policy vars (default: all)
        shock : str
            Name of the shock to plot
        figsize : tuple
            Figure size for each subplot
        save_path : str
            Path to save the plot
        show_ci : bool
            Whether to show confidence intervals
        plot_original_vars : bool
            Whether to plot original variables instead of factors
        max_original_vars : int
            Maximum number of original variables to plot
        """
        if not hasattr(self, 'irfs'):
            print("IRFs has not been computed. Please run compute_irf_with_ci() first.")
            return
        
        if shock not in self.irfs:
            print(f"can't find '{shock}'. Available shocks: {list(self.irfs.keys())}")
            return

        irf_data = self.irfs[shock]
        horizon = irf_data['horizon']
        
        # Determine what to plot
        if plot_original_vars:
            if 'original_responses' not in irf_data:
                print("Original variable IRFs not available. Please run compute_irf_with_ci() with include_original_vars=True.")
                return
            
            responses = irf_data['original_responses']
            lower_ci = irf_data['original_lower']
            upper_ci = irf_data['original_upper']
            var_names = irf_data['original_var_names']
            
            # Select most responsive variables if too many
            if variables is None:
                max_responses = np.max(np.abs(responses), axis=0)
                top_vars = np.argsort(max_responses)[-max_original_vars:][::-1]
                variables = top_vars
            
            variables = variables[:max_original_vars]  # Limit number
            plot_title_prefix = "Original Variables"
            
        else:
            responses = irf_data['responses']
            lower_ci = irf_data['lower']
            upper_ci = irf_data['upper']
            
            if variables is None:
                variables = list(range(responses.shape[1]))
            
            plot_title_prefix = "Factors & Policy"
        
        # check dimensions
        if len(responses) != len(lower_ci) or len(responses) != len(upper_ci):
            raise ValueError("data length mismatch in IRF responses and confidence intervals")
        
        n_vars = len(variables)
        n_cols = 3 if plot_original_vars else 2
        n_rows = (n_vars + n_cols - 1) // n_cols

        total_figsize = (figsize[0] * n_cols, figsize[1] * n_rows)

        plt.style.use('seaborn-whitegrid')
        sns.set_palette("colorblind")
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=total_figsize, squeeze=False)
        fig.subplots_adjust(hspace=0.5, wspace=0.3)
        
        for i, var_idx in enumerate(variables):
            row, col = i // n_cols, i % n_cols
            ax = axes[row, col]
            
            ax.plot(horizon, responses[:, var_idx], linewidth=2.0, 
                    color=sns.color_palette()[0], label='mean')
            
            if show_ci:
                ax.fill_between(horizon, 
                            lower_ci[:, var_idx], 
                            upper_ci[:, var_idx], 
                            color=sns.color_palette()[0], 
                            alpha=0.2, 
                            label='90% CI')

            ax.axhline(y=0, color='black', linestyle='--', alpha=0.6, linewidth=1.2)
            
            # Set title based on what we're plotting
            if plot_original_vars:
                title = var_names[var_idx]
            else:
                if var_idx < getattr(self, 'K', n_vars):
                    title = f"factor {var_idx + 1}"
                else:
                    title = f"FFR {var_idx - self.K + 1}"
            
            ax.set_title(title, fontsize=10, fontweight='bold')
            ax.set_xlabel('time (month)', fontsize=8)
            ax.set_ylabel('response', fontsize=8)

            ax.tick_params(axis='both', which='major', labelsize=7)
            ax.grid(True, linestyle='--', alpha=0.3)
            if i == 0 and show_ci:
                ax.legend(fontsize=7, loc='best')

        for i in range(len(variables), n_rows * n_cols):
            row, col = i // n_cols, i % n_cols
            fig.delaxes(axes[row, col])

        plt.suptitle(f'{plot_title_prefix} IRF to {shock.replace("_", " ").title()}', 
                    fontsize=14, fontweight='bold')

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Image saved to: {save_path}")

        plt.show()
    
    def forecast(self, steps=30, initial_state=None):
        """
        Generate forecasts using the fitted model
        
        Parameters:
        -----------
        steps : int
            Number of steps to forecast
        initial_state : np.array
            Initial state (if None, uses last observed state)
            
        Returns:
        --------
        forecast : np.array
            Forecasted values
        """
        if initial_state is None:
            if self.Z_augmented is None:
                raise ValueError("No data fitted. Call fit() first.")
            initial_state = self.Z_augmented[-1, :]

        print(f"Generating {steps} steps ahead forecast...")
        
        # Initialize forecast array
        forecast = np.zeros((steps, self.Z_augmented.shape[1]))
        forecast[0, :] = initial_state

        for t in range(1, steps):
            # Forecast using the drift matrix A
            forecast[t, :] = (forecast[t-1, :] @ self.A) * self.delta_t + forecast[t-1, :]
        
        return forecast[:, :self.K + self.M]  # Return only factors and policy variables

def main():
    """
    Example usage of the Continuous FAVAR model
    """
    print("CONTINUOUS FAVAR MODEL IMPLEMENTATION")
    print("=====================================")
    
    # This is a demonstration with simulated data
    # In practice, you would load your CSV file here
    
    print("\nNote: This is a demonstration with simulated data.")
    print("To use with real data, replace the simulation with:")
    print("data = pd.read_csv('your_data.csv')")
    print("# Ensure last column is the interest rate (policy variable)")
    
    x_path = "processed_x.csv"  # Path to processed macro data
    y_path = "processed_y.csv"  # Path to processed policy data

    if os.path.exists(x_path) and os.path.exists(y_path):

        print(f"\nLoading data from {x_path} and {y_path}...")
        X = pd.read_csv(x_path)
        ################################################################
        X = X.drop(columns=['open', 'high', 'low', 'close', 'volume'], errors='ignore')  # Drop non-macro columns
        ################################################################
        Y = pd.read_csv(y_path)
        print(f"Data shapes: X={X.shape}, Y={Y.shape}")

        data_df = pd.merge(X, Y, on='date', how='inner')
        data_df.drop(columns=['date'], inplace=True)  # Drop date column if exists
        print(f"Merged data shape: {data_df.shape}")
        print(data_df.head())
    else:
        # Simulate some data for demonstration
        np.random.seed(42)
        T = 1000  # Number of observations
        N = 50    # Number of macro variables (before PCA)
        
        # Generate some autocorrelated data
        data_sim = np.zeros((T, N + 1))  # +1 for policy variable
        
        for i in range(N + 1):
            for t in range(1, T):
                data_sim[t, i] = 0.7 * data_sim[t-1, i] + np.random.normal(0, 1)
        
        # Add some cross-correlation
        for i in range(N):
            data_sim[:, i] += 0.3 * data_sim[:, -1] + np.random.normal(0, 0.5, T)
        
        # Create DataFrame
        columns = [f'var_{i}' for i in range(N)] + ['interest_rate']
        data_df = pd.DataFrame(data_sim, columns=columns)
        
        print(f"\nSimulated data shape: {data_df.shape}")
    
    # Initialize and fit the model
    model = ContinuousFAVAR(K=3, p=2, delta_t=1/30)
    model.fit(data_df)
    
    # # Compute impulse responses
    # print("\nComputing impulse response functions...")
    # irfs = model.compute_irf_with_ci(horizon=40, shock_size=1.0)
    
    # # Plot factor and policy variable IRFs
    # print("\nPlotting factor and policy variable impulse responses...")
    # model.plot_impulse_responses(shock='monetary_policy_shock')
    
    # Compute impulse responses including original variables
    print("\nComputing impulse responses including original variables...")
    original_var_names = ['IP', 'PUNEW', 'FMFBA', 'PMCP', 'FM2', 'EXRJAN', 'LHUR', 'FSDXP', 'GMCQ']
    irfs_with_original = model.compute_irf_with_ci(
        horizon=40, 
        shock_size=1.0, 
        include_original_vars=True,
        original_var_names=original_var_names
    )
    model.plot_impulse_responses(shock='monetary_policy_shock')
    # Plot original variable IRFs
    print("\nPlotting original variable impulse responses...")
    model.plot_impulse_responses(
        shock='monetary_policy_shock', 
        plot_original_vars=True, 
        max_original_vars=9
    )
    
    # Generate forecasts
    print("\nGenerating forecasts...")
    forecasts = model.forecast(steps=30)
    print(f"Forecast shape: {forecasts.shape}")
    
    print("\nModel fitting and analysis complete!")
    print("\nTo use with your CSV file:")
    print("1. Load your data: data = pd.read_csv('your_file.csv')")
    print("2. Ensure the last column is the interest rate")
    print("3. Run: model.fit(data)")
    print("4. Compute IRFs with original vars: model.compute_irf_with_ci(include_original_vars=True)")
    print("5. Plot original var IRFs: model.plot_impulse_responses(plot_original_vars=True)")

if __name__ == "__main__":
    main()
