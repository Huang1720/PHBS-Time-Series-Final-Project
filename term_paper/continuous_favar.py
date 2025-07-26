"""
Continuous-Time FAVAR Model Implementation
Based on the paper's extension of FAVAR to continuous-time differential equations

This implementation follows the SDE framework:
dZ(t) = A*Z(t)*dt + S*dW(t)

Where Z(t) contains both latent factors F(t) and observed policy variables Y(t)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.linalg as la
from scipy.optimize import minimize
from sklearn.decomposition import PCA
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
        
        # Standardize the data
        X_std = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
        
        # Apply PCA
        pca = PCA(n_components=self.K)
        factors = pca.fit_transform(X_std)
        
        # Store explained variance ratio
        self.explained_variance_ratio = pca.explained_variance_ratio_
        print(f"Cumulative explained variance: {np.cumsum(self.explained_variance_ratio)[-1]:.3f}")
        
        return factors
    
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
    
    def compute_impulse_responses(self, horizon=30, shock_size=1.0):
        """
        Compute impulse response functions in continuous time
        
        IRF(h) = exp(A * h) * S[:, j]
        
        Parameters:
        -----------
        horizon : float
            Maximum horizon for IRF computation
        shock_size : float
            Size of the structural shock
            
        Returns:
        --------
        irfs : dict
            Dictionary containing IRFs for each variable and shock
        """
        print(f"Computing impulse responses up to horizon {horizon}...")
        
        # Time grid for IRF computation
        h_grid = np.linspace(0, horizon, int(horizon * 30))  # Daily resolution
        
        # Number of variables in reduced form (first K+M components)
        n_vars = self.K + self.M
        
        # Extract relevant submatrices
        A_reduced = self.A
        S_reduced = self.S
        
        irfs = {}
        
        # Compute IRF for each structural shock
        for shock_idx in range(n_vars):
            shock_name = f"shock_{shock_idx}"
            if shock_idx == n_vars - 1:
                shock_name = "monetary_policy_shock"
            
            irf_matrix = np.zeros((len(h_grid), n_vars))
            
            for i, h in enumerate(h_grid):
                # IRF(h) = exp(A * h) * S[:, shock_idx]
                exp_Ah = la.expm(A_reduced * h)
                irf_h = exp_Ah @ (S_reduced[:, shock_idx] * shock_size)
                irf_matrix[i, :] = irf_h[:n_vars]  # Only take first K+M components
            
            irfs[shock_name] = {
                'horizon': h_grid,
                'responses': irf_matrix
            }
        
        self.irfs = irfs
        return irfs
    
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
        
        # Split data into factor variables and policy variable
        X = data_array[:, :-1]  # All columns except last
        Y = data_array[:, -1:] # Last column (policy variable)
        
        print(f"Data shape: {data_array.shape}")
        print(f"Factor variables: {X.shape[1]}")
        print(f"Policy variables: {Y.shape[1]}")
        
        # Step 1: Extract factors using PCA
        factors = self.extract_factors(X)
        
        # Step 2: Construct state vector Z(t) = [F(t), Y(t)]
        Z = self.construct_state_vector(factors, Y)
        
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
                             figsize=(12, 8), save_path=None):
        """
        Plot impulse response functions
        
        Parameters:
        -----------
        variables : list
            List of variable indices to plot (default: all)
        shock : str
            Name of the shock to plot
        figsize : tuple
            Figure size
        save_path : str
            Path to save the plot
        """
        if not hasattr(self, 'irfs'):
            print("IRFs not computed yet. Run compute_impulse_responses() first.")
            return
        
        if shock not in self.irfs:
            print(f"Shock '{shock}' not found. Available shocks: {list(self.irfs.keys())}")
            return
        
        irf_data = self.irfs[shock]
        horizon = irf_data['horizon']
        responses = irf_data['responses']
        
        if variables is None:
            variables = list(range(responses.shape[1]))
        
        n_vars = len(variables)
        n_cols = 3
        n_rows = (n_vars + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        if n_rows == 1:
            axes = axes.reshape(1, -1)
        
        for i, var_idx in enumerate(variables):
            row, col = i // n_cols, i % n_cols
            ax = axes[row, col]
            
            ax.plot(horizon, responses[:, var_idx], linewidth=2)
            ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)
            ax.grid(True, alpha=0.3)
            
            if var_idx < self.K:
                title = f"Factor {var_idx + 1}"
            else:
                title = f"Policy Var {var_idx - self.K + 1}"
            
            ax.set_title(title)
            ax.set_xlabel('Time Horizon')
            ax.set_ylabel('Response')
        
        # Remove empty subplots
        for i in range(len(variables), n_rows * n_cols):
            row, col = i // n_cols, i % n_cols
            axes[row, col].remove()
        
        plt.tight_layout()
        plt.suptitle(f'Impulse Responses to {shock.replace("_", " ").title()}', 
                     fontsize=14, y=1.02)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
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
    
    # Compute impulse responses
    print("\nComputing impulse response functions...")
    irfs = model.compute_impulse_responses(horizon=12, shock_size=1.0)
    
    # Plot results
    print("\nPlotting impulse responses...")
    model.plot_impulse_responses(shock='monetary_policy_shock')
    
    # Generate forecasts
    print("\nGenerating forecasts...")
    forecasts = model.forecast(steps=30)
    print(f"Forecast shape: {forecasts.shape}")
    
    print("\nModel fitting and analysis complete!")
    print("\nTo use with your CSV file:")
    print("1. Load your data: data = pd.read_csv('your_file.csv')")
    print("2. Ensure the last column is the interest rate")
    print("3. Run: model.fit(data)")

if __name__ == "__main__":
    main()
