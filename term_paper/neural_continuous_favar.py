"""
Neural SDE-based Continuous-Time FAVAR Model Implementation
Based on the paper's extension using neural networks for nonlinear dynamics

This implementation follows the Neural SDE framework:
dZ(t) = f_theta(Z(t), t)*dt + g_phi(Z(t), t)*dW(t)

Where f_theta and g_phi are neural networks parameterizing drift and diffusion functions
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import MultivariateNormal
from sklearn.decomposition import PCA

class DriftNetwork(nn.Module):
    """
    Neural network for the drift function f_theta(Z(t), t)
    """
    def __init__(self, input_dim, hidden_dims=[64, 32], dropout_rate=0.1):
        super(DriftNetwork, self).__init__()
        
        layers = []
        # prev_dim = input_dim + 1  # +1 for time dimension
        prev_dim = input_dim  # simplified for stability
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ])
            prev_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, input_dim))
        
        self.network = nn.Sequential(*layers)
        
        # Initialize output layer with small weights for stability
        with torch.no_grad():
            self.network[-1].weight.data *= 0.1
            self.network[-1].bias.data.zero_()
        
    def forward(self, z, t):
        """
        Forward pass
        
        Parameters:
        -----------
        z : torch.Tensor
            State vector (batch_size, input_dim)
        t : torch.Tensor
            Time (batch_size, 1)
            
        Returns:
        --------
        drift : torch.Tensor
            Drift values (batch_size, input_dim)
        """
        # Concatenate state and time
        # zt = torch.cat([z, t], dim=1)
        drift = self.network(z)
        
        # Scale drift to prevent extreme values
        drift = torch.tanh(drift) * 1.0  # Constrain to [-3, 3]
        
        return drift

class DiffusionNetwork(nn.Module):
    """
    Neural network for the diffusion function g_phi(Z(t), t)
    Output shape: (batch_size, p*(K+M), K+M)
    
    Following paper2.txt: The first K+M rows form a lower triangular structure
    for structural shock identification
    """
    def __init__(self, state_dim, factor_policy_dim, hidden_dims=[64, 32], dropout_rate=0.1):
        super(DiffusionNetwork, self).__init__()
        
        self.state_dim = state_dim  # p*(K+M) - augmented state dimension
        self.factor_policy_dim = factor_policy_dim  # K+M - original factor + policy dimension
        layers = []
        prev_dim = state_dim # simplified for stability
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ])
            prev_dim = hidden_dim
        
        # Output dimension: state_dim * factor_policy_dim for the full matrix
        self.n_params = state_dim * factor_policy_dim
        layers.append(nn.Linear(prev_dim, self.n_params))
        
        self.network = nn.Sequential(*layers)
        
        # Initialize output layer with small weights for stability
        with torch.no_grad():
            self.network[-1].weight.data *= 0.1
            self.network[-1].bias.data *= 0.1
        
    def forward(self, z, t):
        """
        Forward pass
        
        Parameters:
        -----------
        z : torch.Tensor
            State vector (batch_size, input_dim)
        t : torch.Tensor
            Time (batch_size, 1)
            
        Returns:
        --------
        diffusion_matrix : torch.Tensor
            Diffusion matrix (batch_size, state_dim, factor_policy_dim)
            with lower triangular structure in first K+M rows
        """
        batch_size = z.shape[0]
        
        # Get matrix parameters
        matrix_params = self.network(z)
        
        # Scale parameters to prevent extreme values
        matrix_params = torch.tanh(matrix_params) * 1.0  # Constrain to [-1, 1]
        
        # Reshape to matrix form: (batch_size, state_dim, factor_policy_dim)
        diffusion_matrix = matrix_params.view(batch_size, self.state_dim, self.factor_policy_dim)
        
        # Apply lower triangular constraint to first K+M rows
        # This ensures structural identification as per paper2.txt
        mask = torch.triu(torch.ones(self.factor_policy_dim, self.factor_policy_dim), diagonal=1)
        mask = mask.to(diffusion_matrix.device)
        
        # Apply mask to first K+M rows (avoid in-place operation)
        masked_rows = diffusion_matrix[:, :self.factor_policy_dim] * (1 - mask.unsqueeze(0))
        diffusion_matrix = torch.cat([
            masked_rows,
            diffusion_matrix[:, self.factor_policy_dim:]
        ], dim=1)
        
        # Ensure diagonal elements are positive for identification (avoid in-place operation)
        for i in range(self.factor_policy_dim):
            diagonal_vals = torch.abs(diffusion_matrix[:, i, i]) + 1e-6
            # Create a new tensor with updated diagonal values
            diffusion_matrix = diffusion_matrix.clone()
            diffusion_matrix[:, i, i] = diagonal_vals
        
        return diffusion_matrix

class NeuralContinuousFAVAR:
    """
    Neural SDE-based Continuous-Time Factor-Augmented Vector Autoregression Model
    
    This class implements the neural SDE extension of FAVAR following paper2.txt:
    dZ(t) = f_theta(Z(t), t)*dt + g_phi(Z(t), t)*dW(t)
    
    Key features following paper2.txt:
    1. f_theta and g_phi are neural networks parameterizing drift and diffusion
    2. g_phi outputs matrix of shape (p*(K+M), K+M) with lower triangular constraint
       on first K+M rows for structural shock identification
    3. No Cholesky decomposition needed - structural shocks directly identified
       through diffusion matrix structure
    4. IRF computation via Monte Carlo simulation from zero initial state with
       unit shock to specific Brownian motion component
    """
    
    def __init__(self, K=3, p=2, delta_t=1/30, hidden_dims=[64, 32], 
                 learning_rate=0.001, device='cpu', model_save_dir='models'):
        """
        Initialize the Neural Continuous FAVAR model
        
        Parameters:
        -----------
        K : int
            Number of latent factors to extract
        p : int
            VAR lag order for constructing the companion form
        delta_t : float
            Time discretization step (default: 1/30 for daily data in monthly scale)
        hidden_dims : list
            Hidden layer dimensions for neural networks
        learning_rate : float
            Learning rate for optimization
        device : str
            Device for PyTorch computation ('cpu' or 'cuda')
        model_save_dir : str
            Directory to save/load model parameters
        """
        self.K = K
        self.p = p
        self.delta_t = delta_t
        self.hidden_dims = hidden_dims
        self.learning_rate = learning_rate
        self.device = torch.device(device)
        self.M = None  # Number of policy variables (to be determined from data)
        self.model_save_dir = model_save_dir
        
        # Create model directory if it doesn't exist
        os.makedirs(self.model_save_dir, exist_ok=True)
        
        # Neural networks (to be initialized after knowing dimensions)
        self.drift_net = None
        self.diffusion_net = None
        self.optimizer = None
        
        # Data storage
        self.factors = None
        self.rotated_factors = None
        self.policy_vars = None
        self.Z = None
        self.Z_augmented = None
        
        # Training history
        self.training_losses = []
    
    def get_model_filename(self):
        """
        Generate model filename based on model parameters
        """
        return f"neural_favar_K{self.K}_p{self.p}_h{'_'.join(map(str, self.hidden_dims))}.pt"
    
    def save_model(self, save_path=None):
        """
        Save model parameters and training state
        
        Parameters:
        -----------
        save_path : str, optional
            Custom path to save the model. If None, uses default naming convention.
        """
        if save_path is None:
            save_path = os.path.join(self.model_save_dir, self.get_model_filename())
        
        if self.drift_net is None or self.diffusion_net is None:
            print("Warning: Networks not initialized. Cannot save model.")
            return False
        
        # Prepare model state
        model_state = {
            'model_params': {
                'K': self.K,
                'p': self.p,
                'delta_t': self.delta_t,
                'hidden_dims': self.hidden_dims,
                'learning_rate': self.learning_rate,
                'M': self.M
            },
            'drift_net_state': self.drift_net.state_dict(),
            'diffusion_net_state': self.diffusion_net.state_dict(),
            'optimizer_state': self.optimizer.state_dict() if self.optimizer is not None else None,
            'training_losses': self.training_losses
        }
        
        # Save additional data if available
        if hasattr(self, 'X_mean') and hasattr(self, 'X_std'):
            model_state['data_preprocessing'] = {
                'X_mean': self.X_mean,
                'X_std': self.X_std,
                'original_factor_names': getattr(self, 'original_factor_names', None)
            }
        
        if hasattr(self, 'pca'):
            model_state['pca_components'] = self.pca.components_
            model_state['pca_mean'] = self.pca.mean_
        
        if hasattr(self, 'rotation_matrix'):
            model_state['rotation_matrix'] = self.rotation_matrix
        
        if hasattr(self, 'Lambda'):
            model_state['factor_loadings'] = self.Lambda
        
        try:
            torch.save(model_state, save_path)
            print(f"Model saved successfully to: {save_path}")
            return True
        except Exception as e:
            print(f"Error saving model: {e}")
            return False
    
    def load_model(self, load_path=None):
        """
        Load model parameters and training state
        
        Parameters:
        -----------
        load_path : str, optional
            Path to load the model from. If None, uses default naming convention.
        
        Returns:
        --------
        bool : True if loading successful, False otherwise
        """
        if load_path is None:
            load_path = os.path.join(self.model_save_dir, self.get_model_filename())
        
        if not os.path.exists(load_path):
            print(f"Model file not found: {load_path}")
            return False
        
        try:
            model_state = torch.load(load_path, map_location=self.device)
            
            # Load model parameters
            model_params = model_state['model_params']
            
            # Verify compatibility
            if (model_params['K'] != self.K or 
                model_params['p'] != self.p or 
                model_params['hidden_dims'] != self.hidden_dims):
                print("Warning: Model parameters don't match current configuration.")
                print(f"Loaded: K={model_params['K']}, p={model_params['p']}, hidden_dims={model_params['hidden_dims']}")
                print(f"Current: K={self.K}, p={self.p}, hidden_dims={self.hidden_dims}")
                return False
            
            # Update model parameters
            self.M = model_params.get('M', self.M)
            self.delta_t = model_params.get('delta_t', self.delta_t)
            
            # Initialize networks if not already done
            if self.drift_net is None or self.diffusion_net is None:
                # Need to determine input dimension
                input_dim = self.p * (self.K + (self.M or 1))  # Fallback if M not set
                self.initialize_networks(input_dim)
            
            # Load network states
            self.drift_net.load_state_dict(model_state['drift_net_state'])
            self.diffusion_net.load_state_dict(model_state['diffusion_net_state'])
            
            # Load optimizer state if available
            if model_state.get('optimizer_state') is not None and self.optimizer is not None:
                self.optimizer.load_state_dict(model_state['optimizer_state'])
            
            # Load training history
            self.training_losses = model_state.get('training_losses', [])
            
            # Load additional data if available
            if 'data_preprocessing' in model_state:
                data_prep = model_state['data_preprocessing']
                self.X_mean = data_prep.get('X_mean')
                self.X_std = data_prep.get('X_std')
                self.original_factor_names = data_prep.get('original_factor_names')
            
            if 'pca_components' in model_state:
                # Reconstruct PCA object
                from sklearn.decomposition import PCA
                self.pca = PCA(n_components=self.K)
                self.pca.components_ = model_state['pca_components']
                self.pca.mean_ = model_state['pca_mean']
            
            if 'rotation_matrix' in model_state:
                self.rotation_matrix = model_state['rotation_matrix']
            
            if 'factor_loadings' in model_state:
                self.Lambda = model_state['factor_loadings']
            
            print(f"Model loaded successfully from: {load_path}")
            print(f"Training history: {len(self.training_losses)} epochs")
            
            return True
            
        except Exception as e:
            print(f"Error loading model: {e}")
            return False
    
    def check_and_load_existing_model(self):
        """
        Check if a model with current parameters exists and load it
        
        Returns:
        --------
        bool : True if model was loaded, False if no existing model found
        """
        model_path = os.path.join(self.model_save_dir, self.get_model_filename())
        if os.path.exists(model_path):
            print(f"Found existing model: {model_path}")
            return self.load_model(model_path)
        return False
    
    def extract_factors(self, X):
        """
        Extract latent factors using PCA (same as linear model)
        
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
        
        # Store explained variance ratio
        self.explained_variance_ratio = self.pca.explained_variance_ratio_
        print(f"Cumulative explained variance: {np.cumsum(self.explained_variance_ratio)[-1]:.3f}")
        
        return factors
    
    def estimate_factor_loadings(self, original_var_names=None):
        """
        Estimate factor loadings by regressing original variables on rotated factors
        (Same as linear model)
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
    
    def cal_rotated_factors(self, factors, ffr):
        """
        Calculate rotated factors (same as linear model)
        """
        factors = factors.copy()
        print("Calculating rotated factors...")
        for pc in range(factors.shape[1]):
            # Calculate beta_ffr for each factor
            beta_ffr = np.corrcoef(factors[:, pc], ffr[:, 0])[0, 1]
            factors[:, pc] -= beta_ffr * ffr[:, 0]
        self.rotated_factors = factors
        print("Rotated factors calculated.")
        return factors
    
    def construct_state_vector(self, factors, policy_vars):
        """
        Construct the joint state vector Z(t) = [F(t), Y(t)]
        (Same as linear model)
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
        (Same as linear model)
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
    
    def initialize_networks(self, input_dim):
        """
        Initialize the neural networks
        
        Parameters:
        -----------
        input_dim : int
            Dimension of the augmented state vector
        """
        print(f"Initializing neural networks with input dimension: {input_dim}")
        
        # Calculate factor_policy_dim: K + M
        factor_policy_dim = self.K + self.M
        
        # Initialize drift and diffusion networks
        self.drift_net = DriftNetwork(input_dim, self.hidden_dims).to(self.device)
        self.diffusion_net = DiffusionNetwork(input_dim, factor_policy_dim, self.hidden_dims).to(self.device)
        
        # Initialize optimizer
        params = list(self.drift_net.parameters()) + list(self.diffusion_net.parameters())
        self.optimizer = optim.Adam(params, lr=self.learning_rate)
        
        print(f"Drift network parameters: {sum(p.numel() for p in self.drift_net.parameters())}")
        print(f"Diffusion network parameters: {sum(p.numel() for p in self.diffusion_net.parameters())}")
        print(f"Diffusion matrix shape will be: ({input_dim}, {factor_policy_dim})")
    
    def compute_log_likelihood(self, z_batch, t_batch):
        """
        Compute log-likelihood for a batch of state transitions
        
        Parameters:
        -----------
        z_batch : torch.Tensor
            State vectors (batch_size, seq_len, input_dim)
        t_batch : torch.Tensor
            Time values (batch_size, seq_len, 1)
            
        Returns:
        --------
        log_likelihood : torch.Tensor
            Log-likelihood value
        """
        batch_size, seq_len, input_dim = z_batch.shape
        total_log_likelihood = 0.0
        
        for i in range(seq_len - 1):
            z_current = z_batch[:, i, :]  # (batch_size, input_dim)
            z_next = z_batch[:, i + 1, :]  # (batch_size, input_dim)
            t_current = t_batch[:, i, :]  # (batch_size, 1)
            
            # Compute drift and diffusion
            drift = self.drift_net(z_current, t_current)  # (batch_size, input_dim)
            diffusion_matrix = self.diffusion_net(z_current, t_current)  # (batch_size, input_dim, factor_policy_dim)
            
            # Clip diffusion matrix values to prevent numerical issues (avoid in-place operations)
            diffusion_matrix = torch.clamp(diffusion_matrix, min=-10.0, max=10.0)
            
            # Expected change
            mu = drift * self.delta_t
            
            # Covariance matrix: Sigma = G @ G^T * dt, where G is (input_dim, factor_policy_dim)
            # Note: For neural SDE, we need to compute the covariance for the noise term
            # The noise is factor_policy_dim dimensional, so we need to project it to input_dim space
            sigma = torch.bmm(diffusion_matrix, diffusion_matrix.transpose(1, 2)) * self.delta_t
            
            # Add stronger regularization to ensure positive definiteness (avoid in-place operations)
            regularization = torch.eye(input_dim, device=self.device).unsqueeze(0).expand(batch_size, -1, -1) * 1e-3
            sigma = sigma + regularization
            
            # # Additional stabilization: use Cholesky decomposition to ensure positive definiteness
            # try:
            #     # Try Cholesky decomposition
            #     L = torch.linalg.cholesky(sigma)
            #     # Reconstruct from Cholesky to ensure numerical stability
            #     sigma = torch.bmm(L, L.transpose(1, 2))
            # except RuntimeError:
            #     # If Cholesky fails, use eigenvalue decomposition for stabilization
            #     eigenvals, eigenvecs = torch.linalg.eigh(sigma)
            #     # Ensure all eigenvalues are positive (avoid in-place operations)
            #     eigenvals_positive = torch.clamp(eigenvals, min=1e-6)
            #     # Reconstruct positive definite matrix
            #     sigma = torch.bmm(torch.bmm(eigenvecs, torch.diag_embed(eigenvals_positive)), eigenvecs.transpose(1, 2))
            
            # Observed change
            delta_z = z_next - z_current
            
            # Compute log-likelihood with numerical safeguards
            try:
                dist = MultivariateNormal(mu, sigma)
                log_prob = dist.log_prob(delta_z)
                
                # Check for NaN or infinite values
                if torch.isnan(log_prob).any() or torch.isinf(log_prob).any():
                    raise ValueError("NaN or infinite log probability")
                
                total_log_likelihood += log_prob.mean()
                
            except Exception as e:
                # Fallback to simplified Gaussian loss
                print(f"Warning: Log-likelihood computation error at step {i+1}: {e}")
                inv_sigma_diag = 1.0 / torch.diagonal(sigma, dim1=1, dim2=2).clamp(min=1e-6)
                gaussian_loss = 0.5 * torch.sum(inv_sigma_diag * (delta_z - mu) ** 2, dim=1)
                total_log_likelihood -= gaussian_loss.mean()
        
        return total_log_likelihood / (seq_len - 1)
    
    def train_networks(self, Z_aug, n_epochs=100, batch_size=32, sequence_length=50):
        """
        Train the neural networks using maximum likelihood estimation
        
        Parameters:
        -----------
        Z_aug : np.array
            Augmented state vector
        n_epochs : int
            Number of training epochs
        batch_size : int
            Batch size for training
        sequence_length : int
            Length of sequences for training
        """
        print(f"Training neural networks for {n_epochs} epochs...")
        
        T, input_dim = Z_aug.shape
        print("current device:", self.device)
        # Convert to tensor and normalize
        Z_tensor = torch.FloatTensor(Z_aug).to(self.device)
        
        # Normalize the data to improve training stability (avoid in-place operations)
        Z_mean = Z_tensor.mean(dim=0, keepdim=True)
        Z_std = Z_tensor.std(dim=0, keepdim=True) + 1e-8
        Z_tensor_normalized = (Z_tensor - Z_mean) / Z_std
        
        # Store normalization parameters
        self.Z_mean = Z_mean
        self.Z_std = Z_std
        
        # Create time grid
        time_grid = torch.linspace(0, T * self.delta_t, T).to(self.device)
        
        # Learning rate scheduler
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=10, verbose=True
        )
        
        self.training_losses = []
        best_loss = float('inf')
        patience_counter = 0
        patience = 20
        
        for epoch in tqdm(range(n_epochs), desc="Training"):
            epoch_loss = 0.0
            n_batches = 0
            
            # Set networks to training mode
            self.drift_net.train()
            self.diffusion_net.train()
            
            # Create random sequences
            max_batches = max(1, (T - sequence_length) // batch_size)
            print(f"Max batches per epoch: {max_batches}")
            for _ in range(max_batches):
                # Sample random starting points
                start_indices = torch.randint(0, T - sequence_length, (batch_size,))
                
                # Create batch of sequences
                z_batch = torch.stack([
                    Z_tensor_normalized[start:start + sequence_length] 
                    for start in start_indices
                ])  # (batch_size, sequence_length, input_dim)
                
                t_batch = torch.stack([
                    time_grid[start:start + sequence_length].unsqueeze(1)
                    for start in start_indices
                ])  # (batch_size, sequence_length, 1)
                
                # Compute loss
                self.optimizer.zero_grad()
                
                try:
                    log_likelihood = self.compute_log_likelihood(z_batch, t_batch)
                    loss = -log_likelihood  # Negative log-likelihood
                    
                    # Check for NaN or infinite loss
                    if torch.isnan(loss) or torch.isinf(loss):
                        print(f"Warning: Invalid loss at epoch {epoch+1}, batch {n_batches+1}")
                        continue
                    
                    # Backward pass
                    loss.backward()
                    
                    # Gradient clipping for stability
                    torch.nn.utils.clip_grad_norm_(
                        list(self.drift_net.parameters()) + list(self.diffusion_net.parameters()), 
                        max_norm=1.0
                    )
                    
                    self.optimizer.step()
                    
                    epoch_loss += loss.item()
                    n_batches += 1
                    
                except Exception as e:
                    print(f"Warning: Training error at epoch {epoch+1}: {e}")
                    continue
            
            if n_batches > 0:
                avg_loss = epoch_loss / n_batches
                self.training_losses.append(avg_loss)
                
                # Learning rate scheduling
                scheduler.step(avg_loss)
                
                # Early stopping
                if avg_loss < best_loss:
                    best_loss = avg_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break
                
                if (epoch + 1) % 10 == 0:
                    print(f"Epoch {epoch + 1}/{n_epochs}, Loss: {avg_loss:.6f}, LR: {self.optimizer.param_groups[0]['lr']:.6f}")
            else:
                print(f"Warning: No valid batches in epoch {epoch+1}")
        
        print("Neural network training completed!")
    
    def plot_training_loss(self):
        """
        Plot training loss curve
        """
        if not self.training_losses:
            print("No training history available.")
            return
        
        plt.figure(figsize=(10, 6))
        plt.plot(self.training_losses)
        plt.title('Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Negative Log-Likelihood')
        plt.grid(True, alpha=0.3)
        plt.show()
    
    def compute_irf_neural(self, horizon=30, shock_size=1.0, n_simulations=100, 
                          include_original_vars=False, original_var_names=None):
        """
        Compute impulse response functions using Monte Carlo simulation
        Modified methodology:
        - Start from zero initial state Z_shock(0) = 0
        - Apply one-time unit impulse at t=0 by perturbing drift A and diffusion S
        - All time steps have dW = 0 (deterministic evolution after initial perturbation)
        - IRF computed as mean of multiple simulations for confidence intervals
        
        Parameters:
        -----------
        horizon : int
            Maximum horizon for IRF computation
        shock_size : float
            Size of the structural shock
        n_simulations : int
            Number of Monte Carlo simulations for confidence intervals
        include_original_vars : bool
            Whether to compute IRFs for original variables
        original_var_names : list
            Names of original variables
            
        Returns:
        --------
        irfs : dict
            Dictionary containing IRF results
        """
        if self.drift_net is None or self.diffusion_net is None:
            raise ValueError("Neural networks not trained. Please run fit() first.")
        
        print(f"Computing Neural SDE IRFs using {n_simulations} simulations...")
        print(f"Methodology: Perturbation of A and S at t=0, then deterministic evolution (dW=0)")
        
        # Set up original variable names and estimate factor loadings if needed
        if include_original_vars:
            if original_var_names is None:
                original_var_names = self.original_factor_names
            
            # Estimate factor loadings if not already done
            if not hasattr(self, 'factor_loadings') or not all(name in self.factor_loadings for name in original_var_names):
                self.estimate_factor_loadings(original_var_names)
            
            N_original = len(original_var_names)
        
        h_grid = np.linspace(0, horizon, int(horizon * 30))
        T = len(h_grid)
        n_vars = self.K + self.M
        
        # Zero initial state
        input_dim = self.Z_augmented.shape[1]
        initial_state = torch.zeros(1, input_dim, device=self.device)
        
        irfs = {}
        
        # Only compute monetary policy shock (last variable in factor_policy space)
        shock_idx = self.K + self.M - 1  # Last variable is the policy shock
        shock_name = "monetary_policy_shock"
        
        shock_trajectories = []
        
        if include_original_vars:
            shock_original_trajectories = []
        
        print(f"Simulating {shock_name} with perturbation to A and S at t=0...")
        
        self.drift_net.eval()
        self.diffusion_net.eval()
        
        with torch.no_grad():
            for sim in tqdm(range(n_simulations), desc="Monte Carlo simulations"):
                # Simulate trajectory with perturbation to A and S at t=0
                shock_traj = self._simulate_trajectory_with_perturbation(
                    initial_state, h_grid, shock_size=shock_size, shock_idx=shock_idx
                )
                shock_trajectories.append(shock_traj[:, :n_vars])  # Only factors and policy vars
                
                # Compute original variable trajectories if needed
                if include_original_vars:
                    shock_original = self._compute_original_var_trajectory(shock_traj[:, :self.K], original_var_names)
                    shock_original_trajectories.append(shock_original)
        
        # IRF is the mean trajectory across simulations
        irf_responses = np.mean(shock_trajectories, axis=0)
        
        # Compute confidence intervals
        irf_samples = np.array(shock_trajectories)
        irf_lower = np.quantile(irf_samples, 0.05, axis=0)
        irf_upper = np.quantile(irf_samples, 0.95, axis=0)
        
        irf_result = {
            'horizon': h_grid,
            'responses': irf_responses,
            'lower': irf_lower,
            'upper': irf_upper
        }
        
        # Add original variable IRFs if computed
        if include_original_vars:
            original_irf_responses = np.mean(shock_original_trajectories, axis=0)
            
            # Compute confidence intervals for original variables
            original_irf_samples = np.array(shock_original_trajectories)
            original_irf_lower = np.quantile(original_irf_samples, 0.05, axis=0)
            original_irf_upper = np.quantile(original_irf_samples, 0.95, axis=0)
            
            irf_result.update({
                'original_responses': original_irf_responses,
                'original_lower': original_irf_lower,
                'original_upper': original_irf_upper,
                'original_var_names': original_var_names
            })
        
        irfs[shock_name] = irf_result
        self.irfs = irfs
        
        print("IRF computation completed with confidence intervals")
        
        return irfs
    
    def _simulate_trajectory_with_perturbation(self, initial_state, h_grid, shock_size=1.0, shock_idx=0):
        """
        Simulate a single trajectory with perturbation to drift A and diffusion S at t=0
        Modified methodology:
        - Start from zero initial condition Z_shock(0) = 0  
        - At t=0: perturb drift A and diffusion S by adding shock to specified component
        - All time steps have dW = 0 (deterministic evolution)
        - Perturbation propagates through the system dynamics
        
        Parameters:
        -----------
        initial_state : torch.Tensor
            Initial state (1, input_dim) - should be zeros
        h_grid : np.array
            Time grid
        shock_size : float
            Size of perturbation to apply to A and S
        shock_idx : int
            Index of component to perturb (in factor_policy space)
            
        Returns:
        --------
        trajectory : np.array
            Simulated trajectory (T, input_dim)
        """
        T = len(h_grid)
        input_dim = initial_state.shape[1]
        factor_policy_dim = self.K + self.M
        trajectory = np.zeros((T, input_dim), dtype=np.float32)
        
        # Set initial state (should be zeros)
        trajectory[0] = initial_state.detach().cpu().numpy()
        
        for i in range(T - 1):
            z_current = torch.FloatTensor(trajectory[i:i+1]).to(self.device)  # Keep batch dimension
            t_current = torch.FloatTensor([[h_grid[i]]]).to(self.device)
            
            # Compute drift and diffusion at current state
            drift = self.drift_net(z_current, t_current).detach().cpu().numpy()
            diffusion_matrix = self.diffusion_net(z_current, t_current).detach().cpu().numpy()
            
            # Apply perturbation to A (drift) and S (diffusion) ONLY at t=0
            if i == 0:
                noise = np.zeros((1, factor_policy_dim), dtype=np.float32)
                noise[0, shock_idx] += shock_size  # Perturb specified component
            else:
                noise = np.zeros((1, factor_policy_dim), dtype=np.float32)
            
            # Add small noise to drift and diffusion for stability
            drift += np.random.normal(scale=0.1, size=drift.shape).astype(np.float32)
            diffusion_matrix += np.random.normal(scale=0.1, size=diffusion_matrix.shape).astype(np.float32)
            
            # Euler step: Z(t+dt) = Z(t) + f_theta(Z(t),t)*dt + g_phi(Z(t),t)*dW(t)
            # Since dW = 0, this becomes: Z(t+dt) = Z(t) + f_theta(Z(t),t)*dt
            drift_term = drift * self.delta_t
            # Fix: use numpy operations for numpy arrays
            noise_expanded = np.expand_dims(noise, axis=2)  # (1, factor_policy_dim, 1)
            diffusion_term = np.matmul(diffusion_matrix, noise_expanded).squeeze(axis=2) * np.sqrt(self.delta_t)
            
            trajectory[i + 1] = trajectory[i] + drift_term.flatten() + diffusion_term.flatten()
        
        return trajectory
    
    def _simulate_trajectory(self, initial_state, h_grid, shock_size=0.0, shock_idx=0, randn_noise=True):
        """
        Simulate a single trajectory from the neural SDE
        
        Parameters:
        -----------
        initial_state : torch.Tensor
            Initial state (1, input_dim)
        h_grid : np.array
            Time grid
        shock_size : float
            Size of shock to apply
        shock_idx : int
            Index of variable to shock
            
        Returns:
        --------
        trajectory : np.array
            Simulated trajectory (T, input_dim)
        """
        T = len(h_grid)
        input_dim = initial_state.shape[1]
        factor_policy_dim = self.K + self.M
        trajectory = torch.zeros(T, input_dim, device=self.device)
        # Set initial state
        trajectory[0] = initial_state.clone()
        
        for i in range(T - 1):
            z_current = trajectory[i:i+1]  # Keep batch dimension
            t_current = torch.FloatTensor([[h_grid[i]]]).to(self.device)
            
            # Compute drift and diffusion
            drift = self.drift_net(z_current, t_current)
            diffusion_matrix = self.diffusion_net(z_current, t_current)
            
            # Sample noise for the factor_policy_dim dimensional Brownian motion
            if randn_noise and i > 0:
                noise = torch.randn(1, factor_policy_dim, device=self.device)
            else:
                noise = torch.zeros(1, factor_policy_dim, device=self.device)
            
            # Apply shock at first time step
            if i == 0 and shock_size > 0:
                noise = noise.clone()  # Create a copy to avoid in-place operation
                noise[0, shock_idx] += shock_size
            
            # Euler step
            drift_term = drift * self.delta_t
            diffusion_term = torch.bmm(diffusion_matrix, noise.unsqueeze(2)).squeeze(2) * np.sqrt(self.delta_t)
            
            trajectory[i + 1] = trajectory[i] + drift_term + diffusion_term
        
        return trajectory.cpu().numpy()
    
    def _compute_original_var_trajectory(self, factor_trajectory, original_var_names):
        """
        Compute original variable trajectory from factor trajectory using regression coefficients
        
        Parameters:
        -----------
        factor_trajectory : np.array
            Factor trajectory (T, K)
        original_var_names : list
            Names of original variables
            
        Returns:
        --------
        original_trajectory : np.array
            Original variable trajectory (T, N_original)
        """
        T, K = factor_trajectory.shape
        N_original = len(original_var_names)
        
        original_trajectory = np.zeros((T, N_original))
        
        for i, var_name in enumerate(original_var_names):
            if var_name in self.factor_loadings:
                loadings = self.factor_loadings[var_name]
                # Trajectory = intercept + factor_coeffs @ factors
                var_trajectory = (factor_trajectory @ loadings['factor_coeffs'])
                original_trajectory[:, i] = var_trajectory
        
        return original_trajectory
    
    def fit(self, data, n_epochs=100, batch_size=32, force_retrain=False, auto_save=True):
        """
        Fit the neural continuous FAVAR model to data
        
        Parameters:
        -----------
        data : pd.DataFrame or np.array
            Data with columns: [factor_vars..., policy_var]
            Last column should be the policy variable (interest rate)
        n_epochs : int
            Number of training epochs for neural networks
        batch_size : int
            Batch size for training
        force_retrain : bool
            If True, force retraining even if a saved model exists
        auto_save : bool
            If True, automatically save the model after training
        """
        print("=" * 50)
        print("FITTING NEURAL CONTINUOUS FAVAR MODEL")
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

        # Step 2: Calculate rotated factors
        rotated_factors = self.cal_rotated_factors(factors, Y)

        # Step 3: Estimate factor loadings using regression after rotation
        print("Estimating factor loadings using regression...")
        self.estimate_factor_loadings()

        # Step 4: Construct state vector Z(t) = [F(t), Y(t)]
        Z = self.construct_state_vector(rotated_factors, Y)
        
        # Step 5: Create augmented state for VAR(p) -> VAR(1)
        Z_aug = self.create_augmented_state(Z)
        
        # Step 6: Initialize neural networks
        input_dim = Z_aug.shape[1]
        self.initialize_networks(input_dim)

        # Check for existing trained model
        if not force_retrain:
            print("Checking for existing trained model...")
            if self.check_and_load_existing_model():
                print("Loaded existing model. Use force_retrain=True to retrain.")
                return self
            else:
                print("No existing model found. Starting fresh training.")

        
        # Step 7: Train neural networks
        self.train_networks(Z_aug, n_epochs=n_epochs, batch_size=batch_size)
        
        # Step 8: Auto-save model if requested
        if auto_save:
            print("Saving trained model...")
            self.save_model()
        
        print("=" * 50)
        print("NEURAL MODEL ESTIMATION COMPLETE")
        print("=" * 50)
        
        return self
    
    def plot_impulse_responses(self, variables=None, shock='monetary_policy_shock', 
                             figsize=(4, 4), save_path=None, show_ci=True, 
                             plot_original_vars=False, max_original_vars=12):
        """
        Plot impulse response functions (same interface as linear model)
        """
        if not hasattr(self, 'irfs'):
            print("IRFs has not been computed. Please run compute_irf_neural() first.")
            return
        
        if shock not in self.irfs:
            print(f"can't find '{shock}'. Available shocks: {list(self.irfs.keys())}")
            return

        irf_data = self.irfs[shock]
        horizon = irf_data['horizon']
        
        # Determine what to plot
        if plot_original_vars:
            if 'original_responses' not in irf_data:
                print("Original variable IRFs not available. Please run compute_irf_neural() with include_original_vars=True.")
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
            plot_title_prefix = "Original Variables (Neural SDE)"
            
        else:
            responses = irf_data['responses']
            lower_ci = irf_data['lower']
            upper_ci = irf_data['upper']
            
            if variables is None:
                variables = list(range(responses.shape[1]))
            
            plot_title_prefix = "Factors & Policy (Neural SDE)"
        
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
                    color=sns.color_palette()[1], label='mean (Neural)')  # Different color
            
            if show_ci:
                ax.fill_between(horizon, 
                            lower_ci[:, var_idx], 
                            upper_ci[:, var_idx], 
                            color=sns.color_palette()[1], 
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

def main():
    """
    Example usage of the Neural Continuous FAVAR model
    """
    print("NEURAL SDE CONTINUOUS FAVAR MODEL IMPLEMENTATION")
    print("================================================")
    
    # Check if CUDA is available
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    x_path = "processed_x.csv"  # Path to processed macro data
    y_path = "processed_y.csv"  # Path to processed policy data

    if os.path.exists(x_path) and os.path.exists(y_path):
        print(f"\nLoading data from {x_path} and {y_path}...")
        X = pd.read_csv(x_path)
        # Drop non-macro columns
        X = X.drop(columns=['open', 'high', 'low', 'close', 'volume'], errors='ignore')
        Y = pd.read_csv(y_path)
        print(f"Data shapes: X={X.shape}, Y={Y.shape}")

        data_df = pd.merge(X, Y, on='date', how='inner')
        data_df.drop(columns=['date'], inplace=True)  # Drop date column if exists
        print(f"Merged data shape: {data_df.shape}")
        print(data_df.head())
    else:
        # Simulate some data for demonstration
        print("\nGenerating simulated data for demonstration...")
        np.random.seed(42)
        T = 1000  # Number of observations
        N = 50    # Number of macro variables (before PCA)
        
        # Generate some autocorrelated data with nonlinear dynamics
        data_sim = np.zeros((T, N + 1))  # +1 for policy variable
        
        for i in range(N + 1):
            for t in range(1, T):
                # Add some nonlinearity
                nonlinear_term = 0.1 * np.tanh(data_sim[t-1, i])
                data_sim[t, i] = 0.7 * data_sim[t-1, i] + nonlinear_term + np.random.normal(0, 1)
        
        # Add some cross-correlation
        for i in range(N):
            data_sim[:, i] += 0.3 * data_sim[:, -1] + np.random.normal(0, 0.5, T)
        
        # Create DataFrame
        columns = [f'var_{i}' for i in range(N)] + ['interest_rate']
        data_df = pd.DataFrame(data_sim, columns=columns)
        
        print(f"Simulated data shape: {data_df.shape}")
    
    # Initialize and fit the model
    model = NeuralContinuousFAVAR(
        K=3, 
        p=2, 
        delta_t=1/30, 
        hidden_dims=[64, 32],
        learning_rate=0.001,
        device=device
    )
    
    # Fit the model
    model.fit(data_df, n_epochs=30, batch_size=512)  # Fewer epochs for demo
    
    # Plot training loss
    model.plot_training_loss()
    
    # Compute impulse responses
    print("\nComputing neural SDE impulse responses...")
    original_var_names = ['IP', 'PUNEW', 'FMFBA', 'PMCP', 'FM2', 'EXRJAN', 'LHUR', 'FSDXP', 'GMCQ'] if os.path.exists(x_path) else [f'var_{i}' for i in range(9)]
    
    irfs = model.compute_irf_neural(
        horizon=40,
        shock_size=1.0,
        n_simulations=50,  # Fewer simulations for demo
        include_original_vars=True,
        original_var_names=original_var_names
    )
    
    # Plot factor IRFs
    print("\nPlotting factor impulse responses...")
    model.plot_impulse_responses(shock='monetary_policy_shock')
    
    # Plot original variable IRFs
    print("\nPlotting original variable impulse responses...")
    model.plot_impulse_responses(
        shock='monetary_policy_shock', 
        plot_original_vars=True, 
        max_original_vars=9
    )
    
    print("\nNeural model fitting and analysis complete!")
    print("\nKey differences from linear model:")
    print("1. Uses neural networks for drift and diffusion functions")
    print("2. Can capture nonlinear dynamics and state-dependent effects")
    print("3. IRFs computed via Monte Carlo simulation (no closed form)")
    print("4. Requires more computation but offers greater flexibility")

if __name__ == "__main__":
    main()
