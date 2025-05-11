from abc import ABC, abstractmethod
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List, Union
import numpy as np

class Encoder(ABC):
    """
    Abstract base class for encoders to be used with BayesianOptimizer.
    
    Encoders transform input data before it's passed to the Gaussian Process model.
    This can be useful for:
    - Dimensionality reduction
    - Feature extraction
    - Handling categorical variables
    - Applying domain-specific transformations
    """
    
    @abstractmethod
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Transform input data into the encoded space.
        
        Args:
            x (torch.Tensor): Input data of shape (n, input_dim)
            
        Returns:
            torch.Tensor: Encoded data of shape (n, encoded_dim)
        """
        pass
    
    @abstractmethod
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        Transform data from encoded space back to original space.
        
        Args:
            z (torch.Tensor): Encoded data of shape (n, encoded_dim)
            
        Returns:
            torch.Tensor: Decoded data of shape (n, input_dim)
        """
        pass
    
    def reconstruction_loss(self, x: torch.Tensor, reduction: str = 'mean') -> torch.Tensor:
        """
        Calculate the reconstruction loss for a set of points.
        
        Args:
            x (torch.Tensor): Input data of shape (n, input_dim)
            reduction (str): Reduction method ('mean', 'sum', or 'none')
            
        Returns:
            torch.Tensor: Reconstruction loss
        """
        # Encode and then decode
        z = self.encode(x)
        x_reconstructed = self.decode(z)
        
        # Calculate squared error
        squared_error = (x - x_reconstructed).pow(2)
        
        # Apply reduction
        if reduction == 'mean':
            return squared_error.mean()
        elif reduction == 'sum':
            return squared_error.sum()
        elif reduction == 'none':
            return squared_error
        else:
            raise ValueError(f"Unknown reduction method: {reduction}")
    
    @property
    @abstractmethod
    def input_dim(self) -> int:
        """
        The dimension of the input space.
        
        Returns:
            int: Dimension of input space
        """
        pass
    
    @property
    @abstractmethod
    def encoded_dim(self) -> int:
        """
        The dimension of the encoded space.
        
        Returns:
            int: Dimension of encoded space
        """
        pass


class IdentityEncoder(Encoder):
    """
    A simple identity encoder that doesn't transform the data.
    Useful as a default or for testing.
    """
    
    def __init__(self, dim: int):
        """
        Initialize the identity encoder.
        
        Args:
            dim (int): Dimension of the input/output space
        """
        self._dim = dim
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Identity transformation."""
        return x
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Identity transformation."""
        return z
    
    def reconstruction_loss(self, x: torch.Tensor, reduction: str = 'mean') -> torch.Tensor:
        """
        For identity encoder, reconstruction loss is always zero.
        """
        if reduction == 'mean' or reduction == 'sum':
            return torch.tensor(0.0, device=x.device)
        elif reduction == 'none':
            return torch.zeros_like(x)
        else:
            raise ValueError(f"Unknown reduction method: {reduction}")
    
    @property
    def input_dim(self) -> int:
        """Dimension of input space."""
        return self._dim
    
    @property
    def encoded_dim(self) -> int:
        """Dimension of encoded space (same as input for identity)."""
        return self._dim


class PCAEncoder(Encoder):
    """
    A PCA-based encoder that reduces dimensionality.
    """
    
    def __init__(self, input_dim: int, encoded_dim: int, data: Optional[torch.Tensor] = None):
        """
        Initialize the PCA encoder.
        
        Args:
            input_dim (int): Dimension of the input space
            encoded_dim (int): Dimension of the encoded space (must be <= input_dim)
            data (Optional[torch.Tensor]): Data to fit the PCA on. If None, fit must be called later.
        """
        self._input_dim = input_dim
        self._encoded_dim = encoded_dim
        
        if encoded_dim > input_dim:
            raise ValueError(f"Encoded dimension ({encoded_dim}) must be <= input dimension ({input_dim})")
        
        # Initialize PCA components
        self.components = None
        self.mean = None
        
        # Fit PCA if data is provided
        if data is not None:
            self.fit(data)
    
    def fit(self, data: torch.Tensor) -> None:
        """
        Fit the PCA encoder on the provided data.
        
        Args:
            data (torch.Tensor): Data to fit PCA on, shape (n, input_dim)
        """
        if data.shape[1] != self._input_dim:
            raise ValueError(f"Data dimension ({data.shape[1]}) must match input dimension ({self._input_dim})")
        
        # Center the data
        self.mean = data.mean(dim=0)
        centered_data = data - self.mean
        
        # Compute covariance matrix
        cov = torch.mm(centered_data.t(), centered_data) / (data.shape[0] - 1)
        
        # Compute eigenvalues and eigenvectors
        eigenvalues, eigenvectors = torch.linalg.eigh(cov)
        
        # Sort eigenvectors by eigenvalues in descending order
        idx = torch.argsort(eigenvalues, descending=True)
        eigenvectors = eigenvectors[:, idx]
        
        # Select top components
        self.components = eigenvectors[:, :self._encoded_dim]
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Transform input data into the encoded space using PCA.
        
        Args:
            x (torch.Tensor): Input data of shape (n, input_dim)
            
        Returns:
            torch.Tensor: Encoded data of shape (n, encoded_dim)
        """
        if self.components is None:
            raise RuntimeError("PCA has not been fit. Call fit() first.")
        
        # Center the data
        centered_x = x - self.mean
        
        # Project onto principal components
        return torch.mm(centered_x, self.components)
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        Transform data from encoded space back to original space.
        
        Args:
            z (torch.Tensor): Encoded data of shape (n, encoded_dim)
            
        Returns:
            torch.Tensor: Decoded data of shape (n, input_dim)
        """
        if self.components is None:
            raise RuntimeError("PCA has not been fit. Call fit() first.")
        
        # Project back to original space
        reconstructed = torch.mm(z, self.components.t())
        
        # Add back the mean
        return reconstructed + self.mean
    
    @property
    def input_dim(self) -> int:
        """Dimension of input space."""
        return self._input_dim
    
    @property
    def encoded_dim(self) -> int:
        """Dimension of encoded space."""
        return self._encoded_dim


class VAEEncoderModule(nn.Module):
    """
    Encoder part of the VAE.
    """
    def __init__(self, input_dim: int, latent_dim: int, hidden_dims: List[int]):
        super().__init__()
        
        # Build encoder layers
        modules = []
        
        # Input layer
        in_features = input_dim
        
        # Hidden layers
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Linear(in_features, h_dim),
                    nn.GELU(),
                    nn.BatchNorm1d(h_dim)
                )
            )
            in_features = h_dim
        
        self.encoder = nn.Sequential(*modules)
        
        # Mean and variance layers
        self.fc_mu = nn.Linear(hidden_dims[-1], latent_dim)
        self.fc_var = nn.Linear(hidden_dims[-1], latent_dim)
        
    def forward(self, x):
        # Pass through encoder layers
        result = self.encoder(x)
        
        # Get mean and log variance
        mu = self.fc_mu(result)
        log_var = self.fc_var(result)
        
        return mu, log_var

class VAEDecoderModule(nn.Module):
    """
    Decoder part of the VAE.
    """
    def __init__(self, latent_dim: int, output_dim: int, hidden_dims: List[int]):
        super().__init__()
        
        # Build decoder layers
        modules = []
        
        # Reverse hidden dimensions
        hidden_dims = hidden_dims[::-1]
        
        # Input layer
        modules.append(
            nn.Sequential(
                nn.Linear(latent_dim, hidden_dims[0]),
                nn.GELU(),
                nn.BatchNorm1d(hidden_dims[0])
            )
        )
        
        # Hidden layers
        for i in range(len(hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.Linear(hidden_dims[i], hidden_dims[i + 1]),
                    nn.GELU(),
                    nn.BatchNorm1d(hidden_dims[i + 1])
                )
            )
        
        # Output layer
        self.decoder = nn.Sequential(*modules)
        self.final_layer = nn.Linear(hidden_dims[-1], output_dim)
        
    def forward(self, z):
        # Pass through decoder layers
        result = self.decoder(z)
        
        # Final layer without activation (for regression)
        return self.final_layer(result)

class EncoderDecoderPair(nn.Module):
    """
    A paired encoder-decoder module using the standalone VAE components.
    """
    def __init__(self, input_dim: int, latent_dim: int, hidden_dims: List[int]):
        super().__init__()
        self.encoder = VAEEncoderModule(input_dim, latent_dim, hidden_dims)
        self.decoder = VAEDecoderModule(latent_dim, input_dim, hidden_dims)
    
    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode input to latent distribution parameters."""
        return self.encoder(x)
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent vector to reconstruction."""
        return self.decoder(z)
    
    def reparameterize(self, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick."""
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Full forward pass: encode, reparameterize, decode."""
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        recon = self.decode(z)
        return recon, mu, log_var

class VAEEncoder(Encoder):
    """
    A Variational Autoencoder (VAE) based encoder with GELU activations.
    """
    
    def __init__(
        self, 
        input_dim: int, 
        latent_dim: int, 
        hidden_dims: List[int] = [128, 64],
        beta: float = 1.0,
        train_iters: int = 1000,
        batch_size: int = 64,
        learning_rate: float = 1e-3,
        device: str = 'cpu'
    ):
        """
        Initialize the VAE encoder.
        
        Args:
            input_dim (int): Dimension of the input space
            latent_dim (int): Dimension of the latent (encoded) space
            hidden_dims (List[int]): List of hidden dimensions for encoder and decoder
            beta (float): Weight of the KL divergence term (beta-VAE)
            train_iters (int): Number of training iterations
            batch_size (int): Batch size for training
            learning_rate (float): Learning rate for training
            device (str): Device to use for training ('cpu' or 'cuda')
        """
        self._input_dim = input_dim
        self._encoded_dim = latent_dim
        self.hidden_dims = hidden_dims
        self.beta = beta
        self.train_iters = train_iters
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.device = device
        
        # Initialize encoder and decoder
        self.encoder_model = VAEEncoderModule(input_dim, latent_dim, hidden_dims).to(device)
        self.decoder_model = VAEDecoderModule(latent_dim, input_dim, hidden_dims).to(device)
        
        # Initialize optimizer
        self.optimizer = None
        
        # Training state
        self.is_fitted = False

    def eval(self):
        self.encoder_model.eval()
        self.decoder_model.eval()

    def train(self):
        self.encoder_model.train()
        self.decoder_model.train()
    
    def reparameterize(self, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        """
        Reparameterization trick to sample from N(mu, var) from N(0,1).
        
        Args:
            mu (torch.Tensor): Mean of the latent Gaussian
            log_var (torch.Tensor): Log variance of the latent Gaussian
            
        Returns:
            torch.Tensor: Sampled latent vector
        """
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def fit(self, data: torch.Tensor) -> None:
        """
        Fit the VAE encoder on the provided data.
        
        Args:
            data (torch.Tensor): Data to fit VAE on, shape (n, input_dim)
        """
        if data.shape[1] != self._input_dim:
            raise ValueError(f"Data dimension ({data.shape[1]}) must match input dimension ({self._input_dim})")
        
        # Move data to device
        data = data.to(self.device)
        
        # Create optimizer
        params = list(self.encoder_model.parameters()) + list(self.decoder_model.parameters())
        self.optimizer = torch.optim.Adam(params, lr=self.learning_rate)
        
        # Training loop
        self.encoder_model.train()
        self.decoder_model.train()
        
        n_samples = data.shape[0]
        
        for epoch in range(self.train_iters):
            # Shuffle data
            indices = torch.randperm(n_samples)
            
            total_loss = 0
            total_recon_loss = 0
            total_kl_loss = 0
            
            # Mini-batch training
            for start_idx in range(0, n_samples, self.batch_size):
                end_idx = min(start_idx + self.batch_size, n_samples)
                batch_indices = indices[start_idx:end_idx]
                batch_data = data[batch_indices]
                
                # Forward pass
                mu, log_var = self.encoder_model(batch_data)
                z = self.reparameterize(mu, log_var)
                recon_batch = self.decoder_model(z)
                
                # Compute loss
                recon_loss = F.mse_loss(recon_batch, batch_data, reduction='sum')
                kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
                loss = recon_loss + self.beta * kl_loss
                
                # Backward pass and optimize
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item()
                total_recon_loss += recon_loss.item()
                total_kl_loss += kl_loss.item()
            
            # Print progress
            if (epoch + 1) % 100 == 0:
                avg_loss = total_loss / n_samples
                avg_recon_loss = total_recon_loss / n_samples
                avg_kl_loss = total_kl_loss / n_samples
                print(f"Epoch {epoch+1}/{self.train_iters}, Loss: {avg_loss:.4f}, "
                      f"Recon Loss: {avg_recon_loss:.4f}, KL Loss: {avg_kl_loss:.4f}")
        
        # Set to evaluation mode
        self.encoder_model.eval()
        self.decoder_model.eval()
        self.is_fitted = True
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Transform input data into the encoded space using the VAE encoder.
        
        Args:
            x (torch.Tensor): Input data of shape (n, input_dim)
            
        Returns:
            torch.Tensor: Encoded data of shape (n, encoded_dim)
        """
        if not self.is_fitted:
            raise RuntimeError("VAE has not been fit. Call fit() first.")
        
        # Move to device and set to eval mode
        x = x.to(self.device)
        self.encoder_model.eval()
        
        # Encode without gradients
        with torch.no_grad():
            mu, log_var = self.encoder_model(x)
            
            # For Bayesian optimization, we use the mean of the latent distribution
            # rather than sampling, to ensure deterministic behavior
            return mu
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        Transform data from encoded space back to original space using the VAE decoder.
        
        Args:
            z (torch.Tensor): Encoded data of shape (n, encoded_dim)
            
        Returns:
            torch.Tensor: Decoded data of shape (n, input_dim)
        """
        if not self.is_fitted:
            raise RuntimeError("VAE has not been fit. Call fit() first.")
        
        # Move to device and set to eval mode
        z = z.to(self.device)
        self.decoder_model.eval()
        
        # Decode without gradients
        with torch.no_grad():
            return self.decoder_model(z)
    
    def reconstruction_loss(self, x: torch.Tensor, reduction: str = 'mean') -> torch.Tensor:
        """
        Calculate the reconstruction loss for a set of points.
        
        Args:
            x (torch.Tensor): Input data of shape (n, input_dim)
            reduction (str): Reduction method ('mean', 'sum', or 'none')
            
        Returns:
            torch.Tensor: Reconstruction loss
        """
        if not self.is_fitted:
            raise RuntimeError("VAE has not been fit. Call fit() first.")
        
        # Move to device
        x = x.to(self.device)
        
        # Encode and decode
        with torch.no_grad():
            mu, _ = self.encoder_model(x)
            x_reconstructed = self.decoder_model(mu)
        
        # Calculate squared error
        squared_error = (x - x_reconstructed).pow(2)
        
        # Apply reduction
        if reduction == 'mean':
            return squared_error.mean()
        elif reduction == 'sum':
            return squared_error.sum()
        elif reduction == 'none':
            return squared_error
        else:
            raise ValueError(f"Unknown reduction method: {reduction}")
    
    @property
    def input_dim(self) -> int:
        """Dimension of input space."""
        return self._input_dim
    
    @property
    def encoded_dim(self) -> int:
        """Dimension of encoded space."""
        return self._encoded_dim


class EnsembleVAEEncoder(Encoder):
    """
    An ensemble of VAE encoders that combines their outputs and adds an information radius dimension.
    The latent space is composed of:
    - First n-1 dimensions: mean of the encoder means (used for encoding/decoding)
    - Last dimension: information radius of the encoder distributions (appended only at test time)
    """
    
    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        n_encoders: int = 3,
        hidden_dims: List[int] = [128, 64],
        beta: float = 1.0,
        train_iters: int = 1000,
        batch_size: int = 64,
        learning_rate: float = 1e-3,
        device: str = 'cpu'
    ):
        """
        Initialize the ensemble VAE encoder.
        
        Args:
            input_dim (int): Dimension of the input space
            latent_dim (int): Dimension of the combined latent space (including info radius)
            n_encoders (int): Number of VAE encoders in the ensemble
            hidden_dims (List[int]): List of hidden dimensions for encoders and decoder
            beta (float): Weight of the KL divergence term (beta-VAE)
            train_iters (int): Number of training iterations
            batch_size (int): Batch size for training
            learning_rate (float): Learning rate for training
            device (str): Device to use for training ('cpu' or 'cuda')
        """
        if latent_dim <= 1:
            raise ValueError("latent_dim must be greater than 1")
        
        self._input_dim = input_dim
        self._encoded_dim = latent_dim
        self.n_encoders = n_encoders
        self.device = device
        
        # Create ensemble of encoders
        # Each encoder has latent_dim - 1 dimensions
        self.encoders = [
            VAEEncoderModule(input_dim, latent_dim - 1, hidden_dims).to(device)
            for _ in range(n_encoders)
        ]
        
        # Single shared decoder that takes latent_dim - 1 dimensions
        self.decoder = VAEDecoderModule(latent_dim - 1, input_dim, hidden_dims).to(device)
        
        # Training parameters
        self.hidden_dims = hidden_dims
        self.beta = beta
        self.train_iters = train_iters
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        
        # Initialize optimizer
        self.optimizer = None
        
        # Training state
        self.is_fitted = False
    
    def calculate_information_radius(self, mus: List[torch.Tensor], log_vars: List[torch.Tensor]) -> torch.Tensor:
        """
        Calculate the information radius of the encoder distributions.
        This is based on the Jensen-Shannon divergence between the distributions.
        
        Args:
            mus (List[torch.Tensor]): List of mean vectors from each encoder
            log_vars (List[torch.Tensor]): List of log variance vectors from each encoder
            
        Returns:
            torch.Tensor: Information radius for each input point
        """
        n = len(mus)
        batch_size = mus[0].shape[0]
        
        # Calculate average distribution parameters
        mean_mu = torch.stack(mus).mean(dim=0)
        mean_var = torch.stack([torch.exp(log_var) for log_var in log_vars]).mean(dim=0)
        mean_log_var = torch.log(mean_var)
        
        # Calculate KL divergence from each distribution to the average
        kl_divs = []
        for i in range(n):
            kl_div = 0.5 * (
                torch.exp(log_vars[i]) / mean_var
                + (mus[i] - mean_mu).pow(2) / mean_var
                - 1
                + mean_log_var
                - log_vars[i]
            ).sum(dim=1)
            kl_divs.append(kl_div)
        
        # Information radius is the average KL divergence
        info_radius = torch.stack(kl_divs).mean(dim=0)
        
        # Normalize to [0, 1] using sigmoid
        return torch.sigmoid(info_radius)
    
    def fit(self, data: torch.Tensor) -> None:
        """
        Fit the ensemble VAE encoder on the provided data.
        
        Args:
            data (torch.Tensor): Data to fit VAE on, shape (n, input_dim)
        """
        if data.shape[1] != self._input_dim:
            raise ValueError(f"Data dimension ({data.shape[1]}) must match input dimension ({self._input_dim})")
        
        # Move data to device
        data = data.to(self.device)
        
        # Create optimizer for all parameters
        params = []
        for encoder in self.encoders:
            params.extend(list(encoder.parameters()))
        params.extend(list(self.decoder.parameters()))
        self.optimizer = torch.optim.Adam(params, lr=self.learning_rate)
        
        # Training loop
        for encoder in self.encoders:
            encoder.train()
        self.decoder.train()
        
        n_samples = data.shape[0]
        
        for epoch in range(self.train_iters):
            # Shuffle data
            indices = torch.randperm(n_samples)
            
            total_loss = 0
            total_recon_loss = 0
            total_kl_loss = 0
            
            # Mini-batch training
            for start_idx in range(0, n_samples, self.batch_size):
                end_idx = min(start_idx + self.batch_size, n_samples)
                batch_indices = indices[start_idx:end_idx]
                batch_data = data[batch_indices]
                
                # Forward pass through all encoders
                mus = []
                log_vars = []
                zs = []
                
                for encoder in self.encoders:
                    mu, log_var = encoder(batch_data)
                    z = self.reparameterize(mu, log_var)
                    mus.append(mu)
                    log_vars.append(log_var)
                    zs.append(z)
                
                # Combine latent representations (mean of means)
                mean_z = torch.stack(zs).mean(dim=0)
                
                # Decode using only the mean representation
                recon_batch = self.decoder(mean_z)
                
                # Compute loss
                recon_loss = F.mse_loss(recon_batch, batch_data, reduction='sum')
                
                # Sum KL divergence from all encoders
                kl_loss = 0
                for mu, log_var in zip(mus, log_vars):
                    kl_loss += -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
                
                loss = recon_loss + self.beta * kl_loss
                
                # Backward pass and optimize
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item()
                total_recon_loss += recon_loss.item()
                total_kl_loss += kl_loss.item()
            
            # Print progress
            if (epoch + 1) % 100 == 0:
                avg_loss = total_loss / n_samples
                avg_recon_loss = total_recon_loss / n_samples
                avg_kl_loss = total_kl_loss / n_samples
                print(f"Epoch {epoch+1}/{self.train_iters}, Loss: {avg_loss:.4f}, "
                      f"Recon Loss: {avg_recon_loss:.4f}, KL Loss: {avg_kl_loss:.4f}")
        
        # Set to evaluation mode
        for encoder in self.encoders:
            encoder.eval()
        self.decoder.eval()
        self.is_fitted = True
    
    def reparameterize(self, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        """
        Reparameterization trick to sample from N(mu, var) from N(0,1).
        
        Args:
            mu (torch.Tensor): Mean of the latent Gaussian
            log_var (torch.Tensor): Log variance of the latent Gaussian
            
        Returns:
            torch.Tensor: Sampled latent vector
        """
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Transform input data into the encoded space using the ensemble VAE encoder.
        
        Args:
            x (torch.Tensor): Input data of shape (n, input_dim)
            
        Returns:
            torch.Tensor: Encoded data of shape (n, encoded_dim)
        """
        if not self.is_fitted:
            raise RuntimeError("Ensemble VAE has not been fit. Call fit() first.")
        
        # Move to device and set to eval mode
        x = x.to(self.device)
        for encoder in self.encoders:
            encoder.eval()
        
        # Encode without gradients
        with torch.no_grad():
            # Get means and log variances from all encoders
            mus = []
            log_vars = []
            for encoder in self.encoders:
                mu, log_var = encoder(x)
                mus.append(mu)
                log_vars.append(log_var)
            
            # Calculate mean of means for first n-1 dimensions
            mean_mu = torch.stack(mus).mean(dim=0)
            
            # Calculate information radius for last dimension
            info_radius = self.calculate_information_radius(mus, log_vars)
            
            # Combine outputs
            return torch.cat([mean_mu, info_radius.unsqueeze(1)], dim=1)
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        Transform data from encoded space back to original space.
        Uses only the first n-1 dimensions for decoding.
        
        Args:
            z (torch.Tensor): Encoded data of shape (n, encoded_dim)
            
        Returns:
            torch.Tensor: Decoded data of shape (n, input_dim)
        """
        if not self.is_fitted:
            raise RuntimeError("Ensemble VAE has not been fit. Call fit() first.")
        
        # Move to device and set to eval mode
        z = z.to(self.device)
        self.decoder.eval()
        
        # Use only the first n-1 dimensions for decoding
        z_decode = z[:, :-1]
        
        # Decode without gradients
        with torch.no_grad():
            return self.decoder(z_decode)
    
    @property
    def input_dim(self) -> int:
        """Dimension of input space."""
        return self._input_dim
    
    @property
    def encoded_dim(self) -> int:
        """Dimension of encoded space."""
        return self._encoded_dim


class PairedEnsembleVAEEncoder(Encoder):
    """
    An ensemble of VAE encoder-decoder pairs that explicitly trains on both reconstruction error
    and information radius between the encodings of different encoders.
    """
    
    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        n_pairs: int = 3,
        hidden_dims: List[int] = [128, 64],
        beta: float = 1.0,
        gamma: float = 0.1,  # Weight for information radius loss
        train_iters: int = 1000,
        batch_size: int = 64,
        learning_rate: float = 1e-3,
        device: str = 'cpu'
    ):
        """
        Initialize the paired ensemble VAE encoder.
        
        Args:
            input_dim (int): Dimension of the input space
            latent_dim (int): Dimension of the latent space for each encoder
            n_pairs (int): Number of encoder-decoder pairs
            hidden_dims (List[int]): Hidden dimensions for encoders and decoders
            beta (float): Weight of the KL divergence term
            gamma (float): Weight of the information radius loss term
            train_iters (int): Number of training iterations
            batch_size (int): Batch size for training
            learning_rate (float): Learning rate for training
            device (str): Device to use for training
        """
        self._input_dim = input_dim
        self._encoded_dim = latent_dim  # No longer need +1 for information radius
        self.n_pairs = n_pairs
        self.device = device
        self.beta = beta
        self.gamma = gamma
        self.train_iters = train_iters
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        
        # Create encoder-decoder pairs
        self.pairs = nn.ModuleList([
            EncoderDecoderPair(input_dim, latent_dim, hidden_dims).to(device)
            for _ in range(n_pairs)
        ])
        
        # Initialize optimizer
        self.optimizer = None
        
        # Training state
        self.is_fitted = False
    
    def calculate_information_radius(self, mus: List[torch.Tensor], log_vars: List[torch.Tensor]) -> torch.Tensor:
        """Calculate the information radius between encoder distributions."""
        n = len(mus)
        
        # Calculate average distribution parameters
        mean_mu = torch.stack(mus).mean(dim=0)
        mean_var = torch.stack([torch.exp(log_var) for log_var in log_vars]).mean(dim=0)
        mean_log_var = torch.log(mean_var)
        
        # Calculate KL divergence from each distribution to the average
        kl_divs = []
        for i in range(n):
            kl_div = 0.5 * (
                torch.exp(log_vars[i]) / mean_var
                + (mus[i] - mean_mu).pow(2) / mean_var
                - 1
                + mean_log_var
                - log_vars[i]
            ).sum(dim=1)
            kl_divs.append(kl_div)
        
        # Information radius is the average KL divergence
        info_radius = torch.stack(kl_divs).mean(dim=0)
        
        return info_radius
    
    def fit(self, data: torch.Tensor) -> None:
        """
        Fit the paired ensemble VAE on the provided data.
        
        Args:
            data (torch.Tensor): Data to fit VAE on, shape (n, input_dim)
        """
        if data.shape[1] != self._input_dim:
            raise ValueError(f"Data dimension ({data.shape[1]}) must match input dimension ({self._input_dim})")
        
        # Move data to device
        data = data.to(self.device)
        
        # Create optimizer for all parameters
        self.optimizer = torch.optim.Adam(self.pairs.parameters(), lr=self.learning_rate)
        
        # Training loop
        self.pairs.train()
        n_samples = data.shape[0]
        
        for epoch in range(self.train_iters):
            # Shuffle data
            indices = torch.randperm(n_samples)
            
            total_loss = 0
            total_recon_loss = 0
            total_kl_loss = 0
            total_info_radius_loss = 0
            
            # Mini-batch training
            for start_idx in range(0, n_samples, self.batch_size):
                end_idx = min(start_idx + self.batch_size, n_samples)
                batch_indices = indices[start_idx:end_idx]
                batch_data = data[batch_indices]
                
                # Forward pass through all pairs
                recons = []
                mus = []
                log_vars = []
                
                for pair in self.pairs:
                    recon, mu, log_var = pair(batch_data)
                    recons.append(recon)
                    mus.append(mu)
                    log_vars.append(log_var)
                
                # Calculate losses
                # 1. Reconstruction loss (average across pairs)
                recon_loss = sum(F.mse_loss(recon, batch_data, reduction='sum')
                               for recon in recons) / len(recons)
                
                # 2. KL divergence loss (sum across pairs)
                kl_loss = 0
                for mu, log_var in zip(mus, log_vars):
                    kl_loss += -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
                
                # 3. Information radius loss
                info_radius = self.calculate_information_radius(mus, log_vars)
                info_radius_loss = info_radius.mean()
                
                # Total loss
                loss = recon_loss + self.beta * kl_loss + self.gamma * info_radius_loss
                
                # Backward pass and optimize
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item()
                total_recon_loss += recon_loss.item()
                total_kl_loss += kl_loss.item()
                total_info_radius_loss += info_radius_loss.item()
            
            # Print progress
            if (epoch + 1) % 100 == 0:
                avg_loss = total_loss / n_samples
                avg_recon_loss = total_recon_loss / n_samples
                avg_kl_loss = total_kl_loss / n_samples
                avg_info_radius_loss = total_info_radius_loss / n_samples
                print(f"Epoch {epoch+1}/{self.train_iters}, Loss: {avg_loss:.4f}, "
                      f"Recon Loss: {avg_recon_loss:.4f}, KL Loss: {avg_kl_loss:.4f}, "
                      f"Info Radius Loss: {avg_info_radius_loss:.4f}")
        
        # Set to evaluation mode
        self.pairs.eval()
        self.is_fitted = True
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Transform input data into the encoded space.
        Returns mean encoding across all pairs.
        """
        if not self.is_fitted:
            raise RuntimeError("Paired ensemble VAE has not been fit. Call fit() first.")
        
        # Move to device and set to eval mode
        x = x.to(self.device)
        self.pairs.eval()
        
        # Encode without gradients
        with torch.no_grad():
            # Get means from all pairs
            mus = []
            for pair in self.pairs:
                mu, _ = pair.encode(x)
                mus.append(mu)
            
            # Calculate mean encoding (across pairs)
            mean_encoding = torch.stack(mus).mean(dim=0)
            
            return mean_encoding
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        Transform data from encoded space back to original space.
        Uses average of all decoders.
        """
        if not self.is_fitted:
            raise RuntimeError("Paired ensemble VAE has not been fit. Call fit() first.")
        
        # Move to device and set to eval mode
        z = z.to(self.device)
        self.pairs.eval()
        
        # Decode without gradients using all pairs and average
        with torch.no_grad():
            reconstructions = []
            for pair in self.pairs:
                recon = pair.decode(z)
                reconstructions.append(recon)
            
            # Average reconstructions
            return torch.stack(reconstructions).mean(dim=0)
    
    def decoder_variance(self, z: torch.Tensor) -> torch.Tensor:
        """
        Calculate the variance between different decoders for a given latent point.
        
        Args:
            z (torch.Tensor): Encoded data of shape (n, encoded_dim)
            
        Returns:
            torch.Tensor: Variance of shape (n,) representing uncertainty in decoding
        """
        if not self.is_fitted:
            raise RuntimeError("Paired ensemble VAE has not been fit. Call fit() first.")
        
        # Move to device and set to eval mode
        z = z.to(self.device)

        # Get reconstructions from all pairs
        reconstructions = []
        for pair in self.pairs:
            recon = pair.decode(z)
            # Guard against NaN or Infinity
            if torch.isnan(recon).any() or torch.isinf(recon).any():
                # Replace with zeros for this reconstruction
                print(f"Warning: NaN or Inf detected in decoder output for some pairs")
                recon = torch.zeros_like(recon)
            reconstructions.append(recon)
        
        # Stack reconstructions [n_pairs, batch_size, input_dim]
        stacked_recons = torch.stack(reconstructions)
        
        # Calculate variance across pairs for each point
        # First compute variance along pair dimension for each feature
        feature_variance = torch.var(stacked_recons, dim=0)  # [batch_size, input_dim]
        
        # Then sum variance across features and take sqrt for a single scalar per point
        # This gives a measure of total uncertainty for each point
        total_variance = torch.sqrt(torch.sum(feature_variance, dim=1) + 1e-10)  # [batch_size]
        
        # Handle potential NaN or Inf in the variance
        if torch.isnan(total_variance).any() or torch.isinf(total_variance).any():
            # Create a mask for valid values
            valid_mask = ~(torch.isnan(total_variance) | torch.isinf(total_variance))
            
            if valid_mask.sum() > 0:
                # Replace invalid values with the mean of valid values
                valid_mean = total_variance[valid_mask].mean()
                total_variance = torch.where(valid_mask, total_variance, valid_mean)
            else:
                # If all values are invalid, use a default small variance
                print(f"Warning: All variance values are invalid, using default variance")
                total_variance = torch.ones_like(total_variance) * 0.1
        
        # Scale variance to be in a reasonable range for the acquisition function
        # Based on our analysis, we now know what a reasonable range is
        min_variance = 0.05  # Minimum variance to ensure exploration
        max_variance = 2.0   # Cap maximum variance to prevent excessive exploration
        
        # Clip variance to reasonable range
        total_variance = torch.clamp(total_variance, min=min_variance, max=max_variance)
        
        return total_variance
    
    @property
    def input_dim(self) -> int:
        """Dimension of input space."""
        return self._input_dim
    
    @property
    def encoded_dim(self) -> int:
        """Dimension of encoded space."""
        return self._encoded_dim 

class ConvVAEEncoderModule(nn.Module):
    """
    Convolutional encoder part of the VAE for image data.
    Designed for MNIST images (1x28x28).
    """
    def __init__(self, in_channels: int, latent_dim: int, hidden_dims: List[int] = None):
        super().__init__()
        
        if hidden_dims is None:
            hidden_dims = [32, 64, 128, 256]
            
        # Build encoder using convolutional layers
        modules = []
        
        # Input channels -> Hidden dimension layers
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(
                        in_channels, h_dim,
                        kernel_size=3, stride=2, padding=1
                    ),
                    nn.BatchNorm2d(h_dim),
                    nn.LeakyReLU()
                )
            )
            in_channels = h_dim
        
        self.encoder = nn.Sequential(*modules)
        
        # Calculate feature size after convolutions
        # For a 28x28 input with 4 layers of stride 2 convs: 28 -> 14 -> 7 -> 4 -> 2
        # So final feature map size is 2x2
        self.feature_size = 2 * 2 * hidden_dims[-1]
        
        # Flatten then project to latent space
        self.flatten = nn.Flatten()
        self.fc_mu = nn.Linear(self.feature_size, latent_dim)
        self.fc_var = nn.Linear(self.feature_size, latent_dim)
        
    def forward(self, x):
        # Reshape if not already in image format (B, C, H, W)
        if len(x.shape) == 2:  # (B, 784)
            x = x.view(-1, 1, 28, 28)
            
        # Pass through convolutional encoder layers
        features = self.encoder(x)
        flattened = self.flatten(features)
        
        # Get mean and log variance
        mu = self.fc_mu(flattened)
        log_var = self.fc_var(flattened)
        
        return mu, log_var


class ConvVAEDecoderModule(nn.Module):
    """
    Convolutional decoder part of the VAE for image data.
    Designed to reconstruct MNIST images (1x28x28).
    """
    def __init__(self, latent_dim: int, out_channels: int = 1, hidden_dims: List[int] = None):
        super().__init__()
        
        if hidden_dims is None:
            hidden_dims = [256, 128, 64, 32]
        
        # Save hidden dimensions for later use
        self.hidden_dims = hidden_dims.copy()  # Make a copy to avoid modifying the original
        
        # Initial fully connected layer from latent space to spatial features
        # Reverse hidden dimensions for decoder first
        self.hidden_dims = self.hidden_dims[::-1]
        
        # Start with a 2x2 spatial dimension (matching the encoder's output)
        self.initial_size = 2
        self.initial_channels = self.hidden_dims[0]
        self.feature_size = self.initial_size * self.initial_size * self.initial_channels
        
        # Linear layer to go from latent space to initial feature map
        self.decoder_input = nn.Linear(latent_dim, self.feature_size)
        
        # Build decoder using transposed convolutions
        modules = []
        
        # Using calculated dimensions to reach 28x28 output
        # First layer: 2x2 -> 7x7
        modules.append(
            nn.Sequential(
                nn.ConvTranspose2d(
                    self.hidden_dims[0], self.hidden_dims[1],
                    kernel_size=3, stride=2, padding=1, output_padding=1
                ),
                nn.BatchNorm2d(self.hidden_dims[1]),
                nn.LeakyReLU()
            )
        )
        
        # Second layer: 7x7 -> 14x14
        modules.append(
            nn.Sequential(
                nn.ConvTranspose2d(
                    self.hidden_dims[1], self.hidden_dims[2],
                    kernel_size=3, stride=2, padding=1, output_padding=1
                ),
                nn.BatchNorm2d(self.hidden_dims[2]),
                nn.LeakyReLU()
            )
        )
        
        # Final layer: 14x14 -> 28x28
        modules.append(
            nn.Sequential(
                nn.ConvTranspose2d(
                    self.hidden_dims[2], out_channels,
                    kernel_size=3, stride=2, padding=1, output_padding=1
                ),
                nn.Sigmoid()  # Use sigmoid for pixel values in [0, 1]
            )
        )
        
        self.decoder = nn.Sequential(*modules)
        self.out_channels = out_channels
    
    def forward(self, z):
        batch_size = z.shape[0]
        
        # Project from latent space to initial spatial features
        result = self.decoder_input(z)
        
        # Check if the feature size is correct before reshaping
        expected_size = batch_size * self.feature_size
        if result.numel() != expected_size:
            print(f"Warning: Tensor size mismatch. Expected {expected_size} elements, got {result.numel()}")
            # Adjust feature size if needed for this specific input
            if result.numel() % batch_size == 0:
                actual_feature_size = result.numel() // batch_size
                # Calculate new initial size
                actual_channel_size = self.initial_channels
                actual_spatial_size = int(np.sqrt(actual_feature_size / actual_channel_size))
                print(f"Adjusting initial size from {self.initial_size} to {actual_spatial_size}")
                result = result.view(batch_size, self.initial_channels, actual_spatial_size, actual_spatial_size)
            else:
                # Emergency fallback: give up on keeping batch size and just reshape
                print("Warning: Cannot maintain batch size. Using alternative reshaping.")
                result = result.view(batch_size, self.initial_channels, self.initial_size, self.initial_size)
        else:
            # If size is as expected, reshape normally
            result = result.view(batch_size, self.initial_channels, self.initial_size, self.initial_size)
        
        # Pass through decoder layers
        result = self.decoder(result)
        
        # Verify batch size is preserved
        if result.shape[0] != batch_size:
            print(f"Warning: Batch size changed during decoding from {batch_size} to {result.shape[0]}")
            # Use only the needed number of examples
            if result.shape[0] > batch_size:
                result = result[:batch_size]
            else:
                # If we have fewer samples than expected, pad with zeros
                pad_size = batch_size - result.shape[0]
                zeros = torch.zeros((pad_size, result.shape[1], result.shape[2], result.shape[3]), 
                                    device=result.device)
                result = torch.cat([result, zeros], dim=0)
        
        # Ensure dimensions are exactly 28x28
        if result.shape[2] != 28 or result.shape[3] != 28:
            print(f"Warning: Output spatial size {result.shape[2]}x{result.shape[3]} != 28x28. Resizing.")
            result = F.interpolate(result, size=(28, 28), mode='bilinear', align_corners=False)
        
        # Flatten to match expected output format (B, C*H*W)
        result = result.view(batch_size, -1)
        
        return result


class ConvVAEEncoder(Encoder):
    """
    A Convolutional Variational Autoencoder (ConvVAE) encoder for image data.
    Designed for MNIST images (28x28 pixels).
    """
    
    def __init__(
        self, 
        input_channels: int = 1,
        img_size: int = 28,
        latent_dim: int = 10, 
        hidden_dims: List[int] = None,
        beta: float = 1.0,
        train_iters: int = 1000,
        batch_size: int = 64,
        learning_rate: float = 1e-3,
        device: str = 'cpu'
    ):
        """
        Initialize the ConvVAE encoder.
        
        Args:
            input_channels (int): Number of input channels (1 for grayscale, 3 for RGB)
            img_size (int): Size of the square input images
            latent_dim (int): Dimension of the latent (encoded) space
            hidden_dims (List[int]): List of hidden dimensions for encoder and decoder
            beta (float): Weight of the KL divergence term (beta-VAE)
            train_iters (int): Number of training iterations
            batch_size (int): Batch size for training
            learning_rate (float): Learning rate for training
            device (str): Device to use for training ('cpu' or 'cuda')
        """
        self._input_dim = input_channels * img_size * img_size  # For flattened images
        self._encoded_dim = latent_dim
        self.input_channels = input_channels
        self.img_size = img_size
        self.hidden_dims = [32, 64, 128, 256] if hidden_dims is None else hidden_dims
        self.beta = beta
        self.train_iters = train_iters
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.device = device
        
        # Initialize encoder and decoder
        self.encoder_model = ConvVAEEncoderModule(
            in_channels=input_channels, 
            latent_dim=latent_dim, 
            hidden_dims=self.hidden_dims
        ).to(device)
        
        self.decoder_model = ConvVAEDecoderModule(
            latent_dim=latent_dim, 
            out_channels=input_channels, 
            hidden_dims=self.hidden_dims[::-1]  # Reverse for decoder
        ).to(device)
        
        # Initialize optimizer
        self.optimizer = None
        
        # Training state
        self.is_fitted = False

    def eval(self):
        self.encoder_model.eval()
        self.decoder_model.eval()

    def train(self):
        self.encoder_model.train()
        self.decoder_model.train()
    
    def reparameterize(self, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        """
        Reparameterization trick to sample from N(mu, var) from N(0,1).
        
        Args:
            mu (torch.Tensor): Mean of the latent Gaussian
            log_var (torch.Tensor): Log variance of the latent Gaussian
            
        Returns:
            torch.Tensor: Sampled latent vector
        """
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def fit(self, data: torch.Tensor) -> None:
        """
        Fit the ConvVAE encoder on the provided data.
        
        Args:
            data (torch.Tensor): Data to fit VAE on, shape (n, input_dim) or (n, C, H, W)
        """
        # Check if the data matches the expected dimensions
        if len(data.shape) == 2:  # Flattened images
            if data.shape[1] != self._input_dim:
                raise ValueError(f"Data dimension ({data.shape[1]}) must match input dimension ({self._input_dim})")
            # Reshape to (B, C, H, W) format
            data_images = data.view(-1, self.input_channels, self.img_size, self.img_size)
        elif len(data.shape) == 4:  # Already in (B, C, H, W) format
            if data.shape[1] != self.input_channels or data.shape[2] != self.img_size or data.shape[3] != self.img_size:
                raise ValueError(f"Data dimensions ({data.shape[1:4]}) must match expected "
                               f"([{self.input_channels}, {self.img_size}, {self.img_size}])")
            data_images = data
            # Also create flattened version for reconstruction loss
            data = data.view(data.shape[0], -1)
        else:
            raise ValueError(f"Unsupported data shape: {data.shape}")
        
        # Move data to device
        data = data.to(self.device)
        data_images = data_images.to(self.device)
        
        # Create optimizer
        params = list(self.encoder_model.parameters()) + list(self.decoder_model.parameters())
        self.optimizer = torch.optim.Adam(params, lr=self.learning_rate)
        
        # Training loop
        self.encoder_model.train()
        self.decoder_model.train()
        
        n_samples = data.shape[0]
        
        for epoch in range(self.train_iters):
            # Shuffle data
            indices = torch.randperm(n_samples)
            
            total_loss = 0
            total_recon_loss = 0
            total_kl_loss = 0
            
            # Mini-batch training
            for start_idx in range(0, n_samples, self.batch_size):
                end_idx = min(start_idx + self.batch_size, n_samples)
                batch_indices = indices[start_idx:end_idx]
                batch_size = len(batch_indices)
                
                batch_data = data[batch_indices]
                batch_images = data_images[batch_indices]
                
                # Forward pass
                mu, log_var = self.encoder_model(batch_images)
                z = self.reparameterize(mu, log_var)
                recon_batch = self.decoder_model(z)
                
                # Compute loss
                recon_loss = F.mse_loss(recon_batch, batch_data, reduction='sum')
                kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
                loss = recon_loss + self.beta * kl_loss
                
                # Backward pass and optimize
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item()
                total_recon_loss += recon_loss.item()
                total_kl_loss += kl_loss.item()
            
            # Print progress
            if (epoch + 1) % 100 == 0:
                avg_loss = total_loss / n_samples
                avg_recon_loss = total_recon_loss / n_samples
                avg_kl_loss = total_kl_loss / n_samples
                print(f"Epoch {epoch+1}/{self.train_iters}, Loss: {avg_loss:.4f}, "
                      f"Recon Loss: {avg_recon_loss:.4f}, KL Loss: {avg_kl_loss:.4f}")
        
        # Set to evaluation mode
        self.encoder_model.eval()
        self.decoder_model.eval()
        self.is_fitted = True
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Transform input data into the encoded space using the ConvVAE encoder.
        
        Args:
            x (torch.Tensor): Input data, can be:
                - (n, input_dim) for flattened images
                - (n, C, H, W) for batched images
            
        Returns:
            torch.Tensor: Encoded data of shape (n, encoded_dim)
        """
        if not self.is_fitted:
            raise RuntimeError("ConvVAE has not been fit. Call fit() first.")
        
        # Format the input correctly
        if len(x.shape) == 2:  # Flattened images
            if x.shape[1] != self._input_dim:
                raise ValueError(f"Input dimension ({x.shape[1]}) must match expected ({self._input_dim})")
            x_images = x.view(-1, self.input_channels, self.img_size, self.img_size)
        elif len(x.shape) == 4:  # Already in (B, C, H, W) format
            if x.shape[1] != self.input_channels or x.shape[2] != self.img_size or x.shape[3] != self.img_size:
                raise ValueError(f"Input dimensions ({x.shape[1:4]}) must match expected "
                               f"([{self.input_channels}, {self.img_size}, {self.img_size}])")
            x_images = x
        else:
            raise ValueError(f"Unsupported input shape: {x.shape}")
        
        # Move to device and set to eval mode
        x_images = x_images.to(self.device)
        self.encoder_model.eval()
        
        # Encode without gradients
        with torch.no_grad():
            mu, log_var = self.encoder_model(x_images)
            
            # For Bayesian optimization, we use the mean of the latent distribution
            # rather than sampling, to ensure deterministic behavior
            return mu
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        Transform data from encoded space back to original space using the ConvVAE decoder.
        
        Args:
            z (torch.Tensor): Encoded data of shape (n, encoded_dim)
            
        Returns:
            torch.Tensor: Decoded data of shape (n, input_dim) (flattened images)
        """
        if not self.is_fitted:
            raise RuntimeError("ConvVAE has not been fit. Call fit() first.")
        
        # Move to device and set to eval mode
        z = z.to(self.device)
        self.decoder_model.eval()
        
        # Decode without gradients
        with torch.no_grad():
            return self.decoder_model(z)
    
    def reconstruction_loss(self, x: torch.Tensor, reduction: str = 'mean') -> torch.Tensor:
        """
        Calculate the reconstruction loss for a set of points.
        
        Args:
            x (torch.Tensor): Input data of shape (n, input_dim) or (n, C, H, W)
            reduction (str): Reduction method ('mean', 'sum', or 'none')
            
        Returns:
            torch.Tensor: Reconstruction loss
        """
        if not self.is_fitted:
            raise RuntimeError("ConvVAE has not been fit. Call fit() first.")
        
        # Format the input correctly
        if len(x.shape) == 2:  # Flattened images
            if x.shape[1] != self._input_dim:
                raise ValueError(f"Input dimension ({x.shape[1]}) must match expected ({self._input_dim})")
            x_images = x.view(-1, self.input_channels, self.img_size, self.img_size)
            x_flat = x
        elif len(x.shape) == 4:  # Already in (B, C, H, W) format
            if x.shape[1] != self.input_channels or x.shape[2] != self.img_size or x.shape[3] != self.img_size:
                raise ValueError(f"Input dimensions ({x.shape[1:4]}) must match expected "
                               f"([{self.input_channels}, {self.img_size}, {self.img_size}])")
            x_images = x
            x_flat = x.view(x.shape[0], -1)
        else:
            raise ValueError(f"Unsupported input shape: {x.shape}")
        
        # Move to device
        x_images = x_images.to(self.device)
        x_flat = x_flat.to(self.device)
        
        # Encode and decode
        with torch.no_grad():
            mu, _ = self.encoder_model(x_images)
            x_reconstructed = self.decoder_model(mu)
        
        # Calculate squared error
        squared_error = (x_flat - x_reconstructed).pow(2)
        
        # Apply reduction
        if reduction == 'mean':
            return squared_error.mean()
        elif reduction == 'sum':
            return squared_error.sum()
        elif reduction == 'none':
            return squared_error
        else:
            raise ValueError(f"Unknown reduction method: {reduction}")
    
    @property
    def input_dim(self) -> int:
        """Dimension of input space."""
        return self._input_dim
    
    @property
    def encoded_dim(self) -> int:
        """Dimension of encoded space."""
        return self._encoded_dim 

class DecoderEnsemble(Encoder):
    """
    A VAE-based encoder that uses a single encoder but multiple decoders.
    Provides decoder variance measurements for uncertainty quantification.
    """
    
    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        n_decoders: int = 3,
        hidden_dims: List[int] = [128, 64],
        beta: float = 1.0,
        train_iters: int = 1000,
        batch_size: int = 64,
        learning_rate: float = 1e-3,
        device: str = 'cpu'
    ):
        """
        Initialize the DecoderEnsemble VAE encoder.
        
        Args:
            input_dim (int): Dimension of the input space
            latent_dim (int): Dimension of the latent space
            n_decoders (int): Number of decoders in the ensemble
            hidden_dims (List[int]): Hidden dimensions for encoder and decoders
            beta (float): Weight of the KL divergence term
            train_iters (int): Number of training iterations
            batch_size (int): Batch size for training
            learning_rate (float): Learning rate for training
            device (str): Device to use for training
        """
        self._input_dim = input_dim
        self._encoded_dim = latent_dim
        self.n_decoders = n_decoders
        self.device = device
        self.beta = beta
        self.train_iters = train_iters
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.hidden_dims = hidden_dims
        
        # Create a shared encoder
        self.encoder = VAEEncoderModule(input_dim, latent_dim, hidden_dims).to(device)
        
        # Create multiple decoders
        self.decoders = nn.ModuleList([
            VAEDecoderModule(latent_dim, input_dim, hidden_dims).to(device)
            for _ in range(n_decoders)
        ])
        
        # Initialize optimizer
        self.optimizer = None
        
        # Training state
        self.is_fitted = False
        
    def eval(self):
        """Set all components to evaluation mode."""
        self.encoder.eval()
        for decoder in self.decoders:
            decoder.eval()
    
    def train(self):
        """Set all components to training mode."""
        self.encoder.train()
        for decoder in self.decoders:
            decoder.train()
    
    def reparameterize(self, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        """
        Reparameterization trick: sample from N(mu, var) using N(0,1).
        
        Args:
            mu: Mean tensor
            log_var: Log variance tensor
            
        Returns:
            Sampled latent vector
        """
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def fit(self, data: torch.Tensor) -> None:
        """
        Fit the DecoderEnsemble VAE on the provided data.
        
        Args:
            data (torch.Tensor): Data to fit VAE on, shape (n, input_dim)
        """
        if data.shape[1] != self._input_dim:
            raise ValueError(f"Data dimension ({data.shape[1]}) must match input dimension ({self._input_dim})")
        
        # Move data to device
        data = data.to(self.device)
        
        # Create optimizer for all parameters
        self.optimizer = torch.optim.Adam(
            list(self.encoder.parameters()) + 
            [p for decoder in self.decoders for p in decoder.parameters()],
            lr=self.learning_rate
        )
        
        # Training loop
        self.train()
        n_samples = data.shape[0]
        
        for epoch in range(self.train_iters):
            # Shuffle data
            indices = torch.randperm(n_samples)
            
            total_loss = 0
            total_recon_loss = 0
            total_kl_loss = 0
            
            # Mini-batch training
            for start_idx in range(0, n_samples, self.batch_size):
                end_idx = min(start_idx + self.batch_size, n_samples)
                batch_indices = indices[start_idx:end_idx]
                batch_data = data[batch_indices]
                
                # Forward pass through encoder
                mu, log_var = self.encoder(batch_data)
                
                # Reparameterize to get latent vector
                z = self.reparameterize(mu, log_var)
                
                # Forward pass through all decoders
                recons = []
                for decoder in self.decoders:
                    recon = decoder(z)
                    recons.append(recon)
                
                # Calculate losses
                # 1. Reconstruction loss (average across decoders)
                recon_loss = sum(F.mse_loss(recon, batch_data, reduction='sum')
                               for recon in recons) / len(recons)
                
                # 2. KL divergence loss
                kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
                
                # Total loss
                loss = recon_loss + self.beta * kl_loss
                
                # Backward pass and optimize
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item()
                total_recon_loss += recon_loss.item()
                total_kl_loss += kl_loss.item()
            
            # Print progress
            if (epoch + 1) % 100 == 0:
                avg_loss = total_loss / n_samples
                avg_recon_loss = total_recon_loss / n_samples
                avg_kl_loss = total_kl_loss / n_samples
                print(f"Epoch {epoch+1}/{self.train_iters}, Loss: {avg_loss:.4f}, "
                      f"Recon Loss: {avg_recon_loss:.4f}, KL Loss: {avg_kl_loss:.4f}")
        
        # Set to evaluation mode
        self.eval()
        self.is_fitted = True
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Transform input data into the encoded space.
        Returns mean of the latent distribution.
        
        Args:
            x (torch.Tensor): Input data of shape (n, input_dim)
            
        Returns:
            torch.Tensor: Encoded data of shape (n, encoded_dim)
        """
        if not self.is_fitted:
            raise RuntimeError("DecoderEnsemble VAE has not been fit. Call fit() first.")
        
        # Move to device and set to eval mode
        x = x.to(self.device)
        self.eval()
        
        # Encode without gradients
        with torch.no_grad():
            mu, _ = self.encoder(x)
            return mu
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        Transform data from encoded space back to original space.
        Uses average of all decoders.
        
        Args:
            z (torch.Tensor): Encoded data of shape (n, encoded_dim)
            
        Returns:
            torch.Tensor: Decoded data of shape (n, input_dim)
        """
        if not self.is_fitted:
            raise RuntimeError("DecoderEnsemble VAE has not been fit. Call fit() first.")
        
        # Move to device and set to eval mode
        z = z.to(self.device)
        self.eval()
        
        # Decode without gradients using all decoders and average
        with torch.no_grad():
            reconstructions = []
            for decoder in self.decoders:
                recon = decoder(z)
                reconstructions.append(recon)
            
            # Average reconstructions
            return torch.stack(reconstructions).mean(dim=0)
    
    def decoder_variance(self, z: torch.Tensor) -> torch.Tensor:
        """
        Calculate the variance between different decoders for a given latent point.
        
        Args:
            z (torch.Tensor): Encoded data of shape (n, encoded_dim)
            
        Returns:
            torch.Tensor: Variance of shape (n,) representing uncertainty in decoding
        """
        if not self.is_fitted:
            raise RuntimeError("DecoderEnsemble VAE has not been fit. Call fit() first.")
        
        # Move to device and set to eval mode
        z = z.to(self.device)
        self.eval()

        # Get reconstructions from all decoders
        reconstructions = []
        for decoder in self.decoders:
            recon = decoder(z)
            # Guard against NaN or Infinity
            if torch.isnan(recon).any() or torch.isinf(recon).any():
                # Replace with zeros for this reconstruction
                print(f"Warning: NaN or Inf detected in decoder output")
                recon = torch.zeros_like(recon)
            reconstructions.append(recon)
        
        # Stack reconstructions [n_decoders, batch_size, input_dim]
        stacked_recons = torch.stack(reconstructions)
        
        # Calculate variance across decoders for each point
        # First compute variance along decoder dimension for each feature
        feature_variance = torch.var(stacked_recons, dim=0)  # [batch_size, input_dim]
        
        # Then sum variance across features and take sqrt for a single scalar per point
        # This gives a measure of total uncertainty for each point
        total_variance = torch.sqrt(torch.sum(feature_variance, dim=1) + 1e-10)  # [batch_size]
        
        # Handle potential NaN or Inf in the variance
        if torch.isnan(total_variance).any() or torch.isinf(total_variance).any():
            # Create a mask for valid values
            valid_mask = ~(torch.isnan(total_variance) | torch.isinf(total_variance))
            
            if valid_mask.sum() > 0:
                # Replace invalid values with the mean of valid values
                valid_mean = total_variance[valid_mask].mean()
                total_variance = torch.where(valid_mask, total_variance, valid_mean)
                total_variance = torch.clamp(total_variance, min=0.01, max=10)
            else:
                # If all values are invalid, use a default small variance
                print(f"Warning: All variance values are invalid, using default variance")
                total_variance = torch.ones_like(total_variance) * 0.1
        
        # # Scale variance to be in a reasonable range for the acquisition function
        # min_variance = 0.05  # Minimum variance to ensure exploration
        # max_variance = 2.0   # Cap maximum variance to prevent excessive exploration
        
        # # Clip variance to reasonable range
        # total_variance = torch.clamp(total_variance, min=min_variance, max=max_variance)
        
        return total_variance
    
    def reconstruction_loss(self, x: torch.Tensor, reduction: str = 'mean') -> torch.Tensor:
        """
        Calculate the reconstruction loss for a set of points.
        
        Args:
            x (torch.Tensor): Input data of shape (n, input_dim)
            reduction (str): Reduction method ('mean', 'sum', or 'none')
            
        Returns:
            torch.Tensor: Reconstruction loss
        """
        if not self.is_fitted:
            raise RuntimeError("DecoderEnsemble VAE has not been fit. Call fit() first.")
        
        # Move to device
        x = x.to(self.device)
        
        # Encode and decode
        with torch.no_grad():
            mu, _ = self.encoder(x)
            # Get average reconstruction across all decoders
            recons = []
            for decoder in self.decoders:
                recon = decoder(mu)
                recons.append(recon)
            
            x_reconstructed = torch.stack(recons).mean(dim=0)
        
        # Calculate squared error
        squared_error = (x - x_reconstructed).pow(2)
        
        # Apply reduction
        if reduction == 'mean':
            return squared_error.mean()
        elif reduction == 'sum':
            return squared_error.sum()
        elif reduction == 'none':
            return squared_error
        else:
            raise ValueError(f"Unknown reduction method: {reduction}")
    
    @property
    def input_dim(self) -> int:
        """Dimension of input space."""
        return self._input_dim
    
    @property
    def encoded_dim(self) -> int:
        """Dimension of encoded space."""
        return self._encoded_dim

class ConvDecoderEnsemble(Encoder):
    """
    A convolutional VAE-based encoder that uses a single encoder but multiple decoders.
    Designed for image data (like MNIST). Provides decoder variance measurements for uncertainty quantification.
    """
    
    def __init__(
        self,
        input_channels: int = 1,
        img_size: int = 28,
        latent_dim: int = 10,
        n_decoders: int = 3,
        hidden_dims: List[int] = None,
        beta: float = 1.0,
        train_iters: int = 1000,
        batch_size: int = 64,
        learning_rate: float = 1e-3,
        device: str = 'cpu'
    ):
        """
        Initialize the Convolutional DecoderEnsemble VAE encoder.
        
        Args:
            input_channels (int): Number of input channels (e.g., 1 for grayscale, 3 for RGB)
            img_size (int): Image size (assumed to be square)
            latent_dim (int): Dimension of the latent space
            n_decoders (int): Number of decoders in the ensemble
            hidden_dims (List[int]): Hidden dimensions for convolutional layers
            beta (float): Weight of the KL divergence term
            train_iters (int): Number of training iterations
            batch_size (int): Batch size for training
            learning_rate (float): Learning rate for training
            device (str): Device to use for training
        """
        self.input_channels = input_channels
        self.img_size = img_size
        self._input_dim = input_channels * img_size * img_size
        self._encoded_dim = latent_dim
        self.n_decoders = n_decoders
        self.device = device
        self.beta = beta
        self.train_iters = train_iters
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        
        # Set default hidden dimensions if not provided
        if hidden_dims is None:
            self.hidden_dims = [32, 64, 128]
        else:
            self.hidden_dims = hidden_dims
        
        # Calculate the size of the flattened features after conv layers
        # For a standard 28x28 image with strides of 2, we get 4x4 feature maps with the last channel size
        self.feature_size = self.hidden_dims[-1] * (img_size // (2 ** len(self.hidden_dims))) ** 2
        
        # Create a shared encoder
        self._create_encoder()
        
        # Create multiple decoders
        self.decoders = nn.ModuleList([
            self._create_decoder() for _ in range(n_decoders)
        ])
        
        # Initialize optimizer
        self.optimizer = None
        
        # Training state
        self.is_fitted = False
    
    def _create_encoder(self):
        """Create the convolutional encoder network."""
        modules = []
        in_channels = self.input_channels
        
        # Create convolutional layers
        for h_dim in self.hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, h_dim, kernel_size=3, stride=2, padding=1),
                    nn.BatchNorm2d(h_dim),
                    nn.LeakyReLU()
                )
            )
            in_channels = h_dim
        
        self.encoder_conv = nn.Sequential(*modules)
        
        # Create mean and log variance layers
        self.fc_mu = nn.Linear(self.feature_size, self._encoded_dim)
        self.fc_logvar = nn.Linear(self.feature_size, self._encoded_dim)
        
        # Move to device
        self.encoder_conv = self.encoder_conv.to(self.device)
        self.fc_mu = self.fc_mu.to(self.device)
        self.fc_logvar = self.fc_logvar.to(self.device)
    
    def _create_decoder(self):
        """Create a single convolutional decoder network."""
        modules = []
        
        # Initial linear layer to project from latent space
        self.decoder_input = nn.Linear(self._encoded_dim, self.feature_size).to(self.device)
        
        # Reversed hidden dimensions
        hidden_dims = self.hidden_dims[::-1]
        
        # Create transposed convolutional layers
        modules.append(
            nn.Sequential(
                nn.ConvTranspose2d(
                    hidden_dims[0], 
                    hidden_dims[1],
                    kernel_size=3, stride=2, padding=1, output_padding=1
                ),
                nn.BatchNorm2d(hidden_dims[1]),
                nn.LeakyReLU()
            )
        )
        
        # Add intermediate layers
        for i in range(1, len(hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(
                        hidden_dims[i], 
                        hidden_dims[i + 1],
                        kernel_size=3, stride=2, padding=1, output_padding=1
                    ),
                    nn.BatchNorm2d(hidden_dims[i + 1]),
                    nn.LeakyReLU()
                )
            )
        
        # Add final layer
        modules.append(
            nn.Sequential(
                nn.ConvTranspose2d(
                    hidden_dims[-1], 
                    self.input_channels,
                    kernel_size=3, stride=2, padding=1, output_padding=1
                ),
                nn.Sigmoid()  # Use sigmoid for pixel values in [0, 1]
            )
        )
        
        # Create sequential module
        decoder = nn.Sequential(*modules).to(self.device)
        return decoder
    
    def _encode_internal(self, x):
        """
        Internal encoding function that returns both mean and log variance.
        
        Args:
            x: Input tensor of shape [B, C, H, W]
            
        Returns:
            mu: Mean in latent space
            logvar: Log variance in latent space
        """
        # Apply convolutional layers
        x = self.encoder_conv(x)
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Get mean and log variance
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        
        return mu, logvar
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Transform input data into the encoded space.
        Returns mean of the latent distribution.
        
        Args:
            x (torch.Tensor): Input data of shape (n, input_channels, img_size, img_size) or (n, input_dim)
            
        Returns:
            torch.Tensor: Encoded data of shape (n, encoded_dim)
        """
        if not self.is_fitted:
            raise RuntimeError("ConvDecoderEnsemble VAE has not been fit. Call fit() first.")
        
        # Ensure input is properly shaped
        if x.dim() == 2:  # Flattened images
            if x.shape[1] != self._input_dim:
                raise ValueError(f"Input dimension ({x.shape[1]}) must match expected ({self._input_dim})")
            x_images = x.view(-1, self.input_channels, self.img_size, self.img_size)
        elif len(x.shape) == 4:  # Already in (B, C, H, W) format
            if x.shape[1] != self.input_channels or x.shape[2] != self.img_size or x.shape[3] != self.img_size:
                raise ValueError(f"Input dimensions ({x.shape[1:4]}) must match expected "
                               f"([{self.input_channels}, {self.img_size}, {self.img_size}])")
            x_images = x
        else:
            raise ValueError(f"Unsupported input shape: {x.shape}")
        
        # Move to device and set to eval mode
        x_images = x_images.to(self.device)
        self.eval()
        
        # Encode without gradients
        with torch.no_grad():
            mu, _ = self._encode_internal(x_images)
            return mu
    
    def reparameterize(self, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        """
        Reparameterization trick: sample from N(mu, var) using N(0,1).
        
        Args:
            mu: Mean tensor
            log_var: Log variance tensor
            
        Returns:
            Sampled latent vector
        """
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        Transform data from encoded space back to original space.
        Uses average of all decoders.
        
        Args:
            z (torch.Tensor): Encoded data of shape (n, encoded_dim)
            
        Returns:
            torch.Tensor: Decoded data of shape (n, input_dim)
        """
        if not self.is_fitted:
            raise RuntimeError("ConvDecoderEnsemble VAE has not been fit. Call fit() first.")
        
        # Move to device and set to eval mode
        z = z.to(self.device)
        self.eval()
        
        # Decode without gradients using all decoders and average
        with torch.no_grad():
            reconstructions = []
            for decoder in self.decoders:
                recon = self._decode_internal(z, decoder)
                # Flatten to [B, input_dim]
                recon_flat = recon.view(recon.size(0), -1)
                reconstructions.append(recon_flat)
            
            # Average reconstructions
            avg_recon = torch.stack(reconstructions).mean(dim=0)
            return avg_recon
    
    def _decode_internal(self, z, decoder):
        """
        Internal decoding function for a single decoder.
        
        Args:
            z: Latent vector of shape [B, latent_dim]
            decoder: Decoder module to use
            
        Returns:
            Reconstructed image of shape [B, C, H, W]
        """
        # Project and reshape to spatial feature map
        x = self.decoder_input(z)
        x = x.view(-1, self.hidden_dims[-1], self.img_size // (2 ** len(self.hidden_dims)), self.img_size // (2 ** len(self.hidden_dims)))
        
        # Apply decoder
        x = decoder(x)
        
        # Ensure output is the right size (in case of rounding issues with transposed convolutions)
        if x.size(-1) != self.img_size or x.size(-2) != self.img_size:
            x = F.interpolate(x, size=(self.img_size, self.img_size), mode='bilinear', align_corners=False)
        
        return x
    
    def decoder_variance(self, z: torch.Tensor) -> torch.Tensor:
        """
        Calculate per-point variance across decoders to quantify uncertainty.
        
        Args:
            z (torch.Tensor): Encoded data of shape (n, encoded_dim)
            
        Returns:
            torch.Tensor: Variance of shape (n,)
        """
        if not self.is_fitted:
            raise RuntimeError("ConvDecoderEnsemble VAE has not been fit. Call fit() first.")
        
        # Move to device and set to eval mode
        z = z.to(self.device)
        self.eval()
        
        # Get reconstructions from all decoders
        with torch.no_grad():
            reconstructions = []
            for decoder in self.decoders:
                recon = self._decode_internal(z, decoder)
                # Flatten to [B, input_dim]
                recon_flat = recon.view(recon.size(0), -1)
                reconstructions.append(recon_flat)
            
            # Stack reconstructions along a new dimension [n_decoders, n, input_dim]
            stacked = torch.stack(reconstructions)
            
            # Calculate variance across decoders for each point
            # Resulting shape: [n, input_dim]
            variance = torch.var(stacked, dim=0)
            
            # Average variance across input dimensions to get a scalar per point
            # Resulting shape: [n]
            avg_variance = torch.mean(variance, dim=1)
            
            return avg_variance
    
    def fit(self, data: torch.Tensor) -> None:
        """
        Fit the ConvDecoderEnsemble VAE on the provided data.
        
        Args:
            data (torch.Tensor): Data to fit VAE on, shape (n, C, H, W) or (n, input_dim)
        """
        # Ensure input is properly shaped
        if data.dim() == 2:
            if data.shape[1] != self._input_dim:
                raise ValueError(f"Data dimension ({data.shape[1]}) must match input dimension ({self._input_dim})")
            # Reshape to [B, C, H, W]
            data = data.view(-1, self.input_channels, self.img_size, self.img_size)
        elif data.dim() == 4:
            if data.shape[1] != self.input_channels or data.shape[2] != self.img_size or data.shape[3] != self.img_size:
                raise ValueError(f"Expected data shape (n, {self.input_channels}, {self.img_size}, {self.img_size}), "
                                f"got {data.shape}")
        else:
            raise ValueError(f"Unsupported data shape: {data.shape}")
        
        # Move data to device
        data = data.to(self.device)
        
        # Create optimizer for all parameters
        parameters = list(self.encoder_conv.parameters()) + list(self.fc_mu.parameters()) + \
                     list(self.fc_logvar.parameters()) + list(self.decoder_input.parameters())
        
        for decoder in self.decoders:
            parameters.extend(list(decoder.parameters()))
        
        self.optimizer = torch.optim.Adam(parameters, lr=self.learning_rate)
        
        # Training loop
        self.train()
        n_samples = data.shape[0]
        
        for epoch in range(self.train_iters):
            # Shuffle data
            indices = torch.randperm(n_samples)
            
            total_loss = 0
            total_recon_loss = 0
            total_kl_loss = 0
            
            # Mini-batch training
            for start_idx in range(0, n_samples, self.batch_size):
                end_idx = min(start_idx + self.batch_size, n_samples)
                batch_indices = indices[start_idx:end_idx]
                batch_data = data[batch_indices]
                
                # Flatten batch for reconstruction loss
                batch_flat = batch_data.view(batch_data.size(0), -1)
                
                # Forward pass through encoder
                mu, log_var = self._encode_internal(batch_data)
                
                # Reparameterize to get latent vector
                z = self.reparameterize(mu, log_var)
                
                # Forward pass through all decoders
                recons = []
                for decoder in self.decoders:
                    recon = self._decode_internal(z, decoder)
                    # Flatten for loss calculation
                    recon_flat = recon.view(recon.size(0), -1)
                    recons.append(recon_flat)
                
                # Calculate losses
                # 1. Reconstruction loss (average across decoders)
                recon_loss = sum(F.mse_loss(recon, batch_flat, reduction='sum')
                                for recon in recons) / len(recons)
                
                # 2. KL divergence loss
                kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
                
                # Total loss
                loss = recon_loss + self.beta * kl_loss
                
                # Backward pass and optimize
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item()
                total_recon_loss += recon_loss.item()
                total_kl_loss += kl_loss.item()
            
            # Print progress
            if (epoch + 1) % 10 == 0 or epoch == 0:
                avg_loss = total_loss / n_samples
                avg_recon_loss = total_recon_loss / n_samples
                avg_kl_loss = total_kl_loss / n_samples
                print(f"Epoch {epoch+1}/{self.train_iters}, Loss: {avg_loss:.4f}, "
                      f"Recon Loss: {avg_recon_loss:.4f}, KL Loss: {avg_kl_loss:.4f}")
        
        # Set to evaluation mode
        self.eval()
        self.is_fitted = True
    
    def eval(self):
        """Set all components to evaluation mode."""
        self.encoder_conv.eval()
        self.fc_mu.eval()
        self.fc_logvar.eval()
        self.decoder_input.eval()
        for decoder in self.decoders:
            decoder.eval()
    
    def train(self):
        """Set all components to training mode."""
        self.encoder_conv.train()
        self.fc_mu.train()
        self.fc_logvar.train()
        self.decoder_input.train()
        for decoder in self.decoders:
            decoder.train()
    
    def reconstruction_loss(self, x: torch.Tensor, reduction: str = 'mean') -> torch.Tensor:
        """
        Calculate reconstruction loss for given data points.
        
        Args:
            x (torch.Tensor): Input data of shape (n, channels, height, width) or (n, input_dim)
            reduction (str): Reduction method ('mean', 'sum', or 'none')
            
        Returns:
            torch.Tensor: Reconstruction loss
        """
        if not self.is_fitted:
            raise RuntimeError("ConvDecoderEnsemble VAE has not been fit. Call fit() first.")
        
        # Ensure input is properly shaped
        if x.dim() == 2:  # Flattened images
            if x.shape[1] != self._input_dim:
                raise ValueError(f"Input dimension ({x.shape[1]}) must match expected ({self._input_dim})")
            x_images = x.view(-1, self.input_channels, self.img_size, self.img_size)
            x_flat = x
        elif x.dim() == 4:  # Already in (B, C, H, W) format
            if x.shape[1] != self.input_channels or x.shape[2] != self.img_size or x.shape[3] != self.img_size:
                raise ValueError(f"Input dimensions ({x.shape[1:4]}) must match expected "
                               f"([{self.input_channels}, {self.img_size}, {self.img_size}])")
            x_images = x
            x_flat = x.view(x.shape[0], -1)
        else:
            raise ValueError(f"Unsupported input shape: {x.shape}")
        
        # Move to device
        x_images = x_images.to(self.device)
        x_flat = x_flat.to(self.device)
        
        # Encode and decode
        with torch.no_grad():
            mu, _ = self._encode_internal(x_images)
            reconstructions = []
            for decoder in self.decoders:
                recon = self._decode_internal(mu, decoder)
                recon_flat = recon.view(recon.size(0), -1)
                reconstructions.append(recon_flat)
            
            # Average reconstructions
            x_reconstructed = torch.stack(reconstructions).mean(dim=0)
        
        # Calculate squared error
        squared_error = (x_flat - x_reconstructed).pow(2)
        
        # Apply reduction
        if reduction == 'mean':
            return squared_error.mean()
        elif reduction == 'sum':
            return squared_error.sum()
        elif reduction == 'none':
            return squared_error
        else:
            raise ValueError(f"Unknown reduction method: {reduction}")
    
    @property
    def input_dim(self) -> int:
        """Dimension of input space."""
        return self._input_dim
    
    @property
    def encoded_dim(self) -> int:
        """Dimension of encoded space."""
        return self._encoded_dim