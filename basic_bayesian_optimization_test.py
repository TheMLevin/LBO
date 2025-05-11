import torch
import numpy as np
import matplotlib.pyplot as plt
from bayesian_optimizer import BayesianOptimizer, WeightedLogEI
from encoder import IdentityEncoder, PCAEncoder, VAEEncoder, DecoderEnsemble, PairedEnsembleVAEEncoder
from botorch.acquisition import LogExpectedImprovement
import time
import sys
import os
from collections import defaultdict
import copy
import uuid
from typing import Dict, List, Tuple, Optional, Callable

print("Script started")

# Create directory for saving/loading encoders
ENCODER_DIR = "saved_encoders"
os.makedirs(ENCODER_DIR, exist_ok=True)

def get_encoder_filename(encoder_type, params, trial_seed=None):
    """ß
    Generate a unique filename for saving/loading an encoder based on its type and parameters.
    
    Args:
        encoder_type: Type of encoder (e.g., 'vae_small', 'pca', 'decoder_ensemble')
        params: Dictionary of parameters that define the encoder
        trial_seed: Optional trial seed to include in the filename
        
    Returns:
        str: Filename for the encoder
    """
    # Create a parameter string for the filename
    param_str = "_".join([f"{k}-{v}" for k, v in params.items()])
    
    # Add trial seed if provided
    if trial_seed is not None:
        return f"{ENCODER_DIR}/{encoder_type}_{param_str}_seed{trial_seed}.pt"
    else:
        return f"{ENCODER_DIR}/{encoder_type}_{param_str}.pt"

def save_encoder(encoder, filename):
    """
    Save an encoder to a file.
    
    Args:
        encoder: The encoder to save
        filename: Path to save the encoder
    """
    try:
        torch.save(encoder, filename)
        print(f"Saved encoder to {filename}")
    except Exception as e:
        print(f"Error saving encoder to {filename}: {str(e)}")

def load_encoder(filename):
    """
    Load an encoder from a file.
    
    Args:
        filename: Path to the encoder file
        
    Returns:
        The loaded encoder, or None if loading failed
    """
    try:
        if os.path.exists(filename):
            encoder = torch.load(filename)
            print(f"Loaded encoder from {filename}")
            return encoder
        else:
            print(f"Encoder file {filename} not found")
            return None
    except Exception as e:
        print(f"Error loading encoder from {filename}: {str(e)}")
        return None

class ManifoldConstrainedQuadratic:
    """
    Complex non-differentiable function in high dimensions with manifold constraints and discrete dimensions.
    
    The objective is a function with both smooth and non-differentiable components in a high-dimensional space,
    but valid points are constrained to lie on a lower-dimensional manifold.
    
    Some dimensions are discrete (integer or binary), adding complexity to the optimization.
    
    The manifold constraint is implemented as a projection operation that maps
    arbitrary points to the manifold.
    
    Note: This is a MAXIMIZATION problem, with maximum value of 0 at the global maximizer.
    """
    def __init__(self, n_dims=25, manifold_dims=6, discrete_dims=8, binary_dims=4, seed=None):
        # Set seed for reproducibility if provided
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
        
        # Function dimensionality
        self.n_dims = n_dims
        self.manifold_dims = manifold_dims
        
        # Number of discrete dimensions (integer and binary)
        self.discrete_dims = discrete_dims
        self.binary_dims = binary_dims
        
        # Ensure we don't have too many discrete dimensions
        assert discrete_dims + binary_dims <= n_dims, "Too many discrete dimensions"
        
        # The continuous dimensions are the remaining ones
        self.continuous_dims = n_dims - discrete_dims - binary_dims
        
        # Track which dimensions are what type
        self.dim_types = ['continuous'] * self.continuous_dims + ['integer'] * (discrete_dims - binary_dims) + ['binary'] * binary_dims
        np.random.shuffle(self.dim_types)  # Shuffle the dimension types
        
        # Create indices for each type
        self.continuous_indices = [i for i, t in enumerate(self.dim_types) if t == 'continuous']
        self.integer_indices = [i for i, t in enumerate(self.dim_types) if t == 'integer']
        self.binary_indices = [i for i, t in enumerate(self.dim_types) if t == 'binary']
        
        print(f"Dimension types: {self.dim_types}")
        print(f"Continuous indices: {self.continuous_indices}")
        print(f"Integer indices: {self.integer_indices}")
        print(f"Binary indices: {self.binary_indices}")
        
        # The bounds of the function are [0, 1]^n_dims
        self.bounds = torch.zeros(n_dims, 2)
        self.bounds[:, 1] = 1.0
        
        # Create a nonlinear manifold using a combination of sinusoidal functions
        # We'll use random frequencies and phases to make it more complex
        self.frequencies = torch.rand(manifold_dims, manifold_dims) * 5.0 + 1.0  # Random frequencies between 1 and 6
        self.phases = torch.rand(manifold_dims) * 2 * np.pi  # Random phases
        self.amplitudes = torch.rand(manifold_dims) * 0.5 + 0.5  # Random amplitudes between 0.5 and 1.0
        
        # Generate a random linear subspace for the manifold base
        rand_matrix = torch.randn(n_dims, n_dims)
        q, r = torch.linalg.qr(rand_matrix)
        self.projection_matrix = q[:, :manifold_dims]  # First manifold_dims columns
        
        # The global optimum in the manifold space
        self.manifold_optimum = torch.rand(manifold_dims)
        
        # Map the manifold optimum to the full space
        self.global_maximizer = self.nonlinear_map(self.manifold_optimum)
        
        # Ensure the global maximizer is within bounds [0, 1]^n_dims
        self.global_maximizer = torch.clamp(self.global_maximizer, 0.0, 1.0)
        
        # Discretize the appropriate dimensions of the global maximizer
        self.global_maximizer = self.discretize_point(self.global_maximizer)
        
        # The maximum value of the function is 0
        self.global_maximum = 0.0
        
        # Create random weights for the non-differentiable components
        self.nondiff_weights = torch.rand(n_dims) * 0.5  # Random weights for non-differentiable terms
        self.step_thresholds = torch.rand(n_dims) * 0.7 + 0.15  # Thresholds for step functions (0.15-0.85)
        
        # Create dimension clusters for interaction terms
        n_clusters = min(5, n_dims // 5)
        self.clusters = []
        dims = list(range(n_dims))
        np.random.shuffle(dims)
        
        # Split dimensions into clusters for interaction terms
        cluster_size = n_dims // n_clusters
        for i in range(n_clusters):
            start_idx = i * cluster_size
            end_idx = start_idx + cluster_size if i < n_clusters - 1 else n_dims
            self.clusters.append(dims[start_idx:end_idx])
        
        print(f"Created {n_dims}D non-differentiable function with {manifold_dims}D manifold constraint")
        print(f"Includes {discrete_dims} discrete dimensions ({binary_dims} binary)")
        print(f"Global maximum: {self.global_maximum:.6f}")
        print(f"Global maximizer: {self.global_maximizer}")
    
    def nonlinear_map(self, manifold_coords):
        """Map from manifold coordinates to the full space using a nonlinear transformation."""
        if manifold_coords.dim() == 1:
            manifold_coords = manifold_coords.unsqueeze(0)
        
        batch_size = manifold_coords.shape[0]
        
        # First do a linear projection as the base mapping
        base_projection = manifold_coords @ self.projection_matrix.T
        
        # Then add nonlinear components based on sinusoidal functions
        nonlinear_components = torch.zeros_like(base_projection)
        
        for i in range(self.manifold_dims):
            # Compute complex sinusoidal pattern for this manifold dimension
            sinusoid = torch.sin(manifold_coords @ self.frequencies[i] + self.phases[i]) * self.amplitudes[i]
            # Add this pattern to the projection, scaled by the appropriate column of the projection matrix
            nonlinear_components += sinusoid.unsqueeze(1) * self.projection_matrix[:, i].unsqueeze(0)
        
        # Combine base projection with nonlinear components
        full_space_points = base_projection + 0.3 * nonlinear_components  # Scale factor to control nonlinearity
        
        # Ensure points are within bounds
        full_space_points = torch.clamp(full_space_points, 0.0, 1.0)
        
        return full_space_points
    
    def find_manifold_coordinates(self, x, n_iterations=10):
        """Find the manifold coordinates that best map to the given point in the full space."""
        if x.dim() == 1:
            x = x.unsqueeze(0)
        
        batch_size = x.shape[0]
        
        # Start with a random guess for manifold coordinates
        manifold_coords = torch.rand(batch_size, self.manifold_dims)
        manifold_coords.requires_grad_(True)
        
        # Use gradient descent to find the best manifold coordinates
        optimizer = torch.optim.Adam([manifold_coords], lr=0.1)
        
        for i in range(n_iterations):
            # Map manifold coordinates to full space
            full_space_points = self.nonlinear_map(manifold_coords)
            
            # Compute distance to target point
            loss = torch.mean((full_space_points - x) ** 2)
            
            # Update manifold coordinates
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Project to [0, 1] range
            with torch.no_grad():
                manifold_coords.data.clamp_(0.0, 1.0)
        
        # Return the best manifold coordinates found
        manifold_coords.requires_grad_(False)
        return manifold_coords
    
    def project_to_manifold(self, x):
        """Project points onto the manifold."""
        if x.dim() == 1:
            x = x.unsqueeze(0)
        
        # Find the closest manifold coordinates
        manifold_coords = self.find_manifold_coordinates(x)
        
        # Map back to the full space
        projected = self.nonlinear_map(manifold_coords)
        
        # Discretize the appropriate dimensions
        projected = self.discretize_point(projected)
        
        return projected
    
    def discretize_point(self, x):
        """Discretize the appropriate dimensions of the point."""
        if x.dim() == 1:
            x = x.unsqueeze(0)
        
        result = x.clone()
        
        # Discretize integer dimensions
        for idx in self.integer_indices:
            # Scale to [0, 10] range and round to nearest integer, then rescale to [0, 1]
            result[:, idx] = torch.round(x[:, idx] * 10) / 10
        
        # Discretize binary dimensions
        for idx in self.binary_indices:
            # Round to 0 or 1
            result[:, idx] = torch.round(x[:, idx])
        
        return result
    
    def is_valid(self, x):
        """Check if points are on (or close to) the manifold."""
        if x.dim() == 1:
            x = x.unsqueeze(0)
        
        # Project points onto the manifold
        projected = self.project_to_manifold(x)
        
        # Check distance to manifold for continuous dimensions only
        # For discrete dimensions, we only check if they have valid values
        
        # Initialize mask for valid points
        valid = torch.ones(x.shape[0], dtype=torch.bool, device=x.device)
        
        # Check continuous dimensions
        if self.continuous_indices:
            continuous_dists = torch.sqrt(torch.sum(
                (x[:, self.continuous_indices] - projected[:, self.continuous_indices])**2, dim=1))
            continuous_valid = continuous_dists <= 1e-3
            valid = valid & continuous_valid
        
        # Check integer dimensions
        if self.integer_indices:
            for idx in self.integer_indices:
                # Check if values are multiples of 0.1 (scaled integers)
                remainder = torch.abs(x[:, idx] * 10 - torch.round(x[:, idx] * 10))
                integer_valid = remainder <= 1e-3
                valid = valid & integer_valid
        
        # Check binary dimensions
        if self.binary_indices:
            for idx in self.binary_indices:
                # Check if values are close to 0 or 1
                binary_valid = (torch.abs(x[:, idx]) <= 1e-3) | (torch.abs(x[:, idx] - 1.0) <= 1e-3)
                valid = valid & binary_valid
        
        return valid
    
    def _evaluate_complex_function(self, x):
        """
        Evaluate a complex function with non-differentiable components.
        This is a MAXIMIZATION problem, with maximum value of 0 at the global maximizer.
        """
        if x.dim() == 1:
            x = x.unsqueeze(0)
        
        batch_size = x.shape[0]
        
        # Base quadratic component (differentiable)
        base_value = -torch.sum((x - self.global_maximizer)**2, dim=1)
        
        # Add non-differentiable absolute value terms
        abs_terms = -torch.sum(self.nondiff_weights * torch.abs(x - self.global_maximizer), dim=1)
        
        # Add step function terms (non-differentiable)
        step_terms = torch.zeros(batch_size, device=x.device)
        for i in range(self.n_dims):
            # Step function: +0.1 if x[i] > threshold
            step = 0.1 * ((x[:, i] > self.step_thresholds[i]).float() - 0.5)
            step_terms += step
        
        # Add interaction terms for dimension clusters
        interaction_terms = torch.zeros(batch_size, device=x.device)
        for cluster in self.clusters:
            if len(cluster) > 1:
                # Create pairwise interactions within cluster (non-smooth)
                for i in range(len(cluster)):
                    for j in range(i+1, len(cluster)):
                        dim1, dim2 = cluster[i], cluster[j]
                        # Non-smooth interaction: max(0, x_i * x_j - 0.5)
                        interaction = torch.relu(x[:, dim1] * x[:, dim2] - 0.5) * 0.1
                        interaction_terms += interaction
        
        # Combine all components with appropriate weights to ensure the global maximum is still at 0
        quadratic_weight = 0.6
        abs_weight = 0.25
        step_weight = 0.05
        interaction_weight = 0.1
        
        total_value = (quadratic_weight * base_value + 
                       abs_weight * abs_terms + 
                       step_weight * step_terms + 
                       interaction_weight * interaction_terms)
        
        # Normalize to keep maximum at approximately 0
        shift = quadratic_weight * 0.0 + abs_weight * 0.0 + step_weight * 0.05 + interaction_weight * 0.02
        total_value = total_value + shift
        
        return total_value
    
    def __call__(self, x):
        """Evaluate the function at x."""
        if x.dim() == 1:
            x = x.unsqueeze(0)
        
        # Project points onto the manifold
        projected = self.project_to_manifold(x)
        
        # Evaluate the complex function on the projected points
        result = self._evaluate_complex_function(projected)
        
        return result

def generate_manifold_points(n_samples, problem):
    """
    Generate points that lie on the manifold.
    
    Args:
        n_samples: Number of samples to generate
        problem: The manifold-constrained problem
        
    Returns:
        Tensor of shape (n_samples, n_dims) with the samples
    """
    # Generate random points in the manifold space
    manifold_points = torch.rand(n_samples, problem.manifold_dims)
    
    # Map to the full space using the nonlinear map
    full_space_points = problem.nonlinear_map(manifold_points)
    
    # Discretize the appropriate dimensions
    full_space_points = problem.discretize_point(full_space_points)
    
    return full_space_points

def run_single_optimization(problem, encoder, initial_points, initial_values, n_iterations=100, encoder_name="Unknown", weighted_acq_func=False, decoder_variance_fn=None, variance_weight=0.0):
    """
    Run a single optimization experiment with a specific encoder.
    
    Args:
        problem: The optimization problem
        encoder: The encoder to use
        initial_points: Initial points to start optimization
        initial_values: Function values at initial points
        n_iterations: Number of optimization iterations
        encoder_name: Name of the encoder for logging
        weighted_acq_func: Whether to use a weighted acquisition function
        decoder_variance_fn: Function to compute decoder variance (for DecoderEnsemble)
        variance_weight: Weight for decoder variance in acquisition function
        
    Returns:
        Dictionary with optimization results
    """
    # Storage for tracking objective evaluations
    evals = []   # Function values
    raw_evals = []  # Original function values before penalty
    
    # Create objective function wrapper
    
    # Create optimizer with the provided encoder
    print(f"\nCreating optimizer with {encoder_name} encoder...")
    optimizer = BayesianOptimizer(
        bounds=problem.bounds,
        n_initial_points=0,  # We'll provide our own initial points
        initial_points=initial_points,
        initial_values=initial_values,
        encoder=encoder,
        auto_normalize=True,  # Enable input normalization
        kernel_type="matern",  # Use Matern kernel for smoother interpolation
        nu=2.5,  # Set nu parameter for Matern kernel
        constraint_fn=problem.is_valid,  # Check if points are on the manifold
        projection_fn=problem.project_to_manifold  # Project points onto the manifold
    )
    
    # Define acquisition function factory
    if weighted_acq_func:
        if decoder_variance_fn is not None:
            # Use decoder variance as the weight function
            def acq_factory(model, best_f=None, **kwargs):
                return WeightedLogEI(
                    model=model, 
                    best_f=best_f, 
                    weight_fn=decoder_variance_fn, 
                    weight_param=0.5,  # Higher weight for stronger effect
                    downweight_high_values=True  # Downweight points far from manifold
                )
        else:
            # Default to standard weighted EI
            def acq_factory(model, best_f=None, **kwargs):
                return WeightedLogEI(
                    model=model, 
                    best_f=best_f, 
                    weight_fn=encoder.decoder_variance, 
                    weight_param=0.1
                )
    else:
        def acq_factory(model, best_f=None, **kwargs):
            return LogExpectedImprovement(model=model, best_f=best_f)
    
    # Run optimization
    print(f"\nRunning optimization with {encoder_name} encoder...")
    start_time = time.time()
    
    try:
        # Run optimization
        best_x, best_y, regret, projection_distances = optimizer.optimize(
            objective_function=problem,
            n_iterations=n_iterations,
            acq_func_maker=acq_factory,
            use_constraints=True  # Enable constraint handling
        )
        
        end_time = time.time()
        duration = end_time - start_time
        
        # Compute the true objective value at the best point (without penalty)
        
        print(f"Optimization with {encoder_name} completed in {duration:.2f} seconds")
        print(f"Best point found: {best_x[0]}")
        print(f"Best value found: {best_y.item():.6f}")
        print(f"Global maximum: {problem.global_maximum:.6f}")
        print(f"Distance to true optimum: {torch.norm(best_x[0] - problem.global_maximizer).item():.6f}")
        print(f"Optimality gap: {problem.global_maximum - best_y:.6f}")
        
        # Get the optimization trajectory
        Y_values = optimizer.train_Y.squeeze().cpu().numpy()
        best_values = [max(Y_values[:i+1]) for i in range(len(Y_values))]
        
        # Calculate average projection distance
        avg_projection_distance = np.mean(projection_distances) if projection_distances else 0.0
        print(f"Average projection distance: {avg_projection_distance:.6f}")
        
        return {
            "encoder_name": encoder_name,
            "best_x": best_x,
            "best_y": best_y.item(),
            "global_maximum": problem.global_maximum,
            "distance_to_optimum": torch.norm(best_x[0] - problem.global_maximizer).item(),
            "optimality_gap": problem.global_maximum - best_y.item(),
            "time": duration,
            "iterations": n_iterations,
            "Y_values": Y_values,
            "best_values": best_values,
            "evals": evals,  # Penalized values
            "raw_evals": raw_evals,  # Original values without penalty
            "train_X": optimizer.original_train_X,
            "train_Y": optimizer.train_Y,
            "projection_distances": projection_distances,
            "avg_projection_distance": avg_projection_distance,
            "result_id": str(uuid.uuid4())  # Unique ID for this result
        }
        
    except Exception as e:
        print(f"Error during optimization with {encoder_name}: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def run_manifold_optimization_comparison(
    n_dims=20, 
    manifold_dims=5, 
    discrete_dims=5, 
    binary_dims=3, 
    n_iterations=100, 
    n_trials=3,
    base_seed=42,
    force_retrain=False  # New parameter to force retraining even if saved encoders exist
):
    """
    Run Bayesian optimization experiments comparing different encoders over multiple trials.
    
    Args:
        n_dims: Number of dimensions in the full space
        manifold_dims: Number of dimensions in the manifold
        discrete_dims: Number of discrete dimensions (including binary)
        binary_dims: Number of binary dimensions (subset of discrete_dims)
        n_iterations: Number of optimization iterations
        n_trials: Number of trials to run for statistical analysis
        base_seed: Base random seed
        force_retrain: Whether to force retraining of encoders even if saved versions exist
        
    Returns:
        Dictionary with optimization results from different encoders across trials
    """
    print(f"Starting Bayesian optimization comparison with manifold-constrained quadratic function")
    print(f"Full space dimensions: {n_dims}, Manifold dimensions: {manifold_dims}")
    print(f"Discrete dimensions: {discrete_dims} (including {binary_dims} binary)")
    print(f"Running {n_trials} trials for statistical analysis")
    
    # Initialize multi-trial results storage
    all_trial_results = {}
    
    # For each trial
    for trial in range(n_trials):
        print(f"\n{'='*20} TRIAL {trial+1}/{n_trials} {'='*20}")
        
        # Set random seeds for reproducibility, using different seed for each trial
        trial_seed = base_seed + trial
        torch.manual_seed(trial_seed)
        np.random.seed(trial_seed)
        print(f"Random seeds set for trial {trial+1} (seed={trial_seed})")
        
        # Create the problem for this trial
        problem = ManifoldConstrainedQuadratic(
            n_dims=n_dims, 
            manifold_dims=manifold_dims,
            discrete_dims=discrete_dims,
            binary_dims=binary_dims,
            seed=trial_seed
        )
        
        # Generate initial points on the manifold
        n_initial = 15  # More initial points for complex manifold
        print(f"\nGenerating {n_initial} initial points on the manifold...")
        initial_points = generate_manifold_points(n_initial * 4, problem)
        
        # Evaluate initial points
        initial_values = torch.tensor([problem(x.unsqueeze(0)).item() for x in initial_points]).unsqueeze(-1)

        indices = initial_values.argsort()[:n_initial].squeeze()
        initial_points = initial_points[indices]
        initial_values = initial_values[indices]

        # For display, show the values as maximization problem (higher is better)
        print(f"Initial points evaluation complete")
        print(f"  Best initial value: {initial_values.max().item():.6f}")
        print(f"  Mean initial value: {initial_values.mean().item():.6f}")
        print(f"  Initial values range: [{initial_values.min().item():.6f}, {initial_values.max().item():.6f}]")
        
        # Generate manifold data for training encoders
        n_encoder_samples = 1000
        encoder_training_data = generate_manifold_points(n_encoder_samples, problem)
        
        # Add some random perturbations to better capture the variance
        encoder_training_data += torch.randn_like(encoder_training_data) * 0.01
        encoder_training_data = torch.clamp(encoder_training_data, 0.0, 1.0)
        
        # Define fixed latent dimension for all non-identity encoders
        latent_dim = 8  # Increased from 6 to 8 to handle higher dimensions
        print(f"\nUsing latent dimension of {latent_dim} for all non-identity encoders")
        
        # Create encoders to compare
        encoders = {}
        
        # 1. Identity encoder (no dimensionality reduction)
        encoders["identity"] = IdentityEncoder(dim=problem.n_dims)
        
        # 2. PCA encoder (reduce to latent dimension)
        # Create parameters dict for PCA encoder
        pca_params = {
            "input_dim": n_dims,
            "encoded_dim": latent_dim,
            "problem_dims": f"{n_dims}_{manifold_dims}_{discrete_dims}_{binary_dims}"
        }
        pca_filename = get_encoder_filename("pca", pca_params, trial_seed)
        
        # Try to load PCA encoder, train if not found or force_retrain is True
        if not force_retrain and os.path.exists(pca_filename):
            encoders["pca"] = load_encoder(pca_filename)
        else:
            print("\nTraining PCA encoder...")
            encoders["pca"] = PCAEncoder(
                input_dim=n_dims,
                encoded_dim=latent_dim,
                data=encoder_training_data
            )
            save_encoder(encoders["pca"], pca_filename)
        
        # 3. VAE with small architecture
        # Parameters for small VAE
        vae_small_params = {
            "input_dim": n_dims,
            "latent_dim": latent_dim,
            "hidden": "64_32",
            "beta": 0.05,
            "problem_dims": f"{n_dims}_{manifold_dims}_{discrete_dims}_{binary_dims}"
        }
        vae_small_filename = get_encoder_filename("vae_small", vae_small_params, trial_seed)
        
        # Try to load VAE small, train if not found or force_retrain is True
        if not force_retrain and os.path.exists(vae_small_filename):
            encoders["vae_small"] = load_encoder(vae_small_filename)
        else:
            print("\nTraining VAE-Small...")
            vae_small = VAEEncoder(
                input_dim=n_dims,
                latent_dim=latent_dim,
                hidden_dims=[64, 32],  # Increased hidden layer sizes for higher dimensions
                beta=0.05,  # Lower KL divergence weight for more complex function
                train_iters=500,  # More training iterations for more complex problem
                batch_size=128,
                learning_rate=1e-3
            )
            vae_small.fit(encoder_training_data)
            encoders["vae_small"] = vae_small
            save_encoder(vae_small, vae_small_filename)
        
        # 4. VAE with medium architecture
        # Parameters for medium VAE
        vae_medium_params = {
            "input_dim": n_dims,
            "latent_dim": latent_dim,
            "hidden": "128_64",
            "beta": 0.2,
            "problem_dims": f"{n_dims}_{manifold_dims}_{discrete_dims}_{binary_dims}"
        }
        vae_medium_filename = get_encoder_filename("vae_medium", vae_medium_params, trial_seed)
        
        # Try to load VAE medium, train if not found or force_retrain is True
        if not force_retrain and os.path.exists(vae_medium_filename):
            encoders["vae_medium"] = load_encoder(vae_medium_filename)
        else:
            print("\nTraining VAE-Medium...")
            vae_medium = VAEEncoder(
                input_dim=n_dims,
                latent_dim=latent_dim,
                hidden_dims=[128, 64],  # Increased hidden layer sizes
                beta=0.2,  # Medium KL divergence weight
                train_iters=500,
                batch_size=128,
                learning_rate=1e-3
            )
            vae_medium.fit(encoder_training_data)
            encoders["vae_medium"] = vae_medium
            save_encoder(vae_medium, vae_medium_filename)
        
        # 5. VAE with deep architecture
        # Parameters for deep VAE
        vae_deep_params = {
            "input_dim": n_dims,
            "latent_dim": latent_dim,
            "hidden": "128_64_32",
            "beta": 0.1,
            "problem_dims": f"{n_dims}_{manifold_dims}_{discrete_dims}_{binary_dims}"
        }
        vae_deep_filename = get_encoder_filename("vae_deep", vae_deep_params, trial_seed)
        
        # Try to load VAE deep, train if not found or force_retrain is True
        if not force_retrain and os.path.exists(vae_deep_filename):
            encoders["vae_deep"] = load_encoder(vae_deep_filename)
        else:
            print("\nTraining VAE-Deep...")
            vae_deep = VAEEncoder(
                input_dim=n_dims,
                latent_dim=latent_dim,
                hidden_dims=[128, 64, 32],  # Deep architecture with 3 larger layers
                beta=0.1,
                train_iters=500,
                batch_size=128,
                learning_rate=1e-3
            )
            vae_deep.fit(encoder_training_data)
            encoders["vae_deep"] = vae_deep
            save_encoder(vae_deep, vae_deep_filename)
        
        # 6. VAE with wide architecture
        # Parameters for wide VAE
        vae_wide_params = {
            "input_dim": n_dims,
            "latent_dim": latent_dim,
            "hidden": "256_128",
            "beta": 0.1,
            "problem_dims": f"{n_dims}_{manifold_dims}_{discrete_dims}_{binary_dims}"
        }
        vae_wide_filename = get_encoder_filename("vae_wide", vae_wide_params, trial_seed)
        
        # Try to load VAE wide, train if not found or force_retrain is True
        if not force_retrain and os.path.exists(vae_wide_filename):
            encoders["vae_wide"] = load_encoder(vae_wide_filename)
        else:
            print("\nTraining VAE-Wide...")
            vae_wide = VAEEncoder(
                input_dim=n_dims,
                latent_dim=latent_dim,
                hidden_dims=[256, 128],  # Wider layers for higher dimensional space
                beta=0.1,
                train_iters=500,
                batch_size=128, 
                learning_rate=1e-3
            )
            vae_wide.fit(encoder_training_data)
            encoders["vae_wide"] = vae_wide
            save_encoder(vae_wide, vae_wide_filename)
        
        # 7. DecoderEnsemble with multiple decoders
        # Parameters for DecoderEnsemble
        decoder_ensemble_params = {
            "input_dim": n_dims,
            "latent_dim": latent_dim,
            "hidden": "64_32",
            "n_decoders": 5,
            "beta": 0.05,
            "problem_dims": f"{n_dims}_{manifold_dims}_{discrete_dims}_{binary_dims}"
        }
        decoder_ensemble_filename = get_encoder_filename("decoder_ensemble", decoder_ensemble_params, trial_seed)
        
        # Try to load DecoderEnsemble, train if not found or force_retrain is True
        if not force_retrain and os.path.exists(decoder_ensemble_filename):
            encoders["decoder_ensemble"] = load_encoder(decoder_ensemble_filename)
        else:
            print("\nTraining DecoderEnsemble...")
            decoder_ensemble = DecoderEnsemble(
                input_dim=n_dims,
                latent_dim=latent_dim,
                n_decoders=5,  # Use 5 decoders for better variance estimation
                hidden_dims=[64, 128],
                beta=0.05,
                train_iters=500,
                batch_size=128,
                learning_rate=1e-3
            )
            decoder_ensemble.fit(encoder_training_data)
            encoders["decoder_ensemble"] = decoder_ensemble
            save_encoder(decoder_ensemble, decoder_ensemble_filename)
        
        # 8. PairedEnsembleVAEEncoder with multiple encoder-decoder pairs
        # Parameters for PairedEnsembleVAEEncoder
        # paired_ensemble_params = {
        #     "input_dim": n_dims,
        #     "latent_dim": latent_dim,
        #     "hidden": "128_64",
        #     "n_pairs": 3,
        #     "beta": 0.1,
        #     "gamma": 0.1,
        #     "problem_dims": f"{n_dims}_{manifold_dims}_{discrete_dims}_{binary_dims}"
        # }
        # paired_ensemble_filename = get_encoder_filename("paired_ensemble", paired_ensemble_params, trial_seed)
        
        # # Try to load PairedEnsembleVAEEncoder, train if not found or force_retrain is True
        # if not force_retrain and os.path.exists(paired_ensemble_filename):
        #     encoders["paired_ensemble"] = load_encoder(paired_ensemble_filename)
        # else:
        #     print("\nTraining PairedEnsembleVAEEncoder...")
        #     paired_ensemble = PairedEnsembleVAEEncoder(
        #         input_dim=n_dims,
        #         latent_dim=latent_dim,
        #         n_pairs=3,  # Use 3 encoder-decoder pairs
        #         hidden_dims=[128, 64],
        #         beta=0.1,
        #         gamma=0.1,  # Weight for information radius loss
        #         train_iters=500,
        #         batch_size=128,
        #         learning_rate=1e-3
        #     )
        #     paired_ensemble.fit(encoder_training_data)
        #     encoders["paired_ensemble"] = paired_ensemble
        #     save_encoder(paired_ensemble, paired_ensemble_filename)
        
        # Run optimization with each encoder
        trial_results = {}
        
        for name, encoder in encoders.items():
            # Create a displayable name
            display_name = name.replace('_', '-').title()
            
            # Determine if we should use weighted acquisition function for ensemble encoders
            use_weighted_acq = name in ["decoder_ensemble", "paired_ensemble"]
            use_decoder_variance = name in ["decoder_ensemble", "paired_ensemble"]
            decoder_variance_fn = encoder.decoder_variance if hasattr(encoder, 'decoder_variance') else None
            variance_weight = 1 if use_decoder_variance else 0.0
            
            # Run optimization
            print(f"\n{'-'*20} Running optimization with {display_name} for trial {trial+1} {'-'*20}")
            result = run_single_optimization(
                problem=problem,
                encoder=encoder,
                initial_points=initial_points.clone(),
                initial_values=initial_values.clone(),
                n_iterations=n_iterations,
                encoder_name=f"{display_name} (Trial {trial+1})",
                weighted_acq_func=use_weighted_acq,
                decoder_variance_fn=decoder_variance_fn,
                variance_weight=variance_weight
            )
            trial_results[name] = result
            
            # Store in multi-trial results
            if name not in all_trial_results:
                all_trial_results[name] = []
            all_trial_results[name].append(result)
        
        # Compare results for this trial
        print(f"\n===== Trial {trial+1} Optimization Results =====")
        print(f"\n{'Encoder':<30} {'Best Value':<15} {'Opt. Gap':<15} {'Distance':<15} {'Avg. Proj. Dist.':<20} {'Time (s)':<10}")
        print('-' * 90)
        
        for name, result in trial_results.items():
            if result:
                display_name = name.replace('_', '-').title()
                
                print(f"{display_name:<30} {result['best_y']:<15.6f} {result['optimality_gap']:<15.6f} "
                      f"{result['distance_to_optimum']:<15.6f} {result['avg_projection_distance']:<20.6f} {result['time']:<10.2f}")
    
    # Compute statistics across trials
    print("\n===== Multi-Trial Optimization Results =====")
    print(f"\n{'Encoder':<30} {'Mean Best Value':<20} {'Std Dev':<15} {'Median':<15} {'Avg. Proj. Dist.':<20}")
    print('-' * 100)
    
    # Process multi-trial results
    stats = {}
    for name, results_list in all_trial_results.items():
        best_y_values = [r['best_y'] for r in results_list if r]
        distances = [r['distance_to_optimum'] for r in results_list if r]
        projection_distances = [r['avg_projection_distance'] for r in results_list if r]
        
        if best_y_values:
            mean_y = np.mean(best_y_values)
            std_y = np.std(best_y_values)
            median_y = np.median(best_y_values)
            mean_dist = np.mean(distances)
            mean_proj_dist = np.mean(projection_distances)
            
            # Format display name
            display_name = name.replace('_', '-').title()
            
            print(f"{display_name:<30} {mean_y:<20.6f} {std_y:<15.6f} {median_y:<15.6f} {mean_proj_dist:<20.6f}")
            
            # Store statistics for plotting
            stats[name] = {
                'mean_best_y': mean_y,
                'std_best_y': std_y,
                'median_best_y': median_y,
                'mean_distance': mean_dist,
                'mean_projection_distance': mean_proj_dist
            }
    
    # Plot comparison of convergence across trials
    try:
        # Start by processing all trial data into a useful format for statistics
        # We need to compute statistics over iterations for each encoder
        iter_stats = {}
        max_iters = 0
        
        # First, determine the maximum number of iterations across all trials and encoders
        for name, results_list in all_trial_results.items():
            for result in results_list:
                if result and len(result['best_values']) > max_iters:
                    max_iters = len(result['best_values'])
        
        # For each encoder, collect data across trials
        for name, results_list in all_trial_results.items():
            # Initialize arrays for this encoder
            all_values = np.zeros((len(results_list), max_iters))
            all_abs_distances = np.zeros((len(results_list), max_iters))
            all_times = []
            
            # Fill in the data from each trial
            for i, result in enumerate(results_list):
                if result:
                    # Get the best values and pad if necessary
                    values = result['best_values']
                    n_values = len(values)
                    all_values[i, :n_values] = values
                    
                    # If a trial didn't run to completion, repeat the last value
                    if n_values < max_iters:
                        all_values[i, n_values:] = values[-1]
                    
                    # Calculate absolute distances to global optimum
                    abs_distances = [abs(val - result['global_maximum']) for val in values]
                    all_abs_distances[i, :n_values] = abs_distances
                    
                    # If a trial didn't run to completion, repeat the last distance
                    if n_values < max_iters:
                        all_abs_distances[i, n_values:] = abs_distances[-1]
                    
                    # Record optimization time
                    all_times.append(result['time'])
            
            # Calculate statistics across trials for this encoder
            mean_values = np.mean(all_values, axis=0)
            median_values = np.median(all_values, axis=0)
            q25_values = np.percentile(all_values, 25, axis=0)
            q75_values = np.percentile(all_values, 75, axis=0)
            
            mean_abs_dist = np.mean(all_abs_distances, axis=0)
            median_abs_dist = np.median(all_abs_distances, axis=0)
            q25_abs_dist = np.percentile(all_abs_distances, 25, axis=0)
            q75_abs_dist = np.percentile(all_abs_distances, 75, axis=0)
            
            mean_time = np.mean(all_times) if all_times else 0
            
            # Store statistics for this encoder
            iter_stats[name] = {
                'mean_values': mean_values,
                'median_values': median_values,
                'q25_values': q25_values,
                'q75_values': q75_values,
                'mean_abs_dist': mean_abs_dist,
                'median_abs_dist': median_abs_dist,
                'q25_abs_dist': q25_abs_dist,
                'q75_abs_dist': q75_abs_dist,
                'mean_time': mean_time,
                'iterations': np.arange(max_iters)
            }
        
        # Now create the plots
        # Create a figure with two main subplots for iteration and time comparisons
        plt.figure(figsize=(20, 12))
        
        # Define colors for consistent visualization
        colors = {
            'identity': 'blue',
            'pca': 'green',
            'vae_small': 'red',
            'vae_medium': 'purple',
            'vae_deep': 'orange',
            'vae_wide': 'brown',
            'decoder_ensemble': 'teal',
            'paired_ensemble': 'pink'
        }
        
        # Define line styles for consistent visualization
        line_styles = {
            'identity': '-',
            'pca': '--',
            'vae_small': '-.',
            'vae_medium': '-',
            'vae_deep': '-.',
            'vae_wide': ':',
            'decoder_ensemble': '-',
            'paired_ensemble': '--'
        }
        
        # SUBPLOT 1: Absolute Distance (Log Scale) vs. Iterations with Quartiles
        plt.subplot(1, 2, 1)
        
        # Store min/max for y-axis scaling
        min_abs_dist = float('inf')
        max_abs_dist = float('-inf')
        
        # Plot mean absolute distance to global optimum with quartile bands for each encoder
        for name, stats in iter_stats.items():
            display_name = name.replace('_', '-').title()
            iterations = stats['iterations']
            
            # Plot median line
            plt.semilogy(iterations, stats['median_abs_dist'], 
                     linewidth=2.0, 
                     color=colors.get(name, 'black'),
                     linestyle=line_styles.get(name, '-'),
                     label=f"{display_name} (Median)")
            
            # Plot quartile bands
            plt.fill_between(iterations, 
                          stats['q25_abs_dist'], 
                          stats['q75_abs_dist'],
                          alpha=0.2, 
                          color=colors.get(name, 'black'),
                          label=f"{display_name} (25-75%)")
            
            # Track min/max for y-axis scaling
            min_val = min(stats['q25_abs_dist'])
            max_val = max(stats['q75_abs_dist'])
            if min_val < min_abs_dist:
                min_abs_dist = min_val
            if max_val > max_abs_dist:
                max_abs_dist = max_val
        
        plt.xlabel('Iteration', fontsize=14)
        plt.ylabel('Absolute Distance to Optimum (Log Scale)', fontsize=14)
        plt.title(f'Absolute Distance to Global Optimum Across {n_trials} Trials\n(Log Scale - Lower is Better, Median with 25-75% Quartiles)', fontsize=16)
        plt.legend(fontsize=12, loc='upper right')
        plt.grid(True, alpha=0.3, which='both')
        
        # Set y-axis limits with appropriate padding
        if min_abs_dist > 0:  # Avoid log(0) issues
            plt.ylim(min_abs_dist * 0.5, max_abs_dist * 2.0)
        
        # SUBPLOT 2: Absolute Distance vs. Time (Log Scale) with Quartiles
        plt.subplot(1, 2, 2)
        
        # For each encoder, create a time scale based on iterations and mean time
        for name, stats in iter_stats.items():
            display_name = name.replace('_', '-').title()
            mean_time = stats['mean_time']
            n_iters = len(stats['iterations'])
            
            # Create time points (linear distribution based on mean time)
            time_points = np.linspace(0, mean_time, n_iters)
            
            # Plot median line
            plt.semilogy(time_points, stats['median_abs_dist'], 
                     linewidth=2.0, 
                     color=colors.get(name, 'black'),
                     linestyle=line_styles.get(name, '-'),
                     label=f"{display_name} (Median, {mean_time:.1f}s)")
            
            # Plot quartile bands
            plt.fill_between(time_points, 
                          stats['q25_abs_dist'], 
                          stats['q75_abs_dist'],
                          alpha=0.2, 
                          color=colors.get(name, 'black'))
        
        plt.xlabel('Wall Clock Time (seconds)', fontsize=14)
        plt.ylabel('Absolute Distance to Optimum (Log Scale)', fontsize=14)
        plt.title(f'Absolute Distance by Time Across {n_trials} Trials\n(Log Scale - Lower is Better, Median with 25-75% Quartiles)', fontsize=16)
        plt.legend(fontsize=12, loc='upper right')
        plt.grid(True, alpha=0.3, which='both')
        
        # Use the same y-axis limits as the first subplot for consistency
        if min_abs_dist > 0:  # Avoid log(0) issues
            plt.ylim(min_abs_dist * 0.5, max_abs_dist * 2.0)
        
        plt.tight_layout()
        
        # Save the multi-trial comparison plot
        plot_filename = f'encoder_multi_trial_comparison.png'
        plt.savefig(plot_filename, dpi=300)
        print(f"\nMulti-trial encoder comparison plot saved to {plot_filename}")
        
        # Create boxplot of projection distances
        try:
            plt.figure(figsize=(12, 6))
            
            # Collect projection distances for each encoder
            encoder_names = []
            distance_data = []
            
            # Define the color map for consistent colors with other plots
            colors = {
                'identity': 'blue',
                'pca': 'green',
                'vae_small': 'red',
                'vae_medium': 'purple',
                'vae_deep': 'orange',
                'vae_wide': 'brown',
                'decoder_ensemble': 'teal'  # Add color for DecoderEnsemble
            }
            
            # Process each encoder's results
            for name, results_list in all_trial_results.items():
                # Skip encoders with no valid results
                if not results_list:
                    continue
                
                # Get the base encoder name (without weight suffix)
                base_name = name
                display_name = name.replace('_', '-').title()
                
                # Collect projection distances from all trials
                all_distances = []
                for result in results_list:
                    if result and "projection_distances" in result:
                        all_distances.extend(result["projection_distances"])
                
                if all_distances:
                    encoder_names.append(display_name)
                    distance_data.append(all_distances)
                    print(f"Collected {len(all_distances)} projection distances for {display_name}")
            
            # Create the boxplot if we have data
            if distance_data:
                # Create boxplot with custom colors
                boxprops = dict(linewidth=2)
                whiskerprops = dict(linewidth=2)
                capprops = dict(linewidth=2)
                medianprops = dict(linewidth=2, color='black')
                
                # Create boxplots with matching colors from the line plot
                boxplot = plt.boxplot(
                    distance_data, 
                    labels=encoder_names,
                    patch_artist=True,  # Fill boxes with color
                    boxprops=boxprops,
                    whiskerprops=whiskerprops,
                    capprops=capprops,
                    medianprops=medianprops
                )
                
                # Set box colors to match line colors
                for i, (box, name) in enumerate(zip(boxplot['boxes'], all_trial_results.keys())):
                    # Get the base name for coloring
                    base_name = name
                    box.set_facecolor(colors.get(base_name, 'gray'))
                
                plt.title(f'Projection Distances by Encoder Across {n_trials} Trials', fontsize=16)
                plt.ylabel('Projection Distance (lower is better)', fontsize=14)
                plt.xlabel('Encoder', fontsize=14)
                plt.xticks(rotation=45, ha='right', fontsize=12)
                plt.grid(True, linestyle='--', alpha=0.3)
                
                # Add some text explaining what projection distances mean
                plt.figtext(
                    0.5, 0.01, 
                    "Projection distances measure how far predicted points need to be moved to satisfy manifold constraints.\n"
                    "Lower values indicate better alignment with the manifold structure.",
                    ha='center', fontsize=12, wrap=True
                )
                
                plt.tight_layout(rect=[0, 0.05, 1, 1])  # Adjust layout to make room for text
                
                # Save the projection distances plot
                plt.savefig('encoder_projection_distances.png', dpi=300)
                print("Projection distances boxplot saved to encoder_projection_distances.png")
            else:
                print("No projection distance data available for plotting")
            
            plt.close()
            
        except Exception as e:
            print(f"Error creating projection distances boxplot: {str(e)}")
            import traceback
            traceback.print_exc()
        
        # Create a new plot comparing projection weights for each encoder
        try:
            # Get base encoders (without _w suffix) and their weights
            base_encoders = set()
            weight_variants = defaultdict(list)
            
            # If we have any weight variants, create a comparison plot
            if weight_variants:
                plt.figure(figsize=(15, 10))
                
                # First subplot: Compare best values across weights
                plt.subplot(2, 2, 1)
                
                for base_name in sorted(base_encoders):
                    display_name = base_name.replace('_', '-').title()
                    base_color = colors.get(base_name, 'gray')
                    
                    # Collect data for this base encoder at all weights
                    x_weights = [0.0]  # Start with baseline (w=0)
                    y_values = [stats[base_name]['mean_best_y']]  # Start with baseline
                    error_bars = [stats[base_name]['std_best_y']]  # Start with baseline
                    
                    for weight in sorted(weight_variants[base_name]):
                        weight_name = f"{base_name}_w{weight}"
                        if weight_name in stats:
                            x_weights.append(weight)
                            y_values.append(stats[weight_name]['mean_best_y'])
                            error_bars.append(stats[weight_name]['std_best_y'])
                    
                    # Plot line with error bars
                    plt.errorbar(
                        x_weights, y_values, yerr=error_bars,
                        marker='o', markersize=8, linewidth=2,
                        color=base_color, label=display_name
                    )
                
                plt.xlabel('Projection Weight (α)', fontsize=14)
                plt.ylabel('Mean Best Value', fontsize=14)
                plt.title('Effect of Projection Weight on Best Value', fontsize=16)
                plt.grid(True, linestyle='--', alpha=0.7)
                plt.legend(fontsize=12)
                
                # Second subplot: Compare distances to optimum across weights
                plt.subplot(2, 2, 2)
                
                for base_name in sorted(base_encoders):
                    display_name = base_name.replace('_', '-').title()
                    base_color = colors.get(base_name, 'gray')
                    
                    # Collect data for this base encoder at all weights
                    x_weights = [0.0]  # Start with baseline
                    distances = [stats[base_name]['mean_distance']]  # Start with baseline
                    
                    for weight in sorted(weight_variants[base_name]):
                        weight_name = f"{base_name}_w{weight}"
                        if weight_name in stats:
                            x_weights.append(weight)
                            distances.append(stats[weight_name]['mean_distance'])
                    
                    # Plot line
                    plt.plot(
                        x_weights, distances,
                        marker='o', markersize=8, linewidth=2,
                        color=base_color, label=display_name
                    )
                
                plt.xlabel('Projection Weight (α)', fontsize=14)
                plt.ylabel('Distance to True Optimum', fontsize=14)
                plt.title('Effect of Projection Weight on Distance to Optimum', fontsize=16)
                plt.grid(True, linestyle='--', alpha=0.7)
                plt.legend(fontsize=12)
                
                # Third subplot: Compare projection distances across weights
                plt.subplot(2, 2, 3)
                
                for base_name in sorted(base_encoders):
                    display_name = base_name.replace('_', '-').title()
                    base_color = colors.get(base_name, 'gray')
                    
                    # Collect data for this base encoder at all weights
                    x_weights = [0.0]  # Start with baseline
                    proj_distances = [stats[base_name]['mean_projection_distance']]  # Start with baseline
                    
                    for weight in sorted(weight_variants[base_name]):
                        weight_name = f"{base_name}_w{weight}"
                        if weight_name in stats:
                            x_weights.append(weight)
                            proj_distances.append(stats[weight_name]['mean_projection_distance'])
                    
                    # Plot line
                    plt.plot(
                        x_weights, proj_distances,
                        marker='o', markersize=8, linewidth=2,
                        color=base_color, label=display_name
                    )
                
                plt.xlabel('Projection Weight (α)', fontsize=14)
                plt.ylabel('Average Projection Distance', fontsize=14)
                plt.title('Effect of Projection Weight on Projection Distances', fontsize=16)
                plt.grid(True, linestyle='--', alpha=0.7)
                plt.legend(fontsize=12)
                
                # Add explanatory text
                plt.figtext(
                    0.5, 0.02,
                    "Projection weight (α) controls the penalty applied to the objective function: O(x) - α * C(x)\n"
                    "where O(x) is the original objective and C(x) is the projection distance to the manifold.",
                    ha='center', fontsize=14, wrap=True
                )
                
                plt.tight_layout(rect=[0, 0.05, 1, 0.95])
                
                # Save the projection weight comparison plot
                plt.savefig('projection_weight_comparison.png', dpi=300)
                print("Projection weight comparison plot saved to projection_weight_comparison.png")
            
        except Exception as e:
            print(f"Error creating projection weight comparison plot: {str(e)}")
            import traceback
            traceback.print_exc()
        
    except Exception as e:
        print(f"Error creating multi-trial comparison plots: {str(e)}")
        import traceback
        traceback.print_exc()
    
    return all_trial_results, stats

if __name__ == "__main__":
    print("Testing Bayesian optimization with multiple encoders on a complex manifold")
    
    try:
        # Run comparison with higher dimensions and more complexity, over multiple trials
        all_results, stats = run_manifold_optimization_comparison(
            n_dims=25,         # 25D full space
            manifold_dims=6,   # 6D manifold
            discrete_dims=8,   # 8 discrete dimensions
            binary_dims=4,     # 4 binary dimensions (subset of discrete_dims)
            n_iterations=100,  # 100 iterations per trial
            n_trials=3,        # 3 trials for statistical analysis
            force_retrain=False  # Don't retrain encoders if saved versions exist
        )
        print(f"Encoder comparison completed")
        
    except Exception as e:
        print(f"Unhandled exception: {str(e)}")
        import traceback
        traceback.print_exc()
    
    print("All experiments completed")
    
    print("Script completed") 