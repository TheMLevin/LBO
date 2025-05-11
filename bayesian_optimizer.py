import torch
import botorch
from botorch.models import SingleTaskGP
from botorch.models.model import Model
from botorch.acquisition import LogExpectedImprovement, ExpectedImprovement, UpperConfidenceBound
from botorch.acquisition.acquisition import AcquisitionFunction
from botorch.optim import optimize_acqf
from gpytorch.kernels import ScaleKernel, RBFKernel, MaternKernel
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.means import ConstantMean, ZeroMean
from typing import Optional, Tuple, List, Union, Callable, Dict, Any
import numpy as np
from encoder import Encoder, IdentityEncoder, PairedEnsembleVAEEncoder
import time
import logging

# Refactored acquisition function that incorporates any weighting function
class WeightedLogEI(AcquisitionFunction):
    """
    Log Expected Improvement weighted by a custom weighting function.
    This can be used to incorporate various types of uncertainty or 
    preference information into the acquisition function.
    By default, points with high weight values are downweighted.
    """
    def __init__(
        self, 
        model: Model, 
        best_f: torch.Tensor, 
        weight_fn: Callable,
        weight_param: float = 0.1,
        maximize: bool = True,
        downweight_high_values: bool = True
    ):
        """
        Args:
            model: The surrogate model
            best_f: The best objective value observed so far
            weight_fn: Function that computes weights for points (e.g., uncertainty, distance)
            weight_param: Weight factor applied to weighting term (higher = stronger effect)
            maximize: Whether to maximize or minimize the objective
            downweight_high_values: If True, points with high weight values are penalized; 
                                   if False, such points are favored
        """
        super().__init__(model)
        self.model = model
        self.best_f = best_f
        self.weight_fn = weight_fn
        self.weight_param = weight_param
        self.maximize = maximize
        self.downweight_high_values = downweight_high_values
        # Use standard LogExpectedImprovement for better stability
        self.ei = LogExpectedImprovement(model, best_f, maximize=maximize)
        
    def forward(self, X: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Calculate the weighted acquisition value.
        
        Args:
            X: Points at which to evaluate the acquisition function
            **kwargs: Additional keyword arguments that might be passed by botorch
                    (ignored to maintain compatibility)
            
        Returns:
            Acquisition values
        """
        # Get base expected improvement - forward any kwargs to the base EI function
        ei_values = self.ei(X, **kwargs)
        
        # Calculate weights (handle batch dimension properly)
        if X.dim() > 2:
            # For batch evaluation, we operate on the base X without batch dimension
            X_eval = X.squeeze(1)
        else:
            X_eval = X
        
        # Get weights from the provided weighting function
        try:
            weights = self.weight_fn(X_eval)
            
            # For debugging - print weight statistics
            # if weights.numel() < 20:  # Only print for small batches
            #     print(f"Weights: min={weights.min().item():.4f}, "
            #           f"max={weights.max().item():.4f}, "
            #           f"mean={weights.mean().item():.4f}")
            
            # Handle NaN or infinity values (replace with mean)
            if torch.isnan(weights).any() or torch.isinf(weights).any():
                print("Warning: NaN or Inf values in weights, replacing with mean")
                valid_mask = ~(torch.isnan(weights) | torch.isinf(weights))
                if valid_mask.sum() > 0:
                    valid_mean = weights[valid_mask].mean()
                    weights = torch.where(valid_mask, weights, valid_mean)
                else:
                    # If all invalid, use default value
                    weights = torch.ones_like(weights) * 0.5
        except Exception as e:
            print(f"Error computing weights: {str(e)}")
            # Fallback to neutral weights (doesn't affect EI)
            weights = torch.ones_like(ei_values) * 0.5
        
        # Match dimensions with ei_values if needed
        if weights.shape != ei_values.shape:
            if weights.dim() == 1 and ei_values.dim() == 2:
                weights = weights.unsqueeze(-1)
            elif weights.dim() == 2 and ei_values.dim() == 1:
                weights = weights.squeeze(-1)
        
        # Weight ei by the weighting function
        if self.downweight_high_values:
            # Penalize high weight points by multiplying EI by (1 - normalized_weights)
            # This reduces acquisition value for points with high weight values
            weighted_values = ei_values - weights * self.weight_param
            # print(f"Weight factor: {weight_factor}", f"EI values: {ei_values}", f"Weighted values: {weighted_values}")
        else:
            # Favor high weight points by adding to EI
            weighted_values = ei_values + torch.log(self.weight_param * weights)
        
        return weighted_values


class CustomMeanWrapper:
    """
    Wrapper for mean modules that allows using original points instead of encoded points.
    """
    def __init__(self, mean_module, encoder, use_original_points=False):
        self.mean_module = mean_module
        self.encoder = encoder
        self.use_original_points = use_original_points
        self.original_points_map = {}  # Maps encoded points to original points
    
    def register_points(self, encoded_points, original_points):
        """Register mapping between encoded and original points."""
        if self.use_original_points:
            for i in range(encoded_points.shape[0]):
                # Use the encoded point as a key (convert to tuple for hashability)
                key = tuple(encoded_points[i].detach().cpu().numpy().tolist())
                self.original_points_map[key] = original_points[i]
    
    def __call__(self, x):
        """
        Forward pass through the mean module.
        
        Args:
            x (torch.Tensor): Input points (encoded)
            
        Returns:
            torch.Tensor: Mean values
        """
        if not self.use_original_points:
            # Use encoded points directly
            return self.mean_module(x)
        
        # Try to find original points for each encoded point
        original_x = []
        for i in range(x.shape[0]):
            key = tuple(x[i].detach().cpu().numpy().tolist())
            if key in self.original_points_map:
                original_x.append(self.original_points_map[key])
            else:
                # If not found, decode the point
                original_x.append(self.encoder.decode(x[i].unsqueeze(0)).squeeze(0))
        
        # Stack original points and pass to mean module
        original_x = torch.stack(original_x)
        return self.mean_module(original_x)


class BayesianOptimizer:
    def __init__(
        self,
        bounds: Optional[torch.Tensor] = None,
        n_initial_points: int = 5,
        kernel_type: str = "matern",
        nu: float = 2.5,
        candidates: Optional[torch.Tensor] = None,
        initial_points: Optional[torch.Tensor] = None,
        initial_values: Optional[torch.Tensor] = None,
        mean_module: Optional[ConstantMean] = None,
        encoder: Optional[Encoder] = None,
        use_original_points_for_mean: bool = False,
        acqf: AcquisitionFunction = None,
        constraint_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,  # Constraint function
        projection_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,  # Projection function
        auto_normalize: bool = True,  # Whether to automatically normalize input data
    ):
        """
        Initialize the Bayesian Optimizer.
        
        Args:
            bounds (Optional[torch.Tensor]): Bounds for the optimization space, shape (dim, 2).
                                            Required for continuous optimization, optional for candidate-based.
            n_initial_points (int): Number of initial random points to sample
            kernel_type (str): Type of kernel to use ('rbf' or 'matern')
            nu (float): Nu parameter for Matern kernel (only used if kernel_type='matern')
            candidates (Optional[torch.Tensor]): Candidate points to optimize over
            initial_points (Optional[torch.Tensor]): Initial training points, shape (n, dim)
            initial_values (Optional[torch.Tensor]): Values for initial training points, shape (n, 1)
            mean_module (Optional[ConstantMean]): Prior mean module, defaults to ZeroMean
            encoder (Optional[Encoder]): Encoder for transforming input data, defaults to identity
            use_original_points_for_mean (bool): Whether to pass original points to the mean module
            variance_weight (float): Weight for decoder variance term in acquisition function
            constraint_fn (Optional[Callable]): Function that returns True for valid points, False otherwise
            projection_fn (Optional[Callable]): Function that projects invalid points to valid space
            auto_normalize (bool): Whether to automatically normalize input data
        """
        self.candidates = candidates
        self.use_candidates = candidates is not None
        self.use_original_points_for_mean = use_original_points_for_mean
        self.constraint_fn = constraint_fn
        self.projection_fn = projection_fn
        self.auto_normalize = auto_normalize
        self.x_min = None  # For normalization
        self.x_max = None  # For normalization
        
        # Initialize a list to store projection distances
        self.projection_distances = []
        
        # Determine input dimension
        if self.use_candidates:
            self.input_dim = candidates.shape[1]
        elif bounds is not None:
            self.input_dim = bounds.shape[0]
        elif initial_points is not None:
            self.input_dim = initial_points.shape[1]
        else:
            raise ValueError("Cannot determine input dimension. Provide bounds, candidates, or initial_points.")
        
        # Set up encoder
        self.encoder = encoder if encoder is not None else IdentityEncoder(self.input_dim)
        
        # Verify encoder dimensions
        if self.encoder.input_dim != self.input_dim:
            raise ValueError(f"Encoder input dimension ({self.encoder.input_dim}) must match problem dimension ({self.input_dim})")
        
        # The dimension used by the GP model is the encoded dimension
        self.dim = self.encoder.encoded_dim
        
        # Set up mean module
        self.mean_module = mean_module if mean_module is not None else ZeroMean()
        self.mean_wrapper = CustomMeanWrapper(
            self.mean_module, 
            self.encoder, 
            use_original_points=use_original_points_for_mean
        )
        
        # If using candidates, encode them
        if self.use_candidates:
            self.encoded_candidates = self.encoder.encode(candidates)
            # Register candidates with mean wrapper
            self.mean_wrapper.register_points(self.encoded_candidates, candidates)
        else:
            self.encoded_candidates = None
        
        # Handle bounds
        self.bounds = bounds
        if self.use_candidates:
            # If bounds not provided but using candidates, create dummy bounds for acquisition optimization
            if bounds is None:
                if self.encoder.input_dim == self.encoder.encoded_dim:
                    # If encoder doesn't change dimension, infer bounds from candidates
                    min_vals, _ = torch.min(candidates, dim=0)
                    max_vals, _ = torch.max(candidates, dim=0)
                    buffer = 0.1 * (max_vals - min_vals)
                    self.bounds = torch.stack([min_vals - buffer, max_vals + buffer], dim=1)
                else:
                    # If encoder changes dimension, infer bounds from encoded candidates
                    min_vals, _ = torch.min(self.encoded_candidates, dim=0)
                    max_vals, _ = torch.max(self.encoded_candidates, dim=0)
                    buffer = 0.1 * (max_vals - min_vals)
                    self.encoded_bounds = torch.stack([min_vals - buffer, max_vals + buffer], dim=1)
        else:
            # For continuous optimization, bounds are required
            if bounds is None:
                raise ValueError("Bounds must be provided for continuous optimization")
            
            # If encoder changes dimension, we need to compute encoded bounds
            if self.encoder.input_dim != self.encoder.encoded_dim:
                # Sample points along the bounds to estimate encoded bounds
                n_samples = 100
                samples = []
                for d in range(self.input_dim):
                    # Sample along each dimension
                    for bound_idx in range(2):  # 0: lower, 1: upper
                        points = torch.zeros((n_samples, self.input_dim))
                        # Set all dimensions to random values within bounds
                        for dim in range(self.input_dim):
                            points[:, dim] = torch.rand(n_samples) * (bounds[dim, 1] - bounds[dim, 0]) + bounds[dim, 0]
                        # Override the current dimension with the bound value
                        points[:, d] = bounds[d, bound_idx]
                        samples.append(points)
                
                # Combine all samples and encode
                boundary_samples = torch.cat(samples, dim=0)
                encoded_samples = self.encoder.encode(boundary_samples)
                
                # Compute bounds of encoded samples
                min_vals, _ = torch.min(encoded_samples, dim=0)
                max_vals, _ = torch.max(encoded_samples, dim=0)
                buffer = 0.1 * (max_vals - min_vals)
                self.encoded_bounds = torch.stack([min_vals - buffer, max_vals + buffer], dim=1)
        
        self.n_initial_points = n_initial_points
        self.kernel_type = kernel_type
        self.nu = nu
        
        # Initialize training data
        if initial_points is not None and initial_values is not None:
            if initial_points.shape[0] != initial_values.shape[0]:
                raise ValueError("Number of initial points and values must match")
            if initial_values.dim() == 1:
                initial_values = initial_values.unsqueeze(-1)
            
            # Encode initial points
            encoded_initial_points = self.encoder.encode(initial_points)
            
            # Register with mean wrapper
            self.mean_wrapper.register_points(encoded_initial_points, initial_points)
            
            self.train_X = encoded_initial_points
            self.train_Y = initial_values
            
            # Store original points for reference
            self.original_train_X = initial_points
        else:
            self.train_X = None
            self.train_Y = None
            self.original_train_X = None
        
        self.model = None

    def update_candidates(self, new_candidates: torch.Tensor, replace: bool = False):
        """
        Update the candidate set by either replacing or adding to the existing candidates.
        
        Args:
            new_candidates (torch.Tensor): New candidate points to add or replace existing ones
            replace (bool): If True, replace existing candidates; if False, append to existing candidates
        """
        if new_candidates.shape[1] != self.input_dim:
            raise ValueError(f"New candidates must have dimension {self.input_dim}, got {new_candidates.shape[1]}")
        
        # Encode new candidates
        encoded_new_candidates = self.encoder.encode(new_candidates)
        
        # Register with mean wrapper
        self.mean_wrapper.register_points(encoded_new_candidates, new_candidates)
        
        if not self.use_candidates and replace:
            # Switching from continuous to candidate-based optimization
            self.use_candidates = True
            self.candidates = new_candidates
            self.encoded_candidates = encoded_new_candidates
        elif not self.use_candidates and not replace:
            # Initialize candidates if none exist
            self.use_candidates = True
            self.candidates = new_candidates
            self.encoded_candidates = encoded_new_candidates
        elif replace:
            # Replace existing candidates
            self.candidates = new_candidates
            self.encoded_candidates = encoded_new_candidates
        else:
            # Append to existing candidates
            self.candidates = torch.cat([self.candidates, new_candidates], dim=0)
            self.encoded_candidates = torch.cat([self.encoded_candidates, encoded_new_candidates], dim=0)
            
        # Update bounds if they were inferred from candidates
        if self.bounds is None and (not replace or not self.use_candidates):
            # Recalculate bounds based on all candidates
            min_vals, _ = torch.min(self.encoded_candidates, dim=0)
            max_vals, _ = torch.max(self.encoded_candidates, dim=0)
            buffer = 0.1 * (max_vals - min_vals)
            self.encoded_bounds = torch.stack([min_vals - buffer, max_vals + buffer], dim=1)
            
        return self.candidates

    def _normalize_data(self, X: torch.Tensor) -> torch.Tensor:
        """
        Min-max normalize the data to [0, 1] range.
        If this is the first call, store min/max values.
        
        Args:
            X (torch.Tensor): Input data to normalize
            
        Returns:
            torch.Tensor: Normalized data
        """
        # Debug: Check for NaNs in input
        if torch.isnan(X).any():
            print("WARNING: NaN values detected in data to normalize!")
            print(f"Number of NaN values: {torch.isnan(X).sum().item()}")
            # Replace NaNs with zeros
            X_fixed = X.clone()
            X_fixed[torch.isnan(X_fixed)] = 0.0
            print("Replaced NaN values with 0.0 for normalization")
            X = X_fixed
        
        if self.x_min is None or self.x_max is None:
            # First time, compute and store min/max values
            self.x_min, _ = torch.min(X, dim=0)
            self.x_max, _ = torch.max(X, dim=0)
            
            # Handle dimensions with no variation (min=max)
            eps = 1e-8
            self.effective_range = torch.clamp(self.x_max - self.x_min, min=eps)
        
        # Apply normalization
        X_normalized = (X - self.x_min) / self.effective_range
        return X_normalized
        
    def _denormalize_data(self, X_normalized: torch.Tensor) -> torch.Tensor:
        """
        Denormalize data from [0, 1] range back to original scale.
        
        Args:
            X_normalized (torch.Tensor): Normalized data
            
        Returns:
            torch.Tensor: Data in original scale
        """
        if self.x_min is None or self.x_max is None:
            raise ValueError("Cannot denormalize data without normalization parameters")
            
        return X_normalized * self.effective_range + self.x_min

    def _init_model(self) -> SingleTaskGP:
        """Initialize the Gaussian Process model with the specified kernel."""
        # Check for -inf values in training data and replace them
        if (self.train_Y == float('-inf')).any():
            print("WARNING: -inf values found in training data")
            # Find the minimum finite value
            finite_values = self.train_Y[self.train_Y > float('-inf')]
            if len(finite_values) > 0:
                min_value = finite_values.min().item()
                # Replace -inf with something slightly lower than min value
                inf_mask = (self.train_Y == float('-inf')).squeeze(-1)
                self.train_Y[inf_mask] = torch.tensor([min_value - 10.0]).unsqueeze(-1)
                print(f"Replaced -inf values with {min_value - 10.0}")
            else:
                # All values are -inf, set them to a low value
                self.train_Y[:] = torch.tensor([-100.0]).unsqueeze(-1)
                print("All training values are -inf, replaced with -100.0")
        
        if self.kernel_type.lower() == "rbf":
            base_kernel = RBFKernel(ard_num_dims=self.dim)
        elif self.kernel_type.lower() == "matern":
            base_kernel = MaternKernel(nu=self.nu, ard_num_dims=self.dim)
        else:
            raise ValueError(f"Unknown kernel type: {self.kernel_type}")
            
        kernel = ScaleKernel(base_kernel)
        
        # Initialize lengthscales to positive values
        if hasattr(base_kernel, 'lengthscale'):
            base_kernel.lengthscale = torch.ones_like(base_kernel.lengthscale) * 0.5
        
        # Initialize outputscale to a positive value
        kernel.outputscale = torch.tensor(1.0)
        
        # Normalize the training data if auto_normalize is enabled
        train_X = self.train_X
        if self.auto_normalize:
            train_X = self._normalize_data(self.train_X)
        
        # Store normalized training data for use in optimization
        self.normalized_train_X = train_X
            
        # Create the model with our mean wrapper
        model = SingleTaskGP(
            train_X.double(), 
            self.train_Y.double(),
            covar_module=kernel,
            mean_module=self.mean_wrapper
        )
        
        return model

    def _get_initial_points(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate initial points either randomly or from candidates."""
        if self.use_candidates:
            indices = torch.randperm(len(self.candidates))[:self.n_initial_points]
            return self.encoded_candidates[indices], self.candidates[indices]
        else:
            # Generate random points in the original space
            original_points = torch.rand(self.n_initial_points, self.input_dim) * (
                self.bounds[:, 1] - self.bounds[:, 0]
            ) + self.bounds[:, 0]
            
            # Encode the points
            encoded_points = self.encoder.encode(original_points)
            
            # Register with mean wrapper
            self.mean_wrapper.register_points(encoded_points, original_points)
            
            return encoded_points, original_points

    def project_to_constraints(self, x: torch.Tensor) -> torch.Tensor:
        """
        Project points to satisfy constraints.
        
        Args:
            x (torch.Tensor): Points to project
            
        Returns:
            torch.Tensor: Projected points that satisfy constraints
        """
        # If no constraint function or projection function, return original points
        if self.constraint_fn is None or self.projection_fn is None:
            return x
        
        # Check which points satisfy constraints
        valid = self.constraint_fn(x)
        
        # If all points are valid, return original points
        if valid.all():
            return x
            
        # Create result tensor
        result = x.clone()
        
        # Project invalid points
        if (~valid).any():
            invalid_idx = torch.where(~valid)[0]
            result[invalid_idx] = self.projection_fn(x[invalid_idx])
            
        return result

    def optimize(
        self,
        objective_function: Callable[[torch.Tensor], torch.Tensor],
        n_iterations: int = 20,
        use_constraints: bool = True,  # Whether to use constraints/projection
        acq_func_maker = None,
        custom_projection_fn = None  # Custom projection function for tracking
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[float]]:
        """
        Run the Bayesian optimization loop.
        
        Args:
            objective_function: Function to optimize (operates in original space)
            n_iterations: Number of optimization iterations
            exploration_weight: Weight for exploration in acquisition function
            downweight_uncertainty: If True, downweight points with high decoder variance;
                                   if False, upweight such points
            use_constraints: Whether to use constraints/projection functions
            custom_projection_fn: A custom projection function that also tracks projection distance
            
        Returns:
            Tuple of (best_x, best_y, cumulative_regret, projection_distances) in the original space
        """
        # Reset projection distances list at the start of optimization
        self.projection_distances = []
        
        # Define a projection function wrapper that tracks distances
        def projection_tracker(x):
            original_x = x.clone()
            
            # Apply the actual projection function
            if custom_projection_fn is not None:
                projected_x = custom_projection_fn(x)
            elif self.projection_fn is not None:
                projected_x = self.projection_fn(x)
            else:
                projected_x = x  # No projection if no function provided
                
            # Calculate distance between original and projected point
            distance = torch.norm(original_x - projected_x).item()
            self.projection_distances.append(distance)
            
            return projected_x
        
        # Initialize if no data exists
        if self.train_X is None:
            encoded_points, original_points = self._get_initial_points()
            
            # Project initial points to satisfy constraints if needed
            if use_constraints and self.constraint_fn is not None:
                # Use the projection tracker for initial points
                original_points = projection_tracker(original_points)
                
                # Re-encode the projected points
                encoded_points = self.encoder.encode(original_points)
                # Register with mean wrapper
                self.mean_wrapper.register_points(encoded_points, original_points)
            
            # Debug: Print original points to check for issues
            print(f"Generated {len(original_points)} initial points for evaluation")
            
            # Evaluate objective function
            Y = torch.tensor([objective_function(x) for x in original_points]).unsqueeze(-1)
            
            # Debug: Check for NaNs in Y
            if torch.isnan(Y).any():
                print(f"WARNING: NaN values detected in initial evaluations!")
                print(f"Number of NaN values: {torch.isnan(Y).sum().item()}")
                # Print indices of NaN values
                nan_indices = torch.where(torch.isnan(Y.squeeze()))[0]
                print(f"Indices with NaN values: {nan_indices.tolist()}")
                # Print the corresponding points
                for idx in nan_indices:
                    print(f"Point {idx} (producing NaN): {original_points[idx]}")
                # Replace NaNs with -inf
                Y[torch.isnan(Y)] = float('-inf')
                print("Replaced NaN values with -inf")
            
            self.train_X = encoded_points
            self.train_Y = Y
            self.original_train_X = original_points
        
        best_value = float("-inf")
        best_x = None
        cumulative_regret = torch.zeros(n_iterations + 1)
        current_best = self.train_Y.max().item()
        # Add initial regret
        cumulative_regret[0] = current_best
        
        for i in range(n_iterations):
            # Debug: Check for NaNs in training data before model initialization
            if torch.isnan(self.train_X).any():
                print(f"WARNING: NaN values in train_X at iteration {i}")
                print(f"Number of NaN values: {torch.isnan(self.train_X).sum().item()}")
                nan_indices = torch.where(torch.isnan(self.train_X))[0]
                print(f"Indices with NaN values: {nan_indices.tolist()}")
                # Remove NaNs from train_X and corresponding entries from train_Y
                mask = ~torch.isnan(self.train_X).any(dim=1)
                self.train_X = self.train_X[mask]
                self.train_Y = self.train_Y[mask]
                print("Removed NaN values from train_X and corresponding entries from train_Y")
            
            if torch.isnan(self.train_Y).any() or (self.train_Y == float('-inf')).any():
                print(f"WARNING: NaN or -inf values in train_Y at iteration {i}")
                print(f"Number of NaN or -inf values: {(torch.isnan(self.train_Y) | (self.train_Y == float('-inf'))).sum().item()}")
                nan_indices = torch.where(torch.isnan(self.train_Y) | (self.train_Y == float('-inf')))[0]
                print(f"Indices with NaN or -inf values: {nan_indices.tolist()}")
                # Remove NaNs and -inf from train_Y and corresponding entries from train_X
                mask = ~(torch.isnan(self.train_Y) | (self.train_Y == float('-inf')))
                self.train_Y = self.train_Y[mask]
                self.train_X = self.train_X[mask]
                print("Removed NaN and -inf values from train_Y and corresponding entries from train_X")
            
            # Fit GP model
            self.model = self._init_model()
            mll = ExactMarginalLogLikelihood(self.model.likelihood, self.model)
            
            # Fit the model
            self.model.train()
            self.model.likelihood.train()
            
            # Use Adam optimizer with parameter constraints
            optimizer = torch.optim.Adam([
                {'params': self.model.parameters(), 'lr': 0.1}
            ])
            
            # Training loop
            n_iter = 50
            for j in range(n_iter):
                optimizer.zero_grad()
                output = self.model(self.normalized_train_X if self.auto_normalize else self.train_X)
                loss = -mll(output, self.train_Y.squeeze(-1))
                loss.backward()
                
                # Apply constraints before step
                for param in self.model.parameters():
                    if param.grad is not None:
                        param.data = torch.clamp(param.data, min=1e-4)
                
                optimizer.step()
                
                # Apply constraints after step
                for param in self.model.parameters():
                    param.data = torch.clamp(param.data, min=1e-4)
            
            # Set the model to eval mode
            self.model.eval()
            self.model.likelihood.eval()
            
            # Create acquisition function
            best_f = self.train_Y.max()
            # Determine which acquisition function to use
            if acq_func_maker is None:
                acq_func = LogExpectedImprovement(
                    model=self.model,
                    best_f=best_f,
                    maximize=True
                )
            else:
                acq_func = acq_func_maker(
                    model=self.model,
                    best_f=best_f,
                    maximize=True
                )
                
            # Determine bounds for optimization
            if self.use_candidates:
                # For candidate-based optimization, get next candidate
                next_encoded_x = self._candidate_acquisition_optimization(acq_func)
            else:
                # For continuous optimization
                # Handle bounds correctly
                if hasattr(self, 'encoded_bounds'):
                    bounds_to_use = self.encoded_bounds
                else:
                    bounds_to_use = self.bounds
                
                # Adjust bounds for normalized space if needed
                if self.auto_normalize:
                    bounds_to_use = self._normalize_bounds(bounds_to_use)
                
                # Standard acquisition optimization
                try:
                    next_encoded_x, _ = optimize_acqf(
                        acq_function=acq_func,
                        bounds=bounds_to_use.t(),  # Transpose to match optimize_acqf expectations
                        q=1,
                        num_restarts=10,
                        raw_samples=100,
                    )
                except Exception as e:
                    raise e
            
            # Denormalize if needed
            if self.auto_normalize:
                next_encoded_x_orig = self._denormalize_data(next_encoded_x)
            else:
                next_encoded_x_orig = next_encoded_x
            
            # Decode back to original space using the denormalized encoded point
            next_original_x = self.encoder.decode(next_encoded_x_orig)
            
            # Project to satisfy constraints if needed
            if use_constraints:
                # Use the projection tracker instead of direct projection function
                next_original_x = projection_tracker(next_original_x)
                
                # Re-encode the projected point
                next_encoded_x_orig = self.encoder.encode(next_original_x)
                # Re-normalize if needed
                if self.auto_normalize:
                    next_encoded_x = self._normalize_data(next_encoded_x_orig)
                else:
                    next_encoded_x = next_encoded_x_orig
            
            # Register with mean wrapper
            self.mean_wrapper.register_points(next_encoded_x_orig, next_original_x)
            
            # Evaluate and update
            next_y = objective_function(next_original_x).unsqueeze(-1)
            
            # Update with unnormalized data for next iteration
            self.train_X = torch.cat([self.train_X, next_encoded_x_orig if self.auto_normalize else next_encoded_x])
            self.train_Y = torch.cat([self.train_Y, next_y])
            
            # Update original points
            if self.original_train_X is None:
                self.original_train_X = next_original_x
            else:
                self.original_train_X = torch.cat([self.original_train_X, next_original_x])
            
            # Update best observed value
            if next_y > best_value:
                best_value = next_y.item()
                best_x = next_original_x
            
            # Update cumulative regret
            current_best = max(current_best, next_y.item())
            cumulative_regret[i+1] = (i + 1) * current_best - self.train_Y[:i+2].sum().item()
        
        return best_x, torch.tensor(best_value), cumulative_regret, self.projection_distances

    def get_current_best(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return the current best observation in the original space."""
        if self.train_Y is None:
            raise ValueError("No observations available yet")
        
        best_idx = self.train_Y.argmax()
        
        # If we have original points stored, use them
        if self.original_train_X is not None:
            return self.original_train_X[best_idx].unsqueeze(0), self.train_Y[best_idx]
        
        # Otherwise, decode the encoded point
        best_encoded_x = self.train_X[best_idx]
        best_original_x = self.encoder.decode(best_encoded_x.unsqueeze(0)).squeeze(0)
        
        return best_original_x.unsqueeze(0), self.train_Y[best_idx]
    
    def get_reconstruction_loss(self, points: Optional[torch.Tensor] = None, reduction: str = 'mean') -> torch.Tensor:
        """
        Calculate the reconstruction loss for a set of points.
        
        Args:
            points (Optional[torch.Tensor]): Points to calculate loss for. If None, use training points.
            reduction (str): Reduction method ('mean', 'sum', or 'none')
            
        Returns:
            torch.Tensor: Reconstruction loss
        """
        if points is None:
            if self.original_train_X is None:
                raise ValueError("No training points available")
            points = self.original_train_X
        
        return self.encoder.reconstruction_loss(points, reduction=reduction) 

    def _normalize_bounds(self, bounds: torch.Tensor) -> torch.Tensor:
        """
        Normalize bounds for optimization in normalized space.
        
        Args:
            bounds: The bounds to normalize, shape (dim, 2)
            
        Returns:
            Normalized bounds
        """
        if self.x_min is None or self.x_max is None:
            # If normalization constants not set, use standard [0, 1] bounds
            return torch.stack([
                torch.zeros(bounds.shape[0]),
                torch.ones(bounds.shape[0])
            ], dim=1)
        
        # Normalize bounds using the same constants as for data
        norm_bounds = torch.zeros_like(bounds)
        
        # Normalize each dimension
        for d in range(bounds.shape[0]):
            # Skip dimensions with zero range
            if self.x_max[d] == self.x_min[d]:
                norm_bounds[d, 0] = 0.0
                norm_bounds[d, 1] = 1.0
            else:
                # Normalize bounds
                norm_bounds[d, 0] = (bounds[d, 0] - self.x_min[d]) / (self.x_max[d] - self.x_min[d])
                norm_bounds[d, 1] = (bounds[d, 1] - self.x_min[d]) / (self.x_max[d] - self.x_min[d])
        
        # Ensure bounds are within [0, 1]
        norm_bounds = torch.clamp(norm_bounds, 0.0, 1.0)
        
        return norm_bounds 

    def _candidate_acquisition_optimization(self, acq_func):
        """
        Optimize acquisition function over discrete set of candidates.
        
        Args:
            acq_func: Acquisition function
            
        Returns:
            Selected encoded candidate point
        """
        # Ensure encoded_candidates are available
        if self.encoded_candidates is None:
            raise ValueError("No candidates available for optimization")
        
        # Add batch dimension for acquisition function
        candidates_batch = self.encoded_candidates.unsqueeze(1)
        
        # Normalize candidates if needed
        if self.auto_normalize:
            candidates_batch = self._normalize_data(candidates_batch.squeeze(1)).unsqueeze(1)
        
        # Evaluate acquisition function at all candidates
        with torch.no_grad():
            acq_values = acq_func(candidates_batch).squeeze(1)
        
        # Find candidate with maximum acquisition value
        best_idx = torch.argmax(acq_values)
        
        return self.encoded_candidates[best_idx].unsqueeze(0)
