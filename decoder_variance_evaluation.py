import torch
import numpy as np
import matplotlib.pyplot as plt
from encoder import DecoderEnsemble
from sklearn.neighbors import NearestNeighbors
import os
import time
from tqdm import tqdm
import seaborn as sns

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def generate_sample_manifold_data(n_samples=1000, n_dims=10, manifold_dims=2, seed=None):
    """
    Generate sample data that lies approximately on a manifold.
    
    Args:
        n_samples: Number of samples to generate
        n_dims: Number of dimensions in the full space
        manifold_dims: Number of dimensions in the manifold
        seed: Random seed for reproducibility
        
    Returns:
        data: Generated data of shape (n_samples, n_dims)
        latent_coords: Manifold coordinates of shape (n_samples, manifold_dims)
    """
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
    
    # Generate manifold coordinates
    latent_coords = torch.rand(n_samples, manifold_dims)
    
    # Create a random linear map from manifold to full space
    linear_map = torch.randn(manifold_dims, n_dims)
    
    # Map manifold coordinates to full space
    data = torch.matmul(latent_coords, linear_map)
    
    # Normalize to [0, 1] range
    data = (data - data.min(dim=0)[0]) / (data.max(dim=0)[0] - data.min(dim=0)[0])
    
    # Add a small amount of noise
    data = data + 0.01 * torch.randn_like(data)
    data = torch.clamp(data, 0.0, 1.0)
    
    return data, latent_coords

def generate_test_points(encoder, training_latents, n_points=200, n_dims=10, scale_factor=3.0):
    """
    Generate test points in the latent space with varying distances from training data.
    
    Args:
        encoder: Trained encoder
        training_latents: Latent representations of training data
        n_points: Number of test points to generate
        n_dims: Number of dimensions in the full space
        scale_factor: Controls how far test points can be from training data
        
    Returns:
        test_latents: Generated test points in latent space
        test_points: Decoded test points in input space
    """
    # Get latent dimension from encoder
    latent_dim = encoder.encoded_dim
    
    # Generate random points in latent space
    random_latents = torch.randn(n_points // 4, latent_dim, device=device)
    
    # Generate points close to training data
    idx = torch.randint(0, len(training_latents), (n_points // 4,))
    close_latents = training_latents[idx] + 0.1 * torch.randn(n_points // 4, latent_dim, device=device)
    
    # Generate points moderately far from training data
    idx = torch.randint(0, len(training_latents), (n_points // 4,))
    medium_latents = training_latents[idx] + 0.5 * torch.randn(n_points // 4, latent_dim, device=device)
    
    # Generate points very far from training data
    far_latents = scale_factor * torch.randn(n_points // 4, latent_dim, device=device)
    
    # Combine all latent points
    test_latents = torch.cat([random_latents, close_latents, medium_latents, far_latents], dim=0)
    
    # Decode the test points
    with torch.no_grad():
        test_points = encoder.decode(test_latents)
    
    return test_latents, test_points

def compute_nearest_neighbor_distances(test_latents, training_latents, k=3):
    """
    Compute the average distance to the k nearest neighbors in the training set.
    
    Args:
        test_latents: Test points in latent space
        training_latents: Training points in latent space
        k: Number of nearest neighbors to consider
        
    Returns:
        avg_distances: Average distance to k nearest neighbors for each test point
    """
    # Convert to numpy for scikit-learn
    test_np = test_latents.cpu().numpy()
    train_np = training_latents.cpu().numpy()
    
    # Use scikit-learn NearestNeighbors
    nbrs = NearestNeighbors(n_neighbors=k, algorithm='ball_tree').fit(train_np)
    distances, indices = nbrs.kneighbors(test_np)
    
    # Average distance to k nearest neighbors
    avg_distances = distances.mean(axis=1)
    
    return avg_distances

def compute_projection_distances(test_points, manifold_coords, linear_map):
    """
    Compute the distance from each test point to its projection on the manifold.
    
    Args:
        test_points: Test points in the input space
        manifold_coords: Original manifold coordinates used to generate training data
        linear_map: Linear map from manifold to full space
        
    Returns:
        projection_distances: Distance from each test point to its projection on the manifold
    """
    # Convert to numpy for computation
    test_points_np = test_points.cpu().numpy()
    manifold_coords_np = manifold_coords.cpu().numpy()
    linear_map_np = linear_map.cpu().numpy()
    
    # Number of test points
    n_test = test_points_np.shape[0]
    
    # Create a projection function that projects a point onto the manifold
    def project_to_manifold(x):
        """Project a point onto the manifold defined by the linear map."""
        # Use least squares to find the best manifold coordinates
        coeffs, residuals, _, _ = np.linalg.lstsq(linear_map_np.T, x, rcond=None)
        # Map back to full space
        projected = coeffs @ linear_map_np
        return projected, np.linalg.norm(x - projected)
    
    # Compute projection for each test point
    projection_distances = np.zeros(n_test)
    for i in range(n_test):
        _, distance = project_to_manifold(test_points_np[i])
        projection_distances[i] = distance
    
    return projection_distances

def main():
    # Parameters
    n_dims = 10
    latent_dim = 5
    n_training_samples = 1000
    n_test_samples = 200
    manifold_dims = 2
    
    # Directory for saving results
    os.makedirs("decoder_variance_results", exist_ok=True)
    
    # Generate training data
    print("Generating training data...")
    training_data, manifold_coords = generate_sample_manifold_data(
        n_samples=n_training_samples, 
        n_dims=n_dims, 
        manifold_dims=manifold_dims
    )
    
    # Create a random linear map from manifold to full space (same as in generate_sample_manifold_data)
    linear_map = torch.randn(manifold_dims, n_dims)
    
    # Create and train DecoderEnsemble
    encoder_filename = f"decoder_variance_results/decoder_ensemble_latent{latent_dim}_dim{n_dims}.pt"
    
    if os.path.exists(encoder_filename):
        print(f"Loading pre-trained DecoderEnsemble from {encoder_filename}")
        encoder = torch.load(encoder_filename)
    else:
        print("Training DecoderEnsemble...")
        encoder = DecoderEnsemble(
            input_dim=n_dims,
            latent_dim=latent_dim,
            n_decoders=5,
            hidden_dims=[64, 32],
            beta=0.1,
            train_iters=500,
            batch_size=64,
            learning_rate=1e-3,
            device=device
        )
        
        # Train the encoder
        encoder.fit(training_data.to(device))
        
        # Save the trained encoder
        torch.save(encoder, encoder_filename)
        print(f"Saved trained encoder to {encoder_filename}")
    
    # Encode training data to get training latent points
    print("Encoding training data...")
    with torch.no_grad():
        training_latents = encoder.encode(training_data.to(device))
    
    # Generate test points with varying distances from training data
    print("Generating test points...")
    test_latents, test_points = generate_test_points(
        encoder, 
        training_latents, 
        n_points=n_test_samples,
        n_dims=n_dims
    )
    
    # Compute decoder variance for test points
    print("Computing decoder variance...")
    with torch.no_grad():
        decoder_variances = encoder.decoder_variance(test_latents).cpu().numpy()
    
    # Compute average distance to k nearest neighbors
    print("Computing nearest neighbor distances...")
    k = 3  # Number of nearest neighbors to consider
    nn_distances = compute_nearest_neighbor_distances(test_latents, training_latents, k=k)
    
    # Compute projection distances to the manifold
    print("Computing projection distances to manifold...")
    projection_distances = compute_projection_distances(test_points, manifold_coords, linear_map)
    
    # Create the original plot with nearest neighbor distances
    plt.figure(figsize=(10, 8))
    
    # Create scatter plot with color based on point type
    point_types = ['Random', 'Close', 'Medium', 'Far']
    colors = ['blue', 'green', 'orange', 'red']
    
    # Get point type for each test point
    point_indices = np.arange(len(test_latents))
    point_type_indices = np.array([i // (n_test_samples // 4) for i in point_indices])
    
    # Create scatter plot
    for i, (ptype, color) in enumerate(zip(point_types, colors)):
        mask = point_type_indices == i
        plt.scatter(
            nn_distances[mask], 
            decoder_variances[mask], 
            alpha=0.7, 
            label=ptype,
            color=color,
            edgecolors='w',
            s=80
        )
    
    # Compute correlation
    correlation = np.corrcoef(nn_distances, decoder_variances)[0, 1]
    
    # Add trend line
    z = np.polyfit(nn_distances, decoder_variances, 1)
    p = np.poly1d(z)
    plt.plot(np.sort(nn_distances), p(np.sort(nn_distances)), 
             "k--", linewidth=2, 
             label=f"Trend Line (r={correlation:.3f})")
    
    # Compute and show the RMSE between scaled distance and variance
    # First scale distances to match variance range
    scaled_distances = (nn_distances - nn_distances.min()) / (nn_distances.max() - nn_distances.min())
    scaled_distances = scaled_distances * (decoder_variances.max() - decoder_variances.min()) + decoder_variances.min()
    rmse = np.sqrt(np.mean((scaled_distances - decoder_variances) ** 2))
    
    plt.title(f"Decoder Variance vs. Average Distance to {k} Nearest Training Points\nCorrelation: {correlation:.3f}, RMSE: {rmse:.3f}", fontsize=14)
    plt.xlabel(f"Average Distance to {k} Nearest Training Points", fontsize=12)
    plt.ylabel("Decoder Variance", fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=10)
    
    # Save the plot
    plt.tight_layout()
    plt.savefig("decoder_variance_results/decoder_variance_vs_distance.png", dpi=300)
    print("Plot saved to decoder_variance_results/decoder_variance_vs_distance.png")
    
    # Create the new plot with projection distances
    plt.figure(figsize=(10, 8))
    
    # Create scatter plot with color based on point type
    for i, (ptype, color) in enumerate(zip(point_types, colors)):
        mask = point_type_indices == i
        plt.scatter(
            projection_distances[mask], 
            decoder_variances[mask], 
            alpha=0.7, 
            label=ptype,
            color=color,
            edgecolors='w',
            s=80
        )
    
    # Compute correlation for projection distances
    proj_correlation = np.corrcoef(projection_distances, decoder_variances)[0, 1]
    
    # Add trend line for projection distances
    z_proj = np.polyfit(projection_distances, decoder_variances, 1)
    p_proj = np.poly1d(z_proj)
    plt.plot(np.sort(projection_distances), p_proj(np.sort(projection_distances)), 
             "k--", linewidth=2, 
             label=f"Trend Line (r={proj_correlation:.3f})")
    
    # Compute and show the RMSE between scaled projection distance and variance
    scaled_proj_distances = (projection_distances - projection_distances.min()) / (projection_distances.max() - projection_distances.min())
    scaled_proj_distances = scaled_proj_distances * (decoder_variances.max() - decoder_variances.min()) + decoder_variances.min()
    proj_rmse = np.sqrt(np.mean((scaled_proj_distances - decoder_variances) ** 2))
    
    plt.title(f"Decoder Variance vs. Projection Distance to Manifold\nCorrelation: {proj_correlation:.3f}, RMSE: {proj_rmse:.3f}", fontsize=14)
    plt.xlabel("Projection Distance to Manifold", fontsize=12)
    plt.ylabel("Decoder Variance", fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=10)
    
    # Save the plot
    plt.tight_layout()
    plt.savefig("decoder_variance_results/decoder_variance_vs_projection_distance.png", dpi=300)
    print("Plot saved to decoder_variance_results/decoder_variance_vs_projection_distance.png")
    
    # Create a hexbin plot for projection distances
    plt.figure(figsize=(10, 8))
    hb = plt.hexbin(projection_distances, decoder_variances, gridsize=30, cmap='viridis', mincnt=1)
    cb = plt.colorbar(hb, label='Count')
    plt.title(f"Decoder Variance vs. Projection Distance (Hexbin Plot)\nCorrelation: {proj_correlation:.3f}", fontsize=14)
    plt.xlabel("Projection Distance to Manifold", fontsize=12)
    plt.ylabel("Decoder Variance", fontsize=12)
    plt.grid(True, alpha=0.3)
    
    # Add trend line
    plt.plot(np.sort(projection_distances), p_proj(np.sort(projection_distances)), 
             "r--", linewidth=2, 
             label=f"Trend Line (r={proj_correlation:.3f})")
    plt.legend(fontsize=10)
    
    # Save the hexbin plot
    plt.tight_layout()
    plt.savefig("decoder_variance_results/decoder_variance_vs_projection_distance_hexbin.png", dpi=300)
    print("Hexbin plot saved to decoder_variance_results/decoder_variance_vs_projection_distance_hexbin.png")
    
    # Additional plot: Create a joint distribution plot for projection distances
    plt.figure(figsize=(10, 10))
    g = sns.jointplot(
        x=projection_distances, 
        y=decoder_variances, 
        kind="scatter",
        height=8, 
        ratio=5,
        marginal_kws=dict(bins=30, color="gray"),
        joint_kws=dict(alpha=0.7, s=50)
    )
    
    # Add regression line to joint plot
    g.ax_joint.plot(np.sort(projection_distances), p_proj(np.sort(projection_distances)), 
             "r--", linewidth=2)
    
    # Add text annotation with correlation
    g.ax_joint.annotate(
        f"Correlation: {proj_correlation:.3f}\nRMSE: {proj_rmse:.3f}",
        xy=(0.05, 0.95),
        xycoords="axes fraction",
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8),
        fontsize=12,
        ha="left",
        va="top"
    )
    
    g.ax_joint.set_title("Decoder Variance vs. Projection Distance to Manifold", fontsize=14)
    g.ax_joint.set_xlabel("Projection Distance to Manifold", fontsize=12)
    g.ax_joint.set_ylabel("Decoder Variance", fontsize=12)
    g.ax_joint.grid(True, alpha=0.3)
    
    # Save the joint plot
    plt.tight_layout()
    plt.savefig("decoder_variance_results/decoder_variance_vs_projection_distance_joint_plot.png", dpi=300)
    print("Joint plot saved to decoder_variance_results/decoder_variance_vs_projection_distance_joint_plot.png")

if __name__ == "__main__":
    main() 