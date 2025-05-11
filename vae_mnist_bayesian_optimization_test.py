import torch
import torch.nn as nn
import torch.nn.functional as F
try:
    import torchvision
    from torchvision import datasets, transforms
except ImportError:
    # If torchvision is not found, raise a helpful error
    raise ImportError(
        "torchvision is required for this script. "
        "Please install it with 'pip install torchvision'."
    )

try:
    from botorch.acquisition.analytic import ExpectedImprovement
except ImportError:
    # If botorch is not found, raise a helpful error
    raise ImportError(
        "botorch is required for this script. "
        "Please install it with 'pip install botorch'."
    )

import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Subset
from encoder import VAEEncoder
from gradient_ascent_latent_optimization import SimpleConvVAE, BinaryMNISTClassifier
from bayesian_optimizer import BayesianOptimizer
from typing import List, Tuple, Dict, Optional, Callable
import time
import random
import os

# Set random seeds for reproducibility
torch.manual_seed(42)
random.seed(42)
np.random.seed(42)

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Using BinaryMNISTClassifier imported from gradient_ascent_latent_optimization.py
# Instead of the multiclass classifier that was defined here

def train_mnist_classifier(model, train_loader, valid_loader, epochs=10, lr=0.001):
    """Train the binary MNIST classifier."""
    criterion = nn.BCELoss()  # Binary cross entropy loss for binary classifier
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3, verbose=True
    )
    
    best_val_loss = float('inf')
    best_model_state = None
    target_digit = model.target_digit
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            # Convert targets to binary (1 for target_digit, 0 for others)
            binary_target = (target == target_digit).float().unsqueeze(1)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, binary_target)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            predicted = (output >= 0.5).float()  # Binary threshold at 0.5
            train_total += binary_target.size(0)
            train_correct += (predicted == binary_target).sum().item()
            
            if batch_idx % 100 == 0:
                print(f'Epoch: {epoch+1} | Batch: {batch_idx}/{len(train_loader)} | '
                      f'Loss: {train_loss/(batch_idx+1):.4f} | '
                      f'Acc: {100.*train_correct/train_total:.2f}%')
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for data, target in valid_loader:
                data, target = data.to(device), target.to(device)
                
                # Convert targets to binary (1 for target_digit, 0 for others)
                binary_target = (target == target_digit).float().unsqueeze(1)
                
                output = model(data)
                val_loss += criterion(output, binary_target).item()
                predicted = (output >= 0.5).float()  # Binary threshold at 0.5
                val_total += binary_target.size(0)
                val_correct += (predicted == binary_target).sum().item()
        
        val_loss /= len(valid_loader)
        val_accuracy = 100. * val_correct / val_total
        
        print(f'Epoch: {epoch+1} | Val Loss: {val_loss:.4f} | Val Acc: {val_accuracy:.2f}%')
        
        # Update learning rate
        scheduler.step(val_loss)
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict().copy()
    
    # Load best model
    if best_model_state:
        model.load_state_dict(best_model_state)
    
    return model

def prepare_mnist_data(batch_size=64, valid_split=0.1):
    """Prepare MNIST dataset with data augmentation."""
    # Define transformations
    train_transform = transforms.Compose([
        transforms.RandomAffine(degrees=5, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    # Load datasets
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=train_transform)
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=test_transform)
    
    # Split training data into train and validation
    train_size = int((1 - valid_split) * len(train_dataset))
    valid_size = len(train_dataset) - train_size
    train_dataset, valid_dataset = torch.utils.data.random_split(
        train_dataset, [train_size, valid_size]
    )
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    return train_loader, valid_loader, test_loader

def train_mnist_vae(hidden_dims, latent_dim=10, epochs=20, batch_size=128, beta=1.0, target_digit=3, noise_level=0.05):
    """
    Train a VAE on MNIST dataset, using only examples of the target digit.
    This creates a specialized latent space for representing variations of the target digit.
    
    Args:
        hidden_dims: List of hidden dimensions for the VAE
        latent_dim: Dimension of the latent space
        epochs: Number of training epochs
        batch_size: Batch size for training
        beta: Weight of the KL divergence term in the VAE loss
        target_digit: The digit to train the VAE on
        noise_level: Standard deviation of Gaussian noise to add to training data (default: 0.05)
    
    Returns:
        Trained VAE model
    """
    # Prepare data
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    
    # Check if MNIST data is already downloaded
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    
    # Filter dataset to only include examples of the target digit
    digit_indices = [i for i, (_, label) in enumerate(train_dataset) if label == target_digit]
    print(f"Found {len(digit_indices)} examples of digit {target_digit} in the training set")
    
    # Create a subset of the data containing only the target digit
    digit_dataset = Subset(train_dataset, digit_indices)
    train_loader = DataLoader(digit_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    
    print(f"Training SimpleConvVAE with latent dim {latent_dim} on digit {target_digit} examples only")
    
    # Create the SimpleConvVAE model
    vae = SimpleConvVAE(latent_dim=latent_dim).to(device)
    
    # Prepare data for training
    data_samples = []
    print("Preparing data for VAE training...")
    for i, (images, _) in enumerate(train_loader):
        # Pass images directly (no need to flatten for ConvVAE)
        data_samples.append(images)
        # For digit-specific training, we use all available examples
    
    # Concatenate batches
    train_data = torch.cat(data_samples, dim=0)
    print(f"Prepared {len(train_data)} samples of digit {target_digit} for VAE training")
    
    # Add noise to the training data
    if noise_level > 0:
        print(f"Adding Gaussian noise with std={noise_level} to training data")
        noisy_train_data = train_data + train_data.max() * noise_level * torch.randn_like(train_data)
        # Clip values to be in [0, 1] range since these are image pixel values
        noisy_train_data = torch.clamp(noisy_train_data, 0.0, 1.0)
        print(f"Noise added successfully. Training with noisy data.")
    else:
        noisy_train_data = train_data
        print(f"No noise added. Training with original data.")
    
    # Fit the VAE on the noisy training data
    vae.fit(noisy_train_data, batch_size=min(batch_size, len(noisy_train_data)), epochs=epochs, learning_rate=1e-3)
    print(f"SimpleConvVAE training on digit {target_digit} complete!")
    
    return vae

def objective_function(vae):
    """
    Create an objective function for optimization based on:
    - Probability for the target digit using the binary classifier
    - Minimal image density (to favor interesting images)
    
    Returns a function that can be used with the Bayesian optimizer.
    """
    # Normalization parameters matching the classifier training
    mean = 0.1307
    std = 0.3081
    
    def func(z):
        """
        Objective function for the latent space optimization.
        
        Args:
            z: Latent space vector(s)
            
        Returns:
            Objective value(s)
        """
        # We need to use no_grad to prevent backward graph issues
        with torch.no_grad():
            # Decode the latent vector
            decoded = vae.decode(z)
            
            # Simple image intensity metric
            return torch.mean(decoded ** 2, dim=(-1))
    
    return func

def visualize_optimization_results(vae, best_x, target_digit, name):
    """Visualize the optimization results."""
    # Decode the best latent vector
    with torch.no_grad():
        decoded = vae.decode(best_x)
        
        # Calculate image density
        image_density = torch.mean(decoded ** 2, dim=(-2, -1)).item()
        
        # Reshape to (28, 28) for visualization
        img = decoded.view(28, 28).detach().cpu().numpy()
        
        # Reshape to MNIST format for classifier (B, 1, 28, 28)
        mnist_format = decoded.view(-1, 1, 28, 28)
        
        # Normalization parameters
        mean = 0.1307
        std = 0.3081
        
        # Normalize for classifier
        mnist_format_normalized = (mnist_format - mean) / std
        
        # Load classifier
        classifier = BinaryMNISTClassifier(target_digit=target_digit).to(device)
        classifier.load_state_dict(torch.load("mnist_binary_classifier.pt"))
        classifier.eval()
        
        # Get classifier prediction (already a probability)
        digit_prob = classifier(mnist_format_normalized).item()
        
        # Calculate objective value
        objective_value = image_density
        
        # Create figure with 3 rows
        fig, axes = plt.subplots(3, 1, figsize=(6, 10))
        
        # Show the image in the first row
        axes[0].imshow(img, cmap='gray')
        axes[0].set_title(f"Best {name} SimpleConvVAE Result for Digit {target_digit}")
        axes[0].axis('off')
        
        # Show the probability in the second row as text
        axes[1].text(0.5, 0.5, f"Probability for Digit {target_digit}: {digit_prob:.4f}", 
                    horizontalalignment='center', verticalalignment='center',
                    fontsize=12)
        axes[1].axis('off')
        
        # Show the image density and objective value in the third row
        axes[2].text(0.5, 0.5, 
                    f"Image Density: {image_density:.4f}", 
                    horizontalalignment='center', verticalalignment='center',
                    fontsize=12)
        axes[2].axis('off')
        
        plt.tight_layout()
        plt.savefig(f"{name}_simpleconvvae_digit_{target_digit}_result.png")
        plt.show()

def visualize_optimization_progression(vae, latent_points, objective_values, vae_name, target_digit=3):
    """
    Visualize the progression of optimization by showing each new best point.
    
    Args:
        vae: The VAE model
        latent_points: List of latent points during optimization (initial + newly found points)
        objective_values: List of objective values for each point
        vae_name: Name of the VAE (e.g., "SimpleConvVAE_medium")
        target_digit: Target digit for optimization
    """
    # Find points that improved the objective
    best_points = []
    best_values = []
    best_indices = []
    
    current_best = objective_values[0]
    best_points.append(latent_points[0])
    best_values.append(current_best)
    best_indices.append(0)
    
    for i in range(1, len(objective_values)):
        if objective_values[i] > current_best:
            current_best = objective_values[i]
            best_points.append(latent_points[i])
            best_values.append(current_best)
            best_indices.append(i)
    
    # Set up normalization and classifier
    mean = 0.1307
    std = 0.3081
    classifier = BinaryMNISTClassifier(target_digit=target_digit).to(device)
    classifier.load_state_dict(torch.load("mnist_binary_classifier.pt"))
    classifier.eval()
    
    # Create figure for visualization
    n_points = len(best_points)
    n_cols = min(5, n_points)  # Max 5 columns
    n_rows = (n_points + n_cols - 1) // n_cols  # Ceiling division
    
    # Three rows for each point (image, probability, and density)
    fig, axes = plt.subplots(3 * n_rows, n_cols, figsize=(15, 6 * n_rows))
    
    # Handle the case of single row or column
    if n_rows == 1 and n_cols == 1:
        axes = np.array([[axes[0]], [axes[1]], [axes[2]]])
    elif n_rows == 1:
        axes = np.array([axes[0], axes[1], axes[2]])
    elif n_cols == 1:
        axes = np.array([[ax] for ax in axes])
    
    # Visualize each point
    with torch.no_grad():
        for i, (latent, val, idx) in enumerate(zip(best_points, best_values, best_indices)):
            row = i // n_cols
            col = i % n_cols
            
            # Decode the latent vector
            decoded = vae.decode(latent.unsqueeze(0))
            
            # Calculate image density
            image_density = torch.mean(decoded ** 2, dim=(-2, -1)).item()
            
            # Reshape for visualization
            img = decoded.view(28, 28).detach().cpu().numpy()
            axes[3*row, col].imshow(img, cmap='gray')
            axes[3*row, col].set_title(f"Iteration {idx}\nValue: {val:.4f}")
            axes[3*row, col].axis('off')
            
            # Get probability for target digit
            mnist_format = decoded.view(-1, 1, 28, 28)
            # Normalize for classifier
            mnist_format_normalized = (mnist_format - mean) / std
            
            # Get probability directly from binary classifier
            target_prob = classifier(mnist_format_normalized).item()
            
            # Display probability in second row
            axes[3*row+1, col].text(0.5, 0.5, f"P(digit={target_digit}) = {target_prob:.4f}", 
                        ha='center', va='center', fontsize=10,
                        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
            axes[3*row+1, col].axis('off')
            
            # Display image density in third row
            axes[3*row+2, col].text(0.5, 0.5, f"Image Density = {image_density:.4f}", 
                        ha='center', va='center', fontsize=10,
                        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
            axes[3*row+2, col].axis('off')
    
    # Hide unused subplots
    for i in range(len(best_points), n_rows * n_cols):
        row = i // n_cols
        col = i % n_cols
        if 3*row < axes.shape[0] and col < axes.shape[1]:
            axes[3*row, col].axis('off')
            axes[3*row+1, col].axis('off')
            axes[3*row+2, col].axis('off')
    
    plt.tight_layout()
    plt.savefig(f"{vae_name}_optimization_progression_digit_{target_digit}.png")
    plt.show()

def plot_optimization_progress(objective_values, vae_name, target_digit=3):
    """
    Plot the maximum achieved objective value over optimization iterations.
    
    Args:
        objective_values: List of objective values at each iteration
        vae_name: Name of the VAE being optimized
        target_digit: Target digit for optimization
    """
    # Calculate the maximum value achieved up to each iteration
    max_values = []
    current_max = objective_values[0]
    
    for val in objective_values:
        if val > current_max:
            current_max = val
        max_values.append(current_max)
    
    # Create the plot
    plt.figure(figsize=(10, 6))
    plt.plot(range(len(max_values)), max_values, marker='o')
    plt.title(f"Maximum Objective Value vs. Iteration ({vae_name}, Digit {target_digit})")
    plt.xlabel("Iteration")
    plt.ylabel("Maximum Objective Value")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{vae_name}_objective_progress_digit_{target_digit}.png")
    plt.show()

def plot_objective_components(vae, latent_points, classifier, vae_name, target_digit=3):
    """
    Plot the components of the objective function (probability and image density)
    over optimization iterations.
    
    Args:
        vae: The VAE model
        latent_points: List of latent points during optimization
        classifier: The classifier model
        vae_name: Name of the VAE
        target_digit: Target digit for optimization
    """
    # Normalization parameters
    mean = 0.1307
    std = 0.3081
    
    # Calculate probabilities and image densities for all latent points
    probabilities = []
    densities = []
    objective_values = []
    
    with torch.no_grad():
        for latent in latent_points:
            # If the latent point doesn't have batch dimension, add it
            if latent.dim() == 1:
                latent = latent.unsqueeze(0)
                
            # Decode the latent vector
            decoded = vae.decode(latent)
            
            # Calculate image density
            density = torch.mean(decoded ** 2, dim=(-2, -1)).item()
            densities.append(density)
            
            # Get probability from classifier
            mnist_format = decoded.view(-1, 1, 28, 28)
            mnist_format_normalized = (mnist_format - mean) / std
            prob = classifier(mnist_format_normalized).item()
            probabilities.append(prob)
            
            # Calculate objective value
            obj_value = density
            objective_values.append(obj_value)
    
    # Create figure with 3 subplots
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    
    # Plot probability over iterations
    axes[0].plot(probabilities, marker='o')
    axes[0].set_title(f'Digit {target_digit} Probability vs. Iteration')
    axes[0].set_ylabel('Probability')
    axes[0].grid(True)
    
    # Plot image density over iterations
    axes[1].plot(densities, marker='o', color='green')
    axes[1].set_title('Image Density vs. Iteration')
    axes[1].set_ylabel('Image Density')
    axes[1].grid(True)
    
    # Plot objective value over iterations
    axes[2].plot(objective_values, marker='o', color='red')
    axes[2].set_title('Objective Value vs. Iteration')
    axes[2].set_xlabel('Iteration')
    axes[2].set_ylabel('Objective Value')
    axes[2].grid(True)
    
    plt.tight_layout()
    plt.savefig(f"{vae_name}_objective_components_digit_{target_digit}.png")
    plt.show()

def plot_probability_density_tradeoff(vae, all_latents, classifier, vae_name, target_digit=3):
    """
    Create a scatter plot showing the trade-off between probability and image density
    for all optimization points.
    
    Args:
        vae: The VAE model
        all_latents: List of all latent points explored during optimization
        classifier: The classifier model
        vae_name: Name of the VAE
        target_digit: Target digit for optimization
    """
    # Normalization parameters
    mean = 0.1307
    std = 0.3081
    
    # Calculate probabilities and image densities for all latent points
    probabilities = []
    densities = []
    objective_values = []
    
    with torch.no_grad():
        for latent in all_latents:
            # If the latent point doesn't have batch dimension, add it
            if latent.dim() == 1:
                latent = latent.unsqueeze(0)
                
            # Decode the latent vector
            decoded = vae.decode(latent)
            
            # Calculate image density
            density = torch.mean(decoded ** 2, dim=(-2, -1)).item()
            densities.append(density)
            
            # Get probability from classifier
            mnist_format = decoded.view(-1, 1, 28, 28)
            mnist_format_normalized = (mnist_format - mean) / std
            prob = classifier(mnist_format_normalized).item()
            probabilities.append(prob)
            
            # Calculate objective value
            obj_value = density
            objective_values.append(obj_value)
    
    # Create a scatter plot
    plt.figure(figsize=(10, 8))
    
    # Create a color map based on objective values
    normalized_obj = np.array(objective_values) / max(objective_values)
    
    # Create scatter plot with color indicating objective value
    scatter = plt.scatter(probabilities, densities, c=normalized_obj, 
                         s=80, cmap='viridis', alpha=0.7)
    
    # Add colorbar
    cbar = plt.colorbar(scatter)
    cbar.set_label('Normalized Objective Value')
    
    # Add labels and title
    plt.xlabel('Probability of Digit ' + str(target_digit))
    plt.ylabel('Image Density')
    plt.title(f'Probability vs. Image Density Trade-off ({vae_name})')
    
    # Add grid
    plt.grid(True, linestyle='--', alpha=0.6)
    
    # Highlight the best point (highest objective value)
    best_idx = np.argmax(objective_values)
    plt.scatter([probabilities[best_idx]], [densities[best_idx]], 
               s=200, facecolors='none', edgecolors='red', linewidth=2, 
               label='Best Point')
    
    # Add a text annotation for the best point
    plt.annotate(f'Best Point\nProb: {probabilities[best_idx]:.4f}\nDensity: {densities[best_idx]:.4f}',
                xy=(probabilities[best_idx], densities[best_idx]),
                xytext=(probabilities[best_idx] - 0.1, densities[best_idx] + 0.05),
                arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=8),
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black", alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(f"{vae_name}_probability_density_tradeoff_digit_{target_digit}.png")
    plt.show()

def main():
    # Define different SimpleConvVAE configurations to test
    vae_configs = [
        {"name": "SimpleConvVAE_small", "hidden_dims": None, "latent_dim": 3, "noise_level": 0.2}
    ]
    
    # Target digit for optimization
    target_digit = 3
    
    # Path for saving and loading the classifier
    classifier_path = "mnist_binary_classifier.pt"
    
    # Load or train binary MNIST classifier
    classifier = BinaryMNISTClassifier(target_digit=target_digit).to(device)
    
    # Check if a pre-trained classifier exists
    if os.path.exists(classifier_path):
        print(f"Loading pre-trained binary classifier from {classifier_path}")
        classifier.load_state_dict(torch.load(classifier_path))
    else:
        print("Preparing MNIST data...")
        train_loader, valid_loader, test_loader = prepare_mnist_data()
        
        print(f"Training binary classifier for digit {target_digit} for 3 epochs...")
        classifier = train_mnist_classifier(classifier, train_loader, valid_loader, epochs=3)
        
        # Save the trained classifier
        print(f"Saving trained binary classifier to {classifier_path}")
        torch.save(classifier.state_dict(), classifier_path)
    
    # Evaluate classifier
    print("Evaluating binary classifier...")
    # If we loaded the model, we need to prepare data loaders for evaluation
    if not 'test_loader' in locals():
        _, _, test_loader = prepare_mnist_data()
        
    classifier.eval()
    test_correct = 0
    test_total = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            
            # Convert targets to binary (1 for target_digit, 0 for others)
            binary_target = (target == target_digit).float().unsqueeze(1)
            
            output = classifier(data)
            predicted = (output >= 0.5).float()  # Binary threshold at 0.5
            test_total += binary_target.size(0)
            test_correct += (predicted == binary_target).sum().item()
    
    test_accuracy = 100. * test_correct / test_total
    print(f"Binary classifier test accuracy: {test_accuracy:.2f}%")
    
    # Get some examples of the target digit from MNIST
    print(f"Getting examples of digit {target_digit} from MNIST...")
    
    # Load MNIST dataset with the same normalization used for training the classifier
    mnist_data = datasets.MNIST(
        root='./data', 
        train=True, 
        download=True, 
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))  # Same normalization used for classifier
        ])
    )
    
    # Find examples of the target digit
    digit_examples = []
    digit_examples_normalized = []  # For the classifier
    digit_indices = []
    min_density = 10000
    
    for i, (image, label) in enumerate(mnist_data):
        if label == target_digit:
            # Store both normalized and unnormalized versions
            # Unnormalized for VAE
            unnormalized_img = transforms.ToTensor()(transforms.ToPILImage()(image))
            if (unnormalized_img ** 2).mean() < min_density:
                min_density = (unnormalized_img ** 2).mean()
                digit_examples.append(unnormalized_img.view(-1))  # Flatten
                
                # Normalized for classifier
                digit_examples_normalized.append(image.view(-1))  # Already normalized
                
                digit_indices.append(i)
    digit_examples = digit_examples[-5:]
    digit_examples_normalized = digit_examples_normalized[-5:]
    digit_indices = digit_indices[-5:]
    
    # Convert to tensors
    digit_examples = torch.stack(digit_examples)
    digit_examples_normalized = torch.stack(digit_examples_normalized)
    print(f"Found {len(digit_examples)} examples of digit {target_digit}")
    
    # Verify classifier on these examples
    with torch.no_grad():
        # Reshape to MNIST format for classifier (B, 1, 28, 28)
        mnist_format = digit_examples_normalized.view(-1, 1, 28, 28)
        
        # Get probabilities directly from binary classifier
        probs = classifier(mnist_format)
        
        print(f"Binary classifier probabilities for digit {target_digit} examples:")
        for i, prob in enumerate(probs):
            print(f"  Example {i+1}: {prob.item():.4f}")
    
    # Dictionary to store results
    results = {}
    
    # Test each VAE configuration
    for config in vae_configs:
        name = config["name"]
        hidden_dims = config["hidden_dims"]
        latent_dim = config["latent_dim"]
        noise_level = config["noise_level"]
        print(f"\n=== Testing {name} ===")
        
        # Define a filename for the VAE model trained on specific digit
        vae_filename = f"{name}_digit{target_digit}_latent{latent_dim}_noise{noise_level}.pt"
        
        # Check if a pre-trained VAE exists
        if os.path.exists(vae_filename):
            print(f"Loading pre-trained {name} VAE for digit {target_digit} from {vae_filename}")
            vae = torch.load(vae_filename)
        else:
            # Train VAE
            print(f"Training new {name} VAE for digit {target_digit}...")
            vae = train_mnist_vae(
                hidden_dims=hidden_dims,
                latent_dim=latent_dim,
                epochs=15,  # Fewer epochs for faster testing
                batch_size=128,
                beta=1.0,
                target_digit=target_digit,
                noise_level=noise_level  # Add Gaussian noise with std=0.08 to improve robustness and generalization
            )

            # Save the trained VAE
            torch.save(vae, vae_filename)
            print(f"Saved trained VAE to {vae_filename}")
        
        # for latent in torch.normal(0, 1, (10, latent_dim)):
        #     plt.subplot(2, 5, i+1)
        #     decoded = vae.decode(latent).view(28, 28).cpu().detach().numpy()
        #     decoded = (decoded - decoded.min()) / (decoded.max() - decoded.min())
        #     plt.imshow(decoded, cmap='gray')
        #     plt.axis('off')
        # plt.show()

        # Debug: Visualize original examples and their reconstructions
        print("Visualizing original examples and their reconstructions...")
        with torch.no_grad():
            # For SimpleConvVAE, we need to reshape images to match the expected input
            if digit_examples.dim() == 2:
                # If flattened, reshape to [B, 1, 28, 28]
                examples_for_conv = digit_examples.view(-1, 1, 28, 28).to(device)
            else:
                examples_for_conv = digit_examples.to(device)
                
        #     # Encode and decode the examples
        #     mu, logvar = vae.encode(examples_for_conv)
        #     encoded = vae.reparameterize(mu, logvar)
        #     reconstructed = vae.decode(encoded)
            
        #     # Reshape for visualization
        #     original_images = digit_examples.view(-1, 28, 28).cpu()
        #     reconstructed_images = reconstructed.view(-1, 28, 28).cpu()
            
        #     # Create figure with 3 rows: originals, reconstructions, and probabilities
        #     fig, axes = plt.subplots(3, 5, figsize=(15, 9))
            
        #     # Plot original images on top row
        #     for i in range(5):
        #         axes[0, i].imshow(original_images[i], cmap='gray')
        #         axes[0, i].set_title(f"Original {i+1}")
        #         axes[0, i].axis('off')
            
        #     # Plot reconstructed images on middle row
        #     for i in range(5):
        #         axes[1, i].imshow(reconstructed_images[i], cmap='gray')
        #         axes[1, i].set_title(f"Reconstructed {i+1}")
        #         axes[1, i].axis('off')
            
        #     # Normalize reconstructions for classifier
        #     mnist_format_recon = reconstructed.view(-1, 1, 28, 28)
            mean = 0.1307
            std = 0.3081
        #     mnist_format_recon_normalized = (mnist_format_recon - mean) / std
            
        #     # Get classifier probabilities (binary classifier outputs probabilities directly)
        #     probs_recon = classifier(mnist_format_recon_normalized)
            
        #     # Display probabilities in bottom row
        #     for i in range(5):
        #         prob = probs_recon[i].item()
        #         axes[2, i].text(0.5, 0.5, f"P(digit={target_digit}) = {prob:.4f}", 
        #                     ha='center', va='center', fontsize=10,
        #                     bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
        #         axes[2, i].axis('off')
            
        #     plt.tight_layout()
        #     plt.savefig(f"{name}_digit{target_digit}_reconstructions.png")
        #     plt.show()
            
        #     print(f"Binary classifier probabilities for reconstructed examples:")
        #     for i, prob in enumerate(probs_recon[:5]):
        #         print(f"  Example {i+1}: {prob.item():.6f}")
        
        # For SimpleConvVAE, we need to adjust the encoding approach
        # Encode the example digits to get the initial latent vectors
        with torch.no_grad():
            mu, logvar = vae.encode(examples_for_conv)
            encoded_examples = vae.reparameterize(mu, logvar)
        
        print(f"Encoded {len(encoded_examples)} example digits to latent space")
        
        # Extract 5 examples to use as initial points
        n_initial = 5
        initial_points = encoded_examples[:n_initial]
        
        # Create objective function
        obj_func = objective_function(vae)
        
        # Evaluate initial points
        initial_values = obj_func(initial_points).unsqueeze(-1)
        
        # Prepare for Bayesian optimization
        bounds = torch.ones(latent_dim, 2)
        bounds[:, 0] = -3  # Lower bound
        bounds[:, 1] = 3   # Upper bound
        
        # Create optimizer
        optimizer = BayesianOptimizer(
            bounds=bounds,
            n_initial_points=0,  # We're providing our own initial points
            initial_points=initial_points,
            initial_values=initial_values,
            encoder=None  # No additional encoding in latent space
        )
        
        # Storage for tracking optimization progress
        objective_values = [v.item() for v in initial_values]  # Store individual values
        all_latents = [p.clone() for p in initial_points]  # Store initial latent points
        
        # Define a custom objective function wrapper to track values and latent points
        def tracking_objective(z):
            result = obj_func(z)
            
            # Store results
            for i in range(z.shape[0]):
                objective_values.append(result[i].item())
                all_latents.append(z[i].clone())
                
            return result
        
        # Run optimization with fewer iterations for faster testing
        n_iterations = 10  # Reduced from 30
        print(f"Running Bayesian optimization for {n_iterations} iterations...")
        start_time = time.time()
        best_x, best_y, _, _ = optimizer.optimize(
            objective_function=tracking_objective,
            n_iterations=n_iterations
        )
        end_time = time.time()
        
        print(f"Optimization time: {end_time - start_time:.2f} seconds")
        print(f"Best objective value: {best_y.item():.4f}")
        
        # Save intermediate results for this VAE
        print(f"Saving the best latent vector to {name}_digit{target_digit}_best_latent.pt")
        torch.save(best_x, f"{name}_digit{target_digit}_best_latent.pt")
        
        # Plot the optimization progress
        plot_optimization_progress(objective_values, f"{name}_digit{target_digit}", target_digit)
        
        # Visualize results with probabilities
        visualize_optimization_results(vae, best_x, target_digit, f"{name}_digit{target_digit}")
        
        # Visualize the progression of optimal points
        visualize_optimization_progression(vae, all_latents, objective_values, f"{name}_digit{target_digit}", target_digit)
        
        # Plot objective components
        plot_objective_components(vae, all_latents, classifier, f"{name}_digit{target_digit}", target_digit)
        
        # Plot probability density tradeoff
        # plot_probability_density_tradeoff(vae, all_latents, classifier, f"{name}_digit{target_digit}", target_digit)
        
        # Store results
        results[name] = {
            "vae": vae,
            "best_latent": best_x,
            "best_value": best_y.item(),
            "initial_latents": initial_points,
            "all_latents": all_latents,
            "objective_values": objective_values,
            "config": config
        }
    
    # Compare results
    print("\n=== Results Summary ===")
    for name, result in results.items():
        config = result["config"]
        
        # Calculate probability and density for the best point
        with torch.no_grad():
            best_latent = result["best_latent"]
            if best_latent.dim() == 1:
                best_latent = best_latent.unsqueeze(0)
                
            # Decode the latent vector
            decoded = result["vae"].decode(best_latent)
            
            # Calculate image density
            density = torch.mean(decoded ** 2, dim=(-2, -1)).item()
            
            # Get probability from classifier
            mnist_format = decoded.view(-1, 1, 28, 28)
            mnist_format_normalized = (mnist_format - mean) / std
            prob = classifier(mnist_format_normalized).item()
            
        print(f"{name}:")
        print(f"  Latent dim: {config['latent_dim']}")
        print(f"  Probability: {prob:.4f}")
        print(f"  Image Density: {density:.4f}")
        print(f"  Objective Value: {result['best_value']:.4f}")
    

if __name__ == "__main__":
    main() 