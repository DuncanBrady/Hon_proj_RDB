"""
Example usage script for training the simple autoencoder with different loss functions.

This script demonstrates how to use the different loss function options
including MSE, cross-entropy, and Huber loss.
"""

import os
import subprocess
import sys


def run_training_example(data_path, example_name, **kwargs):
    """
    Run a training example with specified parameters.
    
    Args:
        data_path (str): Path to the data file
        example_name (str): Name of the example
        **kwargs: Additional arguments for training
    """
    print(f"\n{'='*60}")
    print(f"Running Example: {example_name}")
    print(f"{'='*60}")
    
    # Base command
    cmd = [
        sys.executable,
        "train_simple_autoencoder.py",
        "--data_path", data_path
    ]
    
    # Add additional arguments
    for key, value in kwargs.items():
        if isinstance(value, list):
            cmd.extend([f"--{key}"] + [str(v) for v in value])
        else:
            cmd.extend([f"--{key}", str(value)])
    
    print(f"Command: {' '.join(cmd)}")
    print()
    
    try:
        # Run the training script
        result = subprocess.run(cmd, check=True, capture_output=False, text=True)
        print(f"\n✅ {example_name} completed successfully!")
        
    except subprocess.CalledProcessError as e:
        print(f"\n❌ {example_name} failed with return code {e.returncode}")
        print(f"Error: {e}")


def main():
    """Main function to run training examples with different loss functions."""
    
    # Example data path - update this to your actual data path
    data_path = "data/synthetic_gene_expression.npz"
    
    print("Simple Autoencoder Training Examples - Loss Function Variants")
    print("=" * 60)
    
    # Check if data file exists
    if not os.path.exists(data_path):
        print(f"❌ Data file not found: {data_path}")
        print("Please run create_synthetic_data.py first to generate test data.")
        return
    
    print(f"✅ Data file found: {data_path}")
    
    # Example 1: MSE Loss (default)
    run_training_example(
        data_path=data_path,
        example_name="MSE Loss (Default)",
        loss_type="mse",
        epochs=50,
        batch_size=32,
        hidden_dims=[256, 128],
        latent_dim=64,
        zero_weight=0.1,
        nonzero_weight=1.0
    )
    
    # Example 2: Cross-Entropy Loss for Discrete Data
    run_training_example(
        data_path=data_path,
        example_name="Cross-Entropy Loss (Classification)",
        loss_type="cross_entropy",
        epochs=50,
        batch_size=32,
        hidden_dims=[256, 128],
        latent_dim=64,
        num_bins=15,  # More bins for finer classification
        zero_weight=0.2,
        nonzero_weight=1.0
    )
    
    # Example 3: Huber Loss (Robust to Outliers)
    run_training_example(
        data_path=data_path,
        example_name="Huber Loss (Robust to Outliers)",
        loss_type="huber",
        epochs=50,
        batch_size=32,
        hidden_dims=[256, 128],
        latent_dim=64,
        huber_delta=1.0,
        zero_weight=0.1,
        nonzero_weight=1.0
    )
    
    # Example 4: Cross-Entropy with Specified Number of Classes
    run_training_example(
        data_path=data_path,
        example_name="Cross-Entropy with Fixed Classes",
        loss_type="cross_entropy",
        epochs=30,
        batch_size=64,
        hidden_dims=[512, 256],
        latent_dim=128,
        num_classes=8,  # Fixed number of classes
        num_bins=8,
        zero_weight=0.1,
        nonzero_weight=2.0  # Higher weight for non-zero values
    )
    
    # Example 5: Large Model with Cross-Entropy
    run_training_example(
        data_path=data_path,
        example_name="Large Model with Cross-Entropy",
        loss_type="cross_entropy",
        epochs=40,
        batch_size=64,
        hidden_dims=[1024, 512, 256],
        latent_dim=256,
        dropout_rate=0.2,
        learning_rate=5e-4,
        num_bins=12
    )
    
    # Example 6: Huber Loss with Custom Delta
    run_training_example(
        data_path=data_path,
        example_name="Huber Loss with Custom Delta",
        loss_type="huber",
        epochs=40,
        batch_size=32,
        hidden_dims=[256, 128],
        latent_dim=64,
        huber_delta=0.5,  # Smaller delta for more sensitivity
        zero_weight=0.05,
        nonzero_weight=1.5
    )
    
    print(f"\n{'='*60}")
    print("All loss function examples completed!")
    print("Check the results directories for detailed training metrics and model weights.")
    print("Each run creates a separate results directory with:")
    print("  - Trained model weights (.pth)")
    print("  - Training history (JSON)")
    print("  - Model architecture info (JSON)")
    print("  - Final metrics summary (TXT)")
    print("  - Training logs")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
