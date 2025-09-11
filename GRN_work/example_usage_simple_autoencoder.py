"""
Example usage script for training the simple autoencoder.

This script demonstrates how to use the simple autoencoder training script
with different configurations and provides examples for common use cases.
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
    """Main function to run training examples."""
    
    # Example data path - update this to your actual data path
    data_path = "C:/Users/rdbra/Documents/honoursProject/code_base/data/sct_matrix_transposed.npz"
    
    print("Simple Autoencoder Training Examples")
    print("=" * 60)
    
    # Check if data file exists
    if not os.path.exists(data_path):
        print(f"❌ Data file not found: {data_path}")
        print("Please update the data_path variable with the correct path to your data file.")
        return
    
    print(f"✅ Data file found: {data_path}")
    
    # Example 1: Basic training with default parameters
    run_training_example(
        data_path=data_path,
        example_name="Basic Training (Default Parameters)",
        epochs=100,  # Reduced for quick testing
        batch_size=64
    )
    
    # Example 2: Small model for quick testing
    run_training_example(
        data_path=data_path,
        example_name="Small Model (Quick Test)",
        hidden_dims=[256, 128],
        latent_dim=64,
        epochs=50,
        batch_size=32,
        learning_rate=1e-3
    )
    
    # Example 3: Large model with more capacity
    run_training_example(
        data_path=data_path,
        example_name="Large Model (High Capacity)",
        hidden_dims=[2048, 1024, 512],
        latent_dim=256,
        epochs=200,
        batch_size=128,
        learning_rate=5e-4,
        dropout_rate=0.2
    )
    
    # Example 4: Custom binning configuration
    run_training_example(
        data_path=data_path,
        example_name="Custom Binning Configuration",
        num_bins=20,
        epochs=150,
        batch_size=64,
        learning_rate=1e-3
    )
    
    print(f"\n{'='*60}")
    print("All examples completed!")
    print("Check the results directories created next to your data file.")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
