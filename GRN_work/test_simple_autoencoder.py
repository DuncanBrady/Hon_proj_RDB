"""
Test script for the simple autoencoder implementation.

This script performs basic tests to ensure the autoencoder model and training
components work correctly before running full training.
"""

import os
import sys
import numpy as np
import torch
import tempfile
import json

# Add the project root to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from src.model.fc_autoencoder import SimpleAutoencoder, weighted_mse_loss, calculate_accuracy, model_summary
from src.preprocess.binning import term_freq_bin


def test_binning_function():
    """Test the term frequency binning function."""
    print("Testing binning function...")
    
    # Create test data
    test_data = np.array([
        [0, 1.5, 2.0, 0, 3.5],
        [0.5, 0, 2.5, 1.0, 0],
        [1.0, 2.0, 0, 2.5, 4.0]
    ], dtype=np.float32)
    
    print(f"Original data:\n{test_data}")
    print(f"Non-zero values: {test_data[test_data > 0]}")
    
    # Apply binning
    num_bins = 5
    binned_data = term_freq_bin(test_data.copy(), num_bins)
    
    print(f"Binned data (with {num_bins} bins):\n{binned_data}")
    print(f"Unique values after binning: {np.unique(binned_data)}")
    
    # Verify binning worked
    assert binned_data[test_data == 0].sum() == 0, "Zero values should remain zero"
    assert len(np.unique(binned_data[binned_data > 0])) <= num_bins, "Should have at most num_bins unique non-zero values"
    
    print("✅ Binning function test passed!")
    return True


def test_autoencoder_model():
    """Test the autoencoder model architecture."""
    print("\nTesting autoencoder model...")
    
    # Model parameters
    input_dim = 1000
    hidden_dims = [512, 256]
    latent_dim = 128
    batch_size = 32
    
    # Create model
    model = SimpleAutoencoder(
        input_dim=input_dim,
        hidden_dims=hidden_dims,
        latent_dim=latent_dim,
        dropout_rate=0.1
    )
    
    print(f"Model created with input_dim={input_dim}, hidden_dims={hidden_dims}, latent_dim={latent_dim}")
    
    # Test forward pass
    model.eval()
    with torch.no_grad():
        test_input = torch.randn(batch_size, input_dim)
        reconstructed, latent = model(test_input)
    
    # Verify shapes
    assert test_input.shape == reconstructed.shape, f"Input and output shapes don't match: {test_input.shape} vs {reconstructed.shape}"
    assert latent.shape == (batch_size, latent_dim), f"Latent shape incorrect: {latent.shape} vs expected {(batch_size, latent_dim)}"
    
    print(f"✅ Forward pass successful!")
    print(f"  Input shape: {test_input.shape}")
    print(f"  Latent shape: {latent.shape}")
    print(f"  Output shape: {reconstructed.shape}")
    
    # Test individual encode/decode functions
    with torch.no_grad():
        encoded = model.encode(test_input)
        decoded = model.decode(encoded)
    
    assert encoded.shape == latent.shape, "Encode function shape mismatch"
    assert decoded.shape == reconstructed.shape, "Decode function shape mismatch"
    
    print(f"✅ Encode/decode functions work correctly!")
    
    # Print model summary
    model_summary(model, test_input.shape)
    
    return True


def test_loss_functions():
    """Test the loss and accuracy functions."""
    print("\nTesting loss and accuracy functions...")
    
    # Create test data
    original = torch.tensor([
        [0.0, 1.0, 2.0, 0.0, 3.0],
        [0.5, 0.0, 1.5, 2.5, 0.0]
    ])
    
    # Perfect reconstruction
    perfect_recon = original.clone()
    loss_perfect = weighted_mse_loss(perfect_recon, original)
    acc_perfect = calculate_accuracy(perfect_recon, original)
    
    print(f"Perfect reconstruction - Loss: {loss_perfect:.6f}, Accuracy: {acc_perfect:.4f}")
    assert loss_perfect < 1e-6, "Perfect reconstruction should have near-zero loss"
    assert acc_perfect > 0.99, "Perfect reconstruction should have high accuracy"
    
    # Noisy reconstruction
    noisy_recon = original + 0.1 * torch.randn_like(original)
    loss_noisy = weighted_mse_loss(noisy_recon, original)
    acc_noisy = calculate_accuracy(noisy_recon, original)
    
    print(f"Noisy reconstruction - Loss: {loss_noisy:.6f}, Accuracy: {acc_noisy:.4f}")
    assert loss_noisy > loss_perfect, "Noisy reconstruction should have higher loss"
    
    # Very poor reconstruction
    poor_recon = torch.randn_like(original)
    loss_poor = weighted_mse_loss(poor_recon, original)
    acc_poor = calculate_accuracy(poor_recon, original)
    
    print(f"Poor reconstruction - Loss: {loss_poor:.6f}, Accuracy: {acc_poor:.4f}")
    assert loss_poor > loss_noisy, "Poor reconstruction should have highest loss"
    
    print("✅ Loss and accuracy functions test passed!")
    return True


def test_full_training_pipeline():
    """Test a minimal training pipeline."""
    print("\nTesting full training pipeline...")
    
    # Create synthetic gene expression data
    n_cells = 100
    n_genes = 500
    
    # Generate synthetic data with some structure
    np.random.seed(42)
    
    # Create some cell types with different expression patterns
    cell_type_1 = np.random.exponential(1.0, (50, n_genes)) * (np.random.random((50, n_genes)) > 0.7)
    cell_type_2 = np.random.exponential(1.5, (50, n_genes)) * (np.random.random((50, n_genes)) > 0.8)
    
    synthetic_data = np.vstack([cell_type_1, cell_type_2]).astype(np.float32)
    
    print(f"Synthetic data shape: {synthetic_data.shape}")
    print(f"Data range: [{synthetic_data.min():.4f}, {synthetic_data.max():.4f}]")
    print(f"Sparsity: {100 * (synthetic_data == 0).sum() / synthetic_data.size:.1f}% zeros")
    
    # Apply binning
    binned_data = term_freq_bin(synthetic_data.copy(), num_bins=10)
    
    # Create dataset
    from torch.utils.data import DataLoader, Dataset
    
    class TestDataset(Dataset):
        def __init__(self, data):
            self.data = torch.tensor(data, dtype=torch.float32)
        def __len__(self):
            return len(self.data)
        def __getitem__(self, idx):
            return self.data[idx]
    
    dataset = TestDataset(binned_data)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
    
    # Create model
    model = SimpleAutoencoder(
        input_dim=n_genes,
        hidden_dims=[256, 128],
        latent_dim=64,
        dropout_rate=0.1
    )
    
    # Setup optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    
    # Training loop (just a few epochs)
    model.train()
    print("Running mini training loop...")
    
    for epoch in range(5):
        epoch_loss = 0.0
        epoch_acc = 0.0
        n_batches = 0
        
        for batch_data in dataloader:
            # Forward pass
            reconstructed, latent = model(batch_data)
            
            # Calculate loss
            loss = weighted_mse_loss(reconstructed, batch_data)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Calculate accuracy
            with torch.no_grad():
                acc = calculate_accuracy(reconstructed, batch_data)
            
            epoch_loss += loss.item()
            epoch_acc += acc
            n_batches += 1
        
        avg_loss = epoch_loss / n_batches
        avg_acc = epoch_acc / n_batches
        
        print(f"  Epoch {epoch+1}/5 - Loss: {avg_loss:.6f}, Accuracy: {avg_acc:.4f}")
    
    print("✅ Full training pipeline test passed!")
    return True


def test_file_saving():
    """Test file saving functionality."""
    print("\nTesting file saving functionality...")
    
    # Create temporary directory
    with tempfile.TemporaryDirectory() as temp_dir:
        print(f"Using temporary directory: {temp_dir}")
        
        # Create simple model
        model = SimpleAutoencoder(input_dim=100, hidden_dims=[50], latent_dim=25)
        
        # Save model
        model_path = os.path.join(temp_dir, "test_model.pth")
        torch.save(model.state_dict(), model_path)
        assert os.path.exists(model_path), "Model file was not saved"
        print(f"✅ Model saved to {model_path}")
        
        # Test loading model
        model2 = SimpleAutoencoder(input_dim=100, hidden_dims=[50], latent_dim=25)
        model2.load_state_dict(torch.load(model_path))
        print("✅ Model loaded successfully")
        
        # Save configuration
        config = {
            'input_dim': 100,
            'hidden_dims': [50],
            'latent_dim': 25,
            'epochs': 10
        }
        
        config_path = os.path.join(temp_dir, "test_config.json")
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        assert os.path.exists(config_path), "Config file was not saved"
        print(f"✅ Configuration saved to {config_path}")
        
        # Test loading configuration
        with open(config_path, 'r') as f:
            loaded_config = json.load(f)
        
        assert loaded_config == config, "Configuration mismatch after loading"
        print("✅ Configuration loaded successfully")
    
    print("✅ File saving test passed!")
    return True


def main():
    """Run all tests."""
    print("Running Simple Autoencoder Tests")
    print("=" * 50)
    
    tests = [
        test_binning_function,
        test_autoencoder_model,
        test_loss_functions,
        test_full_training_pipeline,
        test_file_saving
    ]
    
    passed = 0
    failed = 0
    
    for test_func in tests:
        try:
            test_func()
            passed += 1
        except Exception as e:
            print(f"❌ {test_func.__name__} failed: {e}")
            failed += 1
    
    print("\n" + "=" * 50)
    print(f"Test Results: {passed} passed, {failed} failed")
    
    if failed == 0:
        print("🎉 All tests passed! The autoencoder implementation is working correctly.")
        return True
    else:
        print("⚠️  Some tests failed. Please check the implementation.")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
