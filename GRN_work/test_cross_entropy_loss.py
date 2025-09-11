"""
Test script for the cross-entropy loss function in fc_autoencoder.py

This script tests the new cross-entropy reconstruction loss and the updated
autoencoder model with different output configurations.
"""

import sys
import os
import torch
import numpy as np

# Add the project root to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from src.model.fc_autoencoder import (
    SimpleAutoencoder, 
    weighted_mse_loss, 
    cross_entropy_reconstruction_loss,
    huber_reconstruction_loss,
    get_loss_function,
    calculate_accuracy,
    model_summary
)


def test_cross_entropy_loss():
    """Test the cross-entropy reconstruction loss function."""
    print("Testing cross-entropy reconstruction loss...")
    
    # Create test data with discrete values (like binned gene expression)
    batch_size, num_genes = 16, 100
    num_classes = 5
    
    # Original data as class indices (0, 1, 2, 3, 4)
    original = torch.randint(0, num_classes, (batch_size, num_genes))
    
    # Test case 1: Perfect reconstruction (same class indices)
    reconstructed_perfect = original.float()
    loss_perfect = cross_entropy_reconstruction_loss(reconstructed_perfect, original, num_classes)
    print(f"Perfect reconstruction loss: {loss_perfect:.6f}")
    
    # Test case 2: Random reconstruction
    reconstructed_random = torch.randint(0, num_classes, (batch_size, num_genes)).float()
    loss_random = cross_entropy_reconstruction_loss(reconstructed_random, original, num_classes)
    print(f"Random reconstruction loss: {loss_random:.6f}")
    
    # Test case 3: Logits format (batch_size, num_genes, num_classes)
    logits = torch.randn(batch_size, num_genes, num_classes)
    loss_logits = cross_entropy_reconstruction_loss(logits, original, num_classes)
    print(f"Logits reconstruction loss: {loss_logits:.6f}")
    
    print("✅ Cross-entropy loss test completed!")
    return True


def test_loss_function_factory():
    """Test the loss function factory."""
    print("\nTesting loss function factory...")
    
    # Create test data
    batch_size, num_genes = 8, 50
    original = torch.rand(batch_size, num_genes) * 5  # Values 0-5
    reconstructed = original + 0.1 * torch.randn_like(original)
    
    # Test MSE loss
    mse_loss_fn = get_loss_function('mse', zero_weight=0.1, nonzero_weight=1.0)
    mse_loss = mse_loss_fn(reconstructed, original)
    print(f"MSE loss: {mse_loss:.6f}")
    
    # Test Huber loss
    huber_loss_fn = get_loss_function('huber', delta=1.0, zero_weight=0.1, nonzero_weight=1.0)
    huber_loss = huber_loss_fn(reconstructed, original)
    print(f"Huber loss: {huber_loss:.6f}")
    
    # Test Cross-entropy loss
    original_discrete = torch.randint(0, 5, (batch_size, num_genes))
    reconstructed_discrete = torch.randint(0, 5, (batch_size, num_genes)).float()
    ce_loss_fn = get_loss_function('cross_entropy', num_classes=5, zero_weight=0.1, nonzero_weight=1.0)
    ce_loss = ce_loss_fn(reconstructed_discrete, original_discrete)
    print(f"Cross-entropy loss: {ce_loss:.6f}")
    
    print("✅ Loss function factory test completed!")
    return True


def test_autoencoder_with_cross_entropy():
    """Test the autoencoder with cross-entropy output."""
    print("\nTesting autoencoder with cross-entropy configuration...")
    
    # Model parameters
    input_dim = 500
    hidden_dims = [256, 128]
    latent_dim = 64
    num_classes = 10
    batch_size = 16
    
    # Create model for classification (cross-entropy)
    model_ce = SimpleAutoencoder(
        input_dim=input_dim,
        hidden_dims=hidden_dims,
        latent_dim=latent_dim,
        dropout_rate=0.1,
        output_activation='softmax',
        num_classes=num_classes
    )
    
    # Test forward pass
    model_ce.eval()
    with torch.no_grad():
        test_input = torch.randint(0, num_classes, (batch_size, input_dim)).float()
        reconstructed, latent = model_ce(test_input)
    
    # Verify shapes
    expected_output_shape = (batch_size, input_dim, num_classes)
    assert reconstructed.shape == expected_output_shape, f"Output shape mismatch: {reconstructed.shape} vs {expected_output_shape}"
    
    print(f"✅ Model forward pass successful!")
    print(f"  Input shape: {test_input.shape}")
    print(f"  Latent shape: {latent.shape}")
    print(f"  Output shape: {reconstructed.shape}")
    
    # Test that softmax outputs sum to 1 for each gene
    softmax_sums = reconstructed.sum(dim=2)
    assert torch.allclose(softmax_sums, torch.ones_like(softmax_sums), atol=1e-6), "Softmax outputs don't sum to 1"
    print(f"✅ Softmax outputs correctly normalized!")
    
    # Print model summary
    model_summary(model_ce, test_input.shape)
    
    return True


def test_accuracy_calculation():
    """Test accuracy calculation for different loss types."""
    print("Testing accuracy calculation for different loss types...")
    
    batch_size, num_genes = 16, 100
    
    # Test MSE accuracy
    original_continuous = torch.rand(batch_size, num_genes) * 5
    reconstructed_good = original_continuous + 0.01 * torch.randn_like(original_continuous)
    reconstructed_bad = original_continuous + 0.5 * torch.randn_like(original_continuous)
    
    acc_good_mse = calculate_accuracy(reconstructed_good, original_continuous, loss_type='mse')
    acc_bad_mse = calculate_accuracy(reconstructed_bad, original_continuous, loss_type='mse')
    
    print(f"MSE accuracy (good reconstruction): {acc_good_mse:.4f}")
    print(f"MSE accuracy (bad reconstruction): {acc_bad_mse:.4f}")
    
    # Test cross-entropy accuracy
    num_classes = 5
    original_discrete = torch.randint(0, num_classes, (batch_size, num_genes))
    
    # Perfect reconstruction
    reconstructed_perfect = original_discrete.float()
    acc_perfect_ce = calculate_accuracy(reconstructed_perfect, original_discrete, loss_type='cross_entropy')
    
    # Random reconstruction
    reconstructed_random = torch.randint(0, num_classes, (batch_size, num_genes)).float()
    acc_random_ce = calculate_accuracy(reconstructed_random, original_discrete, loss_type='cross_entropy')
    
    # Logits format (perfect prediction)
    logits_perfect = torch.zeros(batch_size, num_genes, num_classes)
    for i in range(batch_size):
        for j in range(num_genes):
            logits_perfect[i, j, original_discrete[i, j]] = 10.0  # High logit for correct class
    
    acc_logits_ce = calculate_accuracy(logits_perfect, original_discrete, loss_type='cross_entropy')
    
    print(f"Cross-entropy accuracy (perfect): {acc_perfect_ce:.4f}")
    print(f"Cross-entropy accuracy (random): {acc_random_ce:.4f}")
    print(f"Cross-entropy accuracy (logits): {acc_logits_ce:.4f}")
    
    print("✅ Accuracy calculation test completed!")
    return True


def test_training_compatibility():
    """Test that the new model is compatible with training loops."""
    print("\nTesting training compatibility...")
    
    # Create synthetic data
    batch_size, num_genes = 32, 200
    num_classes = 8
    
    # Create model
    model = SimpleAutoencoder(
        input_dim=num_genes,
        hidden_dims=[128, 64],
        latent_dim=32,
        dropout_rate=0.1,
        output_activation='softmax',
        num_classes=num_classes
    )
    
    # Create loss function
    loss_fn = get_loss_function('cross_entropy', num_classes=num_classes)
    
    # Create optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    # Create test data
    test_data = torch.randint(0, num_classes, (batch_size, num_genes))
    
    # Training step
    model.train()
    optimizer.zero_grad()
    
    reconstructed, latent = model(test_data.float())
    loss = loss_fn(reconstructed, test_data)
    
    loss.backward()
    optimizer.step()
    
    # Calculate accuracy
    accuracy = calculate_accuracy(reconstructed, test_data, loss_type='cross_entropy')
    
    print(f"Training step completed:")
    print(f"  Loss: {loss.item():.6f}")
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"✅ Training compatibility test passed!")
    
    return True


def main():
    """Run all tests for the cross-entropy loss functionality."""
    print("Testing Cross-Entropy Loss Functionality")
    print("=" * 50)
    
    tests = [
        test_cross_entropy_loss,
        test_loss_function_factory,
        test_autoencoder_with_cross_entropy,
        test_accuracy_calculation,
        test_training_compatibility
    ]
    
    passed = 0
    failed = 0
    
    for test_func in tests:
        try:
            test_func()
            passed += 1
        except Exception as e:
            print(f"❌ {test_func.__name__} failed: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
    
    print("\n" + "=" * 50)
    print(f"Test Results: {passed} passed, {failed} failed")
    
    if failed == 0:
        print("🎉 All tests passed! Cross-entropy loss functionality is working correctly.")
        return True
    else:
        print("⚠️  Some tests failed. Please check the implementation.")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
