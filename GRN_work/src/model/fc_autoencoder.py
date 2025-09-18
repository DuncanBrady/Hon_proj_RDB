"""
Simple Fully Connected Autoencoder for Gene Expression Reconstruction

This module implements a simple fully connected autoencoder architecture
designed for reconstructing gene expression matrices.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class SimpleAutoencoder(nn.Module):
    """
    A simple fully connected autoencoder for gene expression reconstruction.
    
    Args:
        input_dim (int): Number of input features (genes)
        hidden_dims (list): List of hidden layer dimensions for encoder
        latent_dim (int): Dimension of the latent representation
        dropout_rate (float): Dropout rate for regularization
        output_activation (str): Output activation ('none', 'relu', 'softmax')
        num_classes (int, optional): Number of classes for classification output
    """
    
    def __init__(self, input_dim, hidden_dims=[1024, 512], latent_dim=128, 
                 dropout_rate=0.1, output_activation='none', num_classes=None):
        super(SimpleAutoencoder, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.latent_dim = latent_dim
        self.dropout_rate = dropout_rate
        self.output_activation = output_activation
        self.num_classes = num_classes
        
        # Build encoder
        encoder_layers = []
        current_dim = input_dim
        
        for hidden_dim in hidden_dims:
            encoder_layers.extend([
                nn.Linear(current_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ])
            current_dim = hidden_dim
        
        # Final encoder layer to latent space
        encoder_layers.extend([
            nn.Linear(current_dim, latent_dim),
            nn.ReLU()
        ])
        
        self.encoder = nn.Sequential(*encoder_layers)
        
        # Build decoder (symmetric to encoder)
        decoder_layers = []
        current_dim = latent_dim
        
        for hidden_dim in reversed(hidden_dims):
            decoder_layers.extend([
                nn.Linear(current_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ])
            current_dim = hidden_dim
        
        # Final decoder layer to output space
        if num_classes is not None:
            # For classification: output raw logits (no softmax here; apply in loss/metrics if needed)
            decoder_layers.append(nn.Linear(current_dim, input_dim * num_classes))
        else:
            decoder_layers.append(nn.Linear(current_dim, input_dim))

        # Only apply ReLU for explicit 'relu'; ignore 'softmax' here to keep logits
        if output_activation == 'relu' and num_classes is None:
            decoder_layers.append(nn.ReLU())
        
        self.decoder = nn.Sequential(*decoder_layers)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Initialize model weights using Kaiming normal initialization."""
        if isinstance(module, nn.Linear):
            nn.init.kaiming_normal_(module.weight, nonlinearity='relu')
            if module.bias is not None:
                nn.init.zeros_(module.bias)
    
    def forward(self, x):
        """
        Forward pass through the autoencoder.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_dim)
            
        Returns:
            tuple: (reconstructed, latent_representation)
        """
        latent = self.encoder(x)
        decoded = self.decoder(latent)
        
        # Handle different output formats
        if self.num_classes is not None:
            # Reshape to (batch_size, input_dim, num_classes). Keep as raw logits.
            batch_size = decoded.shape[0]
            decoded = decoded.view(batch_size, self.input_dim, self.num_classes)
        
        return decoded, latent
    
    def encode(self, x):
        """Encode input to latent representation."""
        return self.encoder(x)
    
    def decode(self, latent):
        """Decode latent representation to reconstruction."""
        decoded = self.decoder(latent)
        
        # Handle different output formats
        if self.num_classes is not None:
            batch_size = decoded.shape[0]
            decoded = decoded.view(batch_size, self.input_dim, self.num_classes)
        
        return decoded


def weighted_mse_loss(reconstructed, original, zero_weight=0.1, nonzero_weight=1.0):
    """
    Weighted MSE loss that gives different weights to zero and non-zero values.
    
    Args:
        reconstructed (torch.Tensor): Reconstructed gene expression
        original (torch.Tensor): Original gene expression
        zero_weight (float): Weight for zero values
        nonzero_weight (float): Weight for non-zero values
        
    Returns:
        torch.Tensor: Weighted MSE loss
    """
    # Create weight mask
    weight = torch.where(original > 0, nonzero_weight, zero_weight).to(original.device)
    
    # Calculate weighted MSE
    mse = (reconstructed - original) ** 2
    weighted_mse = weight * mse
    
    return torch.mean(weighted_mse)


def cross_entropy_reconstruction_loss(reconstructed, original, num_classes=None, zero_weight=0.1, nonzero_weight=1.0, class_weights=None):
    """
    Cross-entropy reconstruction loss for discretized gene expression data.
    
    This loss treats gene expression reconstruction as a classification problem,
    where each gene's expression level is predicted as one of several discrete classes.
    This is particularly useful when working with binned gene expression data.
    
    Args:
        reconstructed (torch.Tensor): Raw logits from model (batch_size, num_genes, num_classes)
                                     or (batch_size, num_genes) if num_classes is inferred
        original (torch.Tensor): Original gene expression values as class indices (batch_size, num_genes)
        num_classes (int, optional): Number of expression level classes. If None, inferred from data.
        zero_weight (float): Weight for zero expression values
        nonzero_weight (float): Weight for non-zero expression values
        
    Returns:
        torch.Tensor: Weighted cross-entropy loss
    """
    # Infer number of classes if not provided
    if num_classes is None:
        num_classes = int(original.max().item()) + 1
    
    # Expect reconstructed as logits (batch, genes, classes) or (batch, genes)
    if reconstructed.dim() == 2:
        # Expand single value per gene into identical logits across classes (degenerate)
        batch_size, num_genes = reconstructed.shape
        reconstructed = reconstructed.view(batch_size, num_genes, 1).expand(batch_size, num_genes, num_classes)

    # Ensure original is long tensor
    original = original.long()
    
    # Create weight mask based on zero/non-zero values
    weight_mask = torch.where(original > 0, nonzero_weight, zero_weight).to(original.device)
    
    # Calculate cross-entropy loss for each gene
    # Reshape for cross-entropy: (batch_size * num_genes, num_classes) and (batch_size * num_genes,)
    batch_size, num_genes = original.shape
    
    if reconstructed.dim() == 3:
        reconstructed_flat = reconstructed.view(batch_size * num_genes, num_classes)
    else:
        reconstructed_flat = reconstructed.view(batch_size * num_genes, -1)
        if reconstructed_flat.shape[1] != num_classes:
            # Fallback expansion
            reconstructed_flat = reconstructed_flat.expand(batch_size * num_genes, num_classes)
    
    original_flat = original.view(batch_size * num_genes)
    weight_flat = weight_mask.view(batch_size * num_genes)
    
    # Calculate cross-entropy loss
    weight_vec = None
    if class_weights is not None:
        # class_weights expected as 1D tensor length num_classes
        weight_vec = class_weights.to(reconstructed_flat.device, dtype=reconstructed_flat.dtype)
    ce_loss = F.cross_entropy(reconstructed_flat, original_flat, weight=weight_vec, reduction='none')
    
    # Apply weights
    weighted_ce_loss = ce_loss * weight_flat
    
    return torch.mean(weighted_ce_loss)


def huber_reconstruction_loss(reconstructed, original, delta=1.0, zero_weight=0.1, nonzero_weight=1.0):
    """
    Huber loss for reconstruction, which is less sensitive to outliers than MSE.
    
    Args:
        reconstructed (torch.Tensor): Reconstructed gene expression
        original (torch.Tensor): Original gene expression
        delta (float): Threshold for switching between squared and linear loss
        zero_weight (float): Weight for zero values
        nonzero_weight (float): Weight for non-zero values
        
    Returns:
        torch.Tensor: Weighted Huber loss
    """
    # Create weight mask
    weight = torch.where(original > 0, nonzero_weight, zero_weight).to(original.device)
    
    # Calculate Huber loss
    error = reconstructed - original
    is_small_error = torch.abs(error) <= delta
    squared_loss = 0.5 * (error ** 2)
    linear_loss = delta * (torch.abs(error) - 0.5 * delta)
    huber_loss = torch.where(is_small_error, squared_loss, linear_loss)
    
    # Apply weights
    weighted_huber_loss = weight * huber_loss
    
    return torch.mean(weighted_huber_loss)


def get_loss_function(loss_type='mse', **kwargs):
    """
    Factory function to get the specified loss function with its parameters.
    
    Args:
        loss_type (str): Type of loss function ('mse', 'cross_entropy', 'huber')
        **kwargs: Additional parameters for the loss function
        
    Returns:
        callable: Loss function that takes (reconstructed, original) as arguments
    """
    if loss_type.lower() in ['mse', 'weighted_mse']:
        zero_weight = kwargs.get('zero_weight', 0.1)
        nonzero_weight = kwargs.get('nonzero_weight', 1.0)
        return lambda reconstructed, original: weighted_mse_loss(
            reconstructed, original, zero_weight, nonzero_weight
        )
    
    elif loss_type.lower() in ['cross_entropy', 'ce']:
        num_classes = kwargs.get('num_classes', None)
        zero_weight = kwargs.get('zero_weight', 0.1)
        nonzero_weight = kwargs.get('nonzero_weight', 1.0)
        class_weights = kwargs.get('class_weights', None)
        return lambda reconstructed, original: cross_entropy_reconstruction_loss(
            reconstructed, original, num_classes, zero_weight, nonzero_weight, class_weights
        )
    
    elif loss_type.lower() == 'huber':
        delta = kwargs.get('delta', 1.0)
        zero_weight = kwargs.get('zero_weight', 0.1)
        nonzero_weight = kwargs.get('nonzero_weight', 1.0)
        return lambda reconstructed, original: huber_reconstruction_loss(
            reconstructed, original, delta, zero_weight, nonzero_weight
        )
    
    else:
        raise ValueError(f"Unknown loss type: {loss_type}. Choose from 'mse', 'cross_entropy', 'huber'")


def calculate_accuracy(reconstructed, original, threshold_factor=0.1, loss_type='mse'):
    """
    Calculate accuracy based on loss type and dynamic threshold.
    
    Args:
        reconstructed (torch.Tensor): Reconstructed gene expression
        original (torch.Tensor): Original gene expression
        threshold_factor (float): Factor for dynamic threshold calculation (for MSE/Huber)
        loss_type (str): Type of loss function being used
        
    Returns:
        float: Accuracy percentage
    """
    # Ensure dtype alignment (older torch versions are stricter with torch.where)
    if reconstructed.dtype != original.dtype:
        original = original.to(reconstructed.dtype)

    if loss_type.lower() in ['cross_entropy', 'ce']:
        # Reconstructed expected as logits (batch, genes, classes) now
        if reconstructed.dim() == 3:
            predicted_classes = reconstructed.argmax(dim=2)
        else:
            predicted_classes = torch.round(torch.clamp(reconstructed, 0, None))
        
        # Ensure original is the same type for comparison
        original_classes = original.long() if original.dtype != torch.long else original
        predicted_classes = predicted_classes.long()
        
        # Calculate exact match accuracy
        correct = (predicted_classes == original_classes)
        accuracy = correct.float().mean().item()
        
    else:
        # For MSE and Huber loss, use threshold-based accuracy
        # Dynamic threshold based on original values (create tensors with matching dtype/device)
        zero_tol = torch.full_like(original, 0.05)
        thresh_factor = torch.as_tensor(threshold_factor, dtype=original.dtype, device=original.device)
        threshold = torch.where(
            original == 0,
            zero_tol,
            thresh_factor * torch.abs(original)
        )

        # Calculate accuracy
        diff = torch.abs(reconstructed - original)
        correct = diff < threshold
        accuracy = correct.float().mean().item()
    
    return accuracy


def model_summary(model, input_shape):
    """Print a summary of the model architecture."""
    print(f"\n{'='*60}")
    print(f"{'Model Architecture Summary':^60}")
    print(f"{'='*60}")
    print(f"Input dimension: {model.input_dim}")
    print(f"Hidden dimensions: {model.hidden_dims}")
    print(f"Latent dimension: {model.latent_dim}")
    print(f"Dropout rate: {model.dropout_rate}")
    print(f"Output activation: {model.output_activation}")
    if hasattr(model, 'num_classes') and model.num_classes is not None:
        print(f"Number of classes: {model.num_classes}")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Test forward pass to get shapes - use eval mode to avoid BatchNorm issues
    model.eval()
    with torch.no_grad():
        # Use batch size of 2 to avoid BatchNorm issues
        # Get the device of the model
        device = next(model.parameters()).device
        dummy_input = torch.randn(2, model.input_dim, device=device)
        reconstructed, latent = model(dummy_input)
        print(f"\nInput shape: {dummy_input.shape}")
        print(f"Latent shape: {latent.shape}")
        print(f"Output shape: {reconstructed.shape}")
        
        if hasattr(model, 'num_classes') and model.num_classes is not None and model.output_activation == 'softmax':
            print(f"Output format: (batch_size, genes, classes)")
        else:
            print(f"Output format: (batch_size, genes)")
    model.train()  # Set back to training mode
    
    print(f"{'='*60}\n")
    
    print(f"{'='*60}\n")
