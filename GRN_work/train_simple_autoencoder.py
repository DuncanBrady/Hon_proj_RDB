"""
Simple Autoencoder Training Script

This script trains a fully connected autoencoder on gene expression data with the following workflow:
1. Load and preprocess gene expression data from NPZ file
2. Apply binning to the gene expression values
3. Train a simple autoencoder to reconstruct the gene expression matrix
4. Save the trained model and results

Usage:
    python train_simple_autoencoder.py --data_path /path/to/data.npz [OPTIONS]

Example:
    python train_simple_autoencoder.py --data_path data/sct_matrix_transposed.npz --epochs 1000 --batch_size 64
"""

import argparse
import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import logging
from pathlib import Path
import json

# Import our custom modules
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from src.model.fc_autoencoder import SimpleAutoencoder, get_loss_function, calculate_accuracy, model_summary
from src.preprocess.binning import term_freq_bin


class GeneExpressionDataset(Dataset):
    """Dataset class for gene expression data."""
    
    def __init__(self, data):
        """
        Initialize dataset.
        
        Args:
            data (np.ndarray): Gene expression matrix (n_cells x n_genes)
        """
        self.data = torch.tensor(data, dtype=torch.float32)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]


def setup_logging(log_dir):
    """Setup logging configuration."""
    log_file = os.path.join(log_dir, 'training.log')
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    return logging.getLogger(__name__)


def load_and_preprocess_data(data_path, num_bins, logger):
    """
    Load gene expression data and apply binning.
    
    Args:
        data_path (str): Path to NPZ file containing gene expression data
        num_bins (int): Number of bins for term frequency binning
        logger: Logger instance
        
    Returns:
        np.ndarray: Processed gene expression matrix
    """
    logger.info(f"Loading data from: {data_path}")
    
    # Load NPZ file
    try:
        data = np.load(data_path)
        if isinstance(data, np.lib.npyio.NpzFile):
            # If it's an NPZ file with multiple arrays, get the first one
            key = list(data.keys())[0]
            gene_expression = data[key]
            logger.info(f"Loaded array '{key}' from NPZ file")
        else:
            gene_expression = data
            logger.info("Loaded single array from NPZ file")
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        raise
    
    logger.info(f"Original data shape: {gene_expression.shape}")
    logger.info(f"Data type: {gene_expression.dtype}")
    logger.info(f"Data range: [{gene_expression.min():.4f}, {gene_expression.max():.4f}]")
    logger.info(f"Non-zero values: {np.count_nonzero(gene_expression):,} / {gene_expression.size:,} "
                f"({100 * np.count_nonzero(gene_expression) / gene_expression.size:.2f}%)")
    
    # Apply binning
    logger.info(f"Applying term frequency binning with {num_bins} bins...")
    binned_data = term_freq_bin(gene_expression.copy(), num_bins)
    
    logger.info(f"After binning - Data range: [{binned_data.min():.4f}, {binned_data.max():.4f}]")
    logger.info(f"Unique values after binning: {len(np.unique(binned_data))}")
    
    return binned_data


def create_results_directory(data_path, model_type="fc_autoencoder"):
    """
    Create results directory structure with unique timestamp.
    
    Args:
        data_path (str): Path to the data file
        model_type (str): Type of model being trained
        
    Returns:
        str: Path to the created results directory
    """
    # Get the directory containing the data file
    data_dir = os.path.dirname(os.path.abspath(data_path))
    
    # Extract dataset name from file path
    dataset_name = Path(data_path).stem
    
    # Create timestamp for unique naming
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    
    # Create results directory name with timestamp
    results_dir_name = f"{model_type}_{dataset_name}_results_{timestamp}"
    results_dir = os.path.join(data_dir, results_dir_name)
    
    # Create directory structure
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(os.path.join(results_dir, "models"), exist_ok=True)
    os.makedirs(os.path.join(results_dir, "results"), exist_ok=True)
    os.makedirs(os.path.join(results_dir, "logs"), exist_ok=True)
    
    return results_dir


def train_autoencoder(dataloader, model, device, optimizer, scheduler, logger, num_epochs, loss_function, loss_type='mse'):
    """
    Train the autoencoder model.
    
    Args:
        dataloader: PyTorch DataLoader
        model: Autoencoder model
        device: Device to train on
        optimizer: Optimizer
        scheduler: Learning rate scheduler
        logger: Logger instance
        num_epochs: Number of training epochs
        loss_function: Loss function to use
        loss_type: Type of loss function for accuracy calculation
        
    Returns:
        dict: Training history
    """
    model.train()
    history = {
        'epochs': [],
        'losses': [],
        'accuracies': [],
        'learning_rates': []
    }
    
    logger.info("Starting training...")
    start_time = time.time()
    
    for epoch in range(1, num_epochs + 1):
        epoch_loss = 0.0
        epoch_accuracy = 0.0
        num_batches = 0
        
        for batch_data in dataloader:
            batch_data = batch_data.to(device)
            
            # Forward pass
            reconstructed, latent = model(batch_data)
            
            # Calculate loss
            loss = loss_function(reconstructed, batch_data)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Update learning rate
            if scheduler:
                scheduler.step()
            
            # Calculate metrics
            with torch.no_grad():
                accuracy = calculate_accuracy(reconstructed, batch_data, loss_type=loss_type)
            
            epoch_loss += loss.item()
            epoch_accuracy += accuracy
            num_batches += 1
        
        # Calculate averages
        avg_loss = epoch_loss / num_batches
        avg_accuracy = epoch_accuracy / num_batches
        current_lr = optimizer.param_groups[0]['lr']
        
        # Store history
        history['epochs'].append(epoch)
        history['losses'].append(avg_loss)
        history['accuracies'].append(avg_accuracy)
        history['learning_rates'].append(current_lr)
        
        # Log progress
        if epoch % 10 == 0 or epoch == 1:
            elapsed_time = time.time() - start_time
            eta = elapsed_time * (num_epochs - epoch) / epoch if epoch > 0 else 0
            
            logger.info(f"Epoch {epoch:4d}/{num_epochs} | "
                       f"Loss: {avg_loss:.6f} | "
                       f"Accuracy: {avg_accuracy:.4f} | "
                       f"LR: {current_lr:.2e} | "
                       f"ETA: {eta/60:.1f}m")
    
    total_time = time.time() - start_time
    logger.info(f"Training completed in {total_time/60:.1f} minutes")
    
    return history


def calculate_reconstruction_metrics(original, reconstructed, loss_type='mse'):
    """
    Calculate comprehensive reconstruction metrics.
    
    Args:
        original (np.ndarray): Original data matrix
        reconstructed (np.ndarray): Reconstructed data matrix  
        loss_type (str): Type of loss function used
        
    Returns:
        dict: Dictionary containing various reconstruction metrics
    """
    # Basic regression metrics
    mse = np.mean((original - reconstructed) ** 2)
    mae = np.mean(np.abs(original - reconstructed))
    
    # R² score
    ss_res = np.sum((original - reconstructed) ** 2)
    ss_tot = np.sum((original - np.mean(original)) ** 2)
    r2_score = 1 - (ss_res / (ss_tot + 1e-8))
    
    # Accuracy based on loss type
    if loss_type == 'cross_entropy':
        # For cross-entropy, compare argmax predictions
        original_classes = np.argmax(original, axis=1) if original.ndim > 1 and original.shape[1] > 1 else original.round().astype(int)
        reconstructed_classes = np.argmax(reconstructed, axis=1) if reconstructed.ndim > 1 and reconstructed.shape[1] > 1 else reconstructed.round().astype(int)
        class_accuracy = np.mean(original_classes == reconstructed_classes)
        
        # Use a threshold for "close enough" predictions
        threshold = 0.1  # 10% tolerance
        accuracy = np.mean(np.abs(original - reconstructed) < threshold)
        
        metrics = {
            'mse': float(mse),
            'mae': float(mae),
            'r2_score': float(r2_score),
            'accuracy': float(accuracy),
            'class_accuracy': float(class_accuracy)
        }
    else:
        # For MSE/Huber, use relative tolerance
        relative_tolerance = 0.1  # 10% relative tolerance
        absolute_tolerance = 0.05  # absolute tolerance for small values
        
        # Calculate tolerance per element
        tolerance = np.maximum(absolute_tolerance, relative_tolerance * np.abs(original))
        accuracy = np.mean(np.abs(original - reconstructed) <= tolerance)
        
        metrics = {
            'mse': float(mse),
            'mae': float(mae),
            'r2_score': float(r2_score),
            'accuracy': float(accuracy)
        }
    
    return metrics


def save_results(model, history, config, results_dir, logger, dataloader=None, device=None, original_data=None):
    """Save model, training history, configuration, and reconstructed matrix."""
    
    # Save model
    model_path = os.path.join(results_dir, "models", "autoencoder_model.pth")
    torch.save(model.state_dict(), model_path)
    logger.info(f"Model saved to: {model_path}")
    
    # Generate final reconstruction if dataloader is provided
    if dataloader is not None and device is not None:
        logger.info("Generating final reconstruction matrix...")
        model.eval()
        
        all_original = []
        all_reconstructed = []
        
        with torch.no_grad():
            for batch_data in dataloader:
                batch_data = batch_data.to(device)
                reconstructed, _ = model(batch_data)
                
                all_original.append(batch_data.cpu().numpy())
                all_reconstructed.append(reconstructed.cpu().numpy())
        
        # Concatenate all batches
        original_matrix = np.concatenate(all_original, axis=0)
        reconstructed_matrix = np.concatenate(all_reconstructed, axis=0)
        
        # Save original and reconstructed data
        reconstruction_path = os.path.join(results_dir, "results", "reconstruction_data.npz")
        np.savez_compressed(
            reconstruction_path,
            original=original_matrix,
            reconstructed=reconstructed_matrix
        )
        logger.info(f"Reconstruction matrices saved to: {reconstruction_path}")
        
        # Calculate and save reconstruction metrics
        reconstruction_metrics = calculate_reconstruction_metrics(original_matrix, reconstructed_matrix, config.get('loss_type', 'mse'))
        
        metrics_path = os.path.join(results_dir, "results", "reconstruction_metrics.json")
        with open(metrics_path, 'w') as f:
            json.dump(reconstruction_metrics, f, indent=2)
        logger.info(f"Reconstruction metrics saved to: {metrics_path}")
        
        # Save a summary of the reconstruction quality
        summary_path = os.path.join(results_dir, "results", "reconstruction_summary.txt")
        with open(summary_path, 'w') as f:
            f.write("Reconstruction Quality Summary\n")
            f.write("=" * 40 + "\n")
            f.write(f"Original data shape: {original_matrix.shape}\n")
            f.write(f"Reconstructed data shape: {reconstructed_matrix.shape}\n")
            f.write(f"Mean Squared Error: {reconstruction_metrics['mse']:.6f}\n")
            f.write(f"Mean Absolute Error: {reconstruction_metrics['mae']:.6f}\n")
            f.write(f"Reconstruction Accuracy: {reconstruction_metrics['accuracy']:.4f}\n")
            f.write(f"R² Score: {reconstruction_metrics['r2_score']:.4f}\n")
            if 'class_accuracy' in reconstruction_metrics:
                f.write(f"Classification Accuracy: {reconstruction_metrics['class_accuracy']:.4f}\n")
        logger.info(f"Reconstruction summary saved to: {summary_path}")
    
    # Save model architecture info
    model_info = {
        'input_dim': model.input_dim,
        'hidden_dims': model.hidden_dims,
        'latent_dim': model.latent_dim,
        'dropout_rate': model.dropout_rate,
        'output_activation': model.output_activation,
        'num_classes': getattr(model, 'num_classes', None),
        'total_parameters': sum(p.numel() for p in model.parameters())
    }
    
    model_info_path = os.path.join(results_dir, "models", "model_architecture.json")
    with open(model_info_path, 'w') as f:
        json.dump(model_info, f, indent=2)
    
    # Save training history
    history_path = os.path.join(results_dir, "results", "training_history.json")
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    logger.info(f"Training history saved to: {history_path}")
    
    # Save configuration
    config_path = os.path.join(results_dir, "results", "config.json")
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    logger.info(f"Configuration saved to: {config_path}")
    
    # Save final metrics as text
    final_metrics_path = os.path.join(results_dir, "results", "final_metrics.txt")
    with open(final_metrics_path, 'w') as f:
        f.write("Final Training Metrics\n")
        f.write("=" * 30 + "\n")
        f.write(f"Final Loss: {history['losses'][-1]:.6f}\n")
        f.write(f"Final Accuracy: {history['accuracies'][-1]:.4f}\n")
        f.write(f"Best Loss: {min(history['losses']):.6f}\n")
        f.write(f"Best Accuracy: {max(history['accuracies']):.4f}\n")
        f.write(f"Total Parameters: {model_info['total_parameters']:,}\n")
    
    logger.info("All results saved successfully!")


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description='Train Simple Autoencoder on Gene Expression Data')
    
    # Required arguments
    parser.add_argument('--data_path', type=str, required=True,
                       help='Path to NPZ file containing gene expression data')
    
    # Model architecture arguments
    parser.add_argument('--hidden_dims', type=int, nargs='+', default=[1024, 512],
                       help='Hidden layer dimensions for encoder (default: [1024, 512])')
    parser.add_argument('--latent_dim', type=int, default=128,
                       help='Latent space dimension (default: 128)')
    parser.add_argument('--dropout_rate', type=float, default=0.1,
                       help='Dropout rate (default: 0.1)')
    
    # Loss function arguments
    parser.add_argument('--loss_type', type=str, default='mse', 
                       choices=['mse', 'cross_entropy', 'huber'],
                       help='Loss function type (default: mse)')
    parser.add_argument('--num_classes', type=int, default=None,
                       help='Number of classes for cross-entropy loss (auto-detected if None)')
    parser.add_argument('--zero_weight', type=float, default=0.1,
                       help='Weight for zero values in loss function (default: 0.1)')
    parser.add_argument('--nonzero_weight', type=float, default=1.0,
                       help='Weight for non-zero values in loss function (default: 1.0)')
    parser.add_argument('--huber_delta', type=float, default=1.0,
                       help='Delta parameter for Huber loss (default: 1.0)')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=1000,
                       help='Number of training epochs (default: 1000)')
    parser.add_argument('--batch_size', type=int, default=64,
                       help='Batch size (default: 64)')
    parser.add_argument('--learning_rate', type=float, default=1e-3,
                       help='Initial learning rate (default: 1e-3)')
    parser.add_argument('--weight_decay', type=float, default=1e-5,
                       help='Weight decay for optimizer (default: 1e-5)')
    
    # Data preprocessing arguments
    parser.add_argument('--num_bins', type=int, default=10,
                       help='Number of bins for term frequency binning (default: 10)')
    
    # Other arguments
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed (default: 42)')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device to use (cuda/cpu/auto) (default: auto)')
    
    args = parser.parse_args()
    
    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Determine device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    # Create results directory
    results_dir = create_results_directory(args.data_path)
    
    # Setup logging
    logger = setup_logging(os.path.join(results_dir, "logs"))
    
    logger.info("=" * 60)
    logger.info("Simple Autoencoder Training")
    logger.info("=" * 60)
    logger.info(f"Device: {device}")
    logger.info(f"Results directory: {results_dir}")
    
    # Store configuration
    config = vars(args)
    config['device'] = str(device)
    config['results_dir'] = results_dir
    
    try:
        # Stage 1: Load and preprocess data
        gene_expression_data = load_and_preprocess_data(args.data_path, args.num_bins, logger)
        
        # Create dataset and dataloader
        dataset = GeneExpressionDataset(gene_expression_data)
        dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
        
        # Stage 2: Create and train model
        input_dim = gene_expression_data.shape[1]  # Number of genes
        
        # Determine model configuration based on loss type
        output_activation = 'softmax' if args.loss_type == 'cross_entropy' else 'none'
        num_classes = args.num_classes
        
        # Auto-detect number of classes for cross-entropy if not specified
        if args.loss_type == 'cross_entropy' and num_classes is None:
            num_classes = int(gene_expression_data.max()) + 1
            logger.info(f"Auto-detected number of classes: {num_classes}")
        
        model = SimpleAutoencoder(
            input_dim=input_dim,
            hidden_dims=args.hidden_dims,
            latent_dim=args.latent_dim,
            dropout_rate=args.dropout_rate,
            output_activation=output_activation,
            num_classes=num_classes
        )
        
        model = model.to(device)
        
        # Create loss function
        loss_function = get_loss_function(
            loss_type=args.loss_type,
            num_classes=num_classes,
            zero_weight=args.zero_weight,
            nonzero_weight=args.nonzero_weight,
            delta=args.huber_delta
        )
        
        logger.info(f"Using {args.loss_type} loss function")
        if args.loss_type == 'cross_entropy':
            logger.info(f"Number of classes: {num_classes}")
        
        # Print model summary
        model_summary(model, (args.batch_size, input_dim))
        
        # Setup optimizer and scheduler
        optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
        
        # Use OneCycleLR scheduler
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=args.learning_rate * 10,
            steps_per_epoch=len(dataloader),
            epochs=args.epochs,
            pct_start=0.1,
            anneal_strategy='cos'
        )
        
        # Train the model
        history = train_autoencoder(
            dataloader=dataloader,
            model=model,
            device=device,
            optimizer=optimizer,
            scheduler=scheduler,
            logger=logger,
            num_epochs=args.epochs,
            loss_function=loss_function,
            loss_type=args.loss_type
        )
        
        # Stage 3: Save results
        save_results(model, history, config, results_dir, logger, dataloader, device, gene_expression_data)
        
        logger.info("Training completed successfully!")
        
    except Exception as e:
        logger.error(f"Training failed with error: {e}")
        raise


if __name__ == "__main__":
    main()
