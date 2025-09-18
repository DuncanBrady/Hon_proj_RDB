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
from typing import Dict, Any

# Optional visualization libs (only used if available)
try:
    import matplotlib.pyplot as plt  # type: ignore
    import seaborn as sns  # type: ignore
    _HAS_PLOTTING = True
except Exception:  # pragma: no cover
    _HAS_PLOTTING = False

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


def _compute_per_bin_stats(original_batch: torch.Tensor, reconstructed_batch: torch.Tensor, num_bins: int, loss_type: str):
    """Compute per-bin accuracy and loss for a batch.

    For MSE/Huber we treat reconstruction as regression and map predictions back to nearest bin via digitization
    using the batch's (or provided) bin edges derived from original non-zero values. For cross-entropy the model
    output already represents class probabilities.

    Returns dict with:
      per_bin_correct, per_bin_total, per_bin_loss_sum (all length num_bins)
    """
    # Move to CPU for numpy operations
    # Force float32 to avoid upcasting to float64 in older PyTorch / NumPy interactions
    orig = original_batch.detach().to(torch.float32).cpu().numpy()
    # If cross-entropy: reconstructed may be (B, G, C). We'll keep tensor version for per-element loss.
    recon_tensor = reconstructed_batch.detach().to(torch.float32)
    recon = recon_tensor.cpu().numpy()

    # Determine bin indices for original data (bins start at 0; zero values remain 0)
    # We assume term_freq_bin produced integer bin labels 0..num_bins with 0 reserved for zeros; adjust if max == num_bins
    # Clip to [0, num_bins]
    orig_bins = np.clip(orig.astype(int), 0, num_bins)

    is_ce = loss_type in ['cross_entropy', 'ce'] and reconstructed_batch.dim() == 3
    if is_ce:
        # recon shape: (batch, genes, classes) -> predicted class indices via argmax
        pred_bins = recon_tensor.argmax(dim=2).detach().cpu().numpy()
    else:
        # For regression outputs, round to nearest int and clip
        pred_bins = np.clip(np.rint(recon).astype(int), 0, num_bins)

    # Per-element loss proxy:
    #  - For regression style (MSE/Huber) keep squared error.
    #  - For cross-entropy use negative log probability of the true class (NLL) so higher means worse.
    if is_ce:
        # recon_tensor: (B, G, C) already softmax probabilities (model applies softmax)
        # Need true class indices from orig (integers after binning). Ensure within range.
        true_classes = torch.clamp(original_batch.long(), 0, recon_tensor.shape[2]-1)
        # Gather probabilities of true class
        probs_true = torch.gather(recon_tensor, 2, true_classes.unsqueeze(-1)).squeeze(-1)
        # Numerical stability: clamp probabilities
        probs_true = torch.clamp(probs_true, 1e-8, 1.0)
        per_elem_loss = (-probs_true.log()).cpu().numpy()
    else:
        per_elem_loss = (recon - orig) ** 2

    per_bin_correct = np.zeros(num_bins + 1, dtype=np.int64)
    per_bin_total = np.zeros(num_bins + 1, dtype=np.int64)
    per_bin_loss_sum = np.zeros(num_bins + 1, dtype=np.float64)

    flat_orig = orig_bins.flatten()
    flat_pred = pred_bins.flatten()
    flat_loss = per_elem_loss.flatten()

    for b in range(num_bins + 1):
        mask = flat_orig == b
        if not np.any(mask):
            continue
        per_bin_total[b] += mask.sum()
        per_bin_correct[b] += (flat_pred[mask] == b).sum()
        per_bin_loss_sum[b] += flat_loss[mask].sum()

    return {
        'per_bin_correct': per_bin_correct,
        'per_bin_total': per_bin_total,
        'per_bin_loss_sum': per_bin_loss_sum
    }


def _accumulate_bin_stats(aggregate: Dict[str, np.ndarray], batch_stats: Dict[str, np.ndarray]):
    """Accumulate batch stats into aggregate containers."""
    if not aggregate:
        for k, v in batch_stats.items():
            aggregate[k] = v.copy()
    else:
        for k, v in batch_stats.items():
            aggregate[k] += v
    return aggregate


def _finalize_bin_metrics(aggregate: Dict[str, np.ndarray]):
    """Compute accuracy and mean loss per bin from aggregates."""
    per_bin_accuracy = []
    per_bin_mean_loss = []
    for correct, total, loss_sum in zip(aggregate['per_bin_correct'], aggregate['per_bin_total'], aggregate['per_bin_loss_sum']):
        if total == 0:
            per_bin_accuracy.append(None)
            per_bin_mean_loss.append(None)
        else:
            per_bin_accuracy.append(correct / total)
            per_bin_mean_loss.append(loss_sum / total)
    return per_bin_accuracy, per_bin_mean_loss


def _generate_confusion_matrix(aggregate: Dict[str, np.ndarray], num_bins: int):
    """Create pseudo confusion matrix using correct/total per bin where diagonal=correct and off-diagonal mass distributed uniformly among errors.
    This is a lightweight approximation to avoid storing all predictions. For detailed matrix, final reconstruction step handles full confusion.
    """
    cm = np.zeros((num_bins + 1, num_bins + 1), dtype=np.int64)
    # Only diagonal information available at epoch-level; errors allocated to 'other' bin (column 0) for visibility
    for b in range(num_bins + 1):
        total = aggregate['per_bin_total'][b]
        correct = aggregate['per_bin_correct'][b]
        if total == 0:
            continue
        cm[b, b] = correct
        cm[b, 0] += (total - correct)  # lump errors into column 0
    return cm


def _compute_binary_accuracy(original_batch: torch.Tensor, reconstructed_batch: torch.Tensor, loss_type: str, zero_threshold: float) -> float:
    """Compute binary accuracy (zero vs non-zero) for a batch.

    cross_entropy: treat class 0 as zero, others as non-zero (uses argmax over class dim if present).
    mse/huber: threshold both original and reconstructed using zero_threshold.
    """
    with torch.no_grad():
        if loss_type in ['cross_entropy', 'ce'] and reconstructed_batch.dim() == 3:
            preds = reconstructed_batch.argmax(dim=2)
            true = original_batch.long().clamp_min(0)
            pred_zero = preds.eq(0)
            true_zero = true.eq(0)
        else:
            pred_zero = reconstructed_batch <= zero_threshold
            true_zero = original_batch <= zero_threshold
        return (pred_zero == true_zero).float().mean().item()


def train_autoencoder(dataloader, model, device, optimizer, scheduler, logger, num_epochs, loss_function, loss_type='mse', num_bins: int = 10, track_epoch_confusion: bool = False):
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
    'binary_accuracies': [],
        'learning_rates': [],
        'per_bin_accuracy': [],  # list of lists (epoch -> bin accuracy)
        'per_bin_loss': [],      # list of lists (epoch -> bin mean loss)
        'epoch_confusion_matrices': []  # optional approximate matrices
    }
    
    logger.info("Starting training...")
    start_time = time.time()
    
    scaler = torch.cuda.amp.GradScaler(enabled=getattr(model, 'use_amp', False))

    best_loss = float('inf')
    best_epoch = 0
    patience = getattr(model, 'early_stopping_patience', None)
    min_delta = getattr(model, 'early_stopping_min_delta', 0.0)
    grad_clip = getattr(model, 'grad_clip_norm', None)

    zero_threshold = getattr(model, 'zero_threshold', 1e-8)

    for epoch in range(1, num_epochs + 1):
        epoch_loss = 0.0
        epoch_accuracy = 0.0
        epoch_binary_accuracy = 0.0
        num_batches = 0
        
        # Aggregate per-bin stats across batches for this epoch
        aggregate_bin_stats: Dict[str, np.ndarray] = {}

        for batch_data in dataloader:
            batch_data = batch_data.to(device)
            optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=getattr(model, 'use_amp', False)):
                reconstructed, latent = model(batch_data)
                loss = loss_function(reconstructed, batch_data)

            if getattr(model, 'use_amp', False):
                scaler.scale(loss).backward()
                if grad_clip is not None:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                if grad_clip is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()

            # Update learning rate
            if scheduler:
                scheduler.step()

            # Calculate metrics
            with torch.no_grad():
                accuracy = calculate_accuracy(reconstructed, batch_data, loss_type=loss_type)
                binary_acc = _compute_binary_accuracy(batch_data, reconstructed, loss_type=loss_type, zero_threshold=zero_threshold)
                # Per-bin stats
                batch_stats = _compute_per_bin_stats(batch_data, reconstructed, num_bins=num_bins, loss_type=loss_type)
                aggregate_bin_stats = _accumulate_bin_stats(aggregate_bin_stats, batch_stats)

            epoch_loss += loss.item()
            epoch_accuracy += accuracy
            epoch_binary_accuracy += binary_acc
            num_batches += 1
        
        avg_loss = epoch_loss / max(1, num_batches)
        avg_accuracy = epoch_accuracy / max(1, num_batches)
        avg_binary_accuracy = epoch_binary_accuracy / max(1, num_batches)
        current_lr = optimizer.param_groups[0]['lr']

        # Finalize per-bin metrics for epoch
        per_bin_accuracy, per_bin_mean_loss = _finalize_bin_metrics(aggregate_bin_stats)

        # Store history
        history['epochs'].append(epoch)
        history['losses'].append(avg_loss)
        history['accuracies'].append(avg_accuracy)
        history['learning_rates'].append(current_lr)
        history['binary_accuracies'].append(avg_binary_accuracy)
        history['per_bin_accuracy'].append(per_bin_accuracy)
        history['per_bin_loss'].append(per_bin_mean_loss)

        if track_epoch_confusion:
            cm = _generate_confusion_matrix(aggregate_bin_stats, num_bins)
            history['epoch_confusion_matrices'].append(cm.tolist())
        
        # Log progress
        if epoch % 10 == 0 or epoch == 1:
            elapsed_time = time.time() - start_time
            eta = elapsed_time * (num_epochs - epoch) / epoch if epoch > 0 else 0
            improvement = best_loss - avg_loss
            logger.info(f"Epoch {epoch:4d}/{num_epochs} | Loss: {avg_loss:.6f} (best {best_loss:.6f}) | "
                        f"Δbest: {improvement:.6f} | Acc: {avg_accuracy:.4f} | BinAcc: {avg_binary_accuracy:.4f} | LR: {current_lr:.2e} | ETA: {eta/60:.1f}m")

        # Early stopping check
        if avg_loss + min_delta < best_loss:
            best_loss = avg_loss
            best_epoch = epoch
        elif patience is not None and (epoch - best_epoch) >= patience:
            logger.info(f"Early stopping at epoch {epoch} (best epoch {best_epoch}, best loss {best_loss:.6f})")
            break
    
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
    # If cross-entropy with 3D reconstructed probs: reduce to class indices for metric compatibility
    recon_is_probs = loss_type == 'cross_entropy' and reconstructed.ndim == 3
    if recon_is_probs:
        # reconstructed: (cells, genes, classes)
        pred_class_matrix = reconstructed.argmax(axis=2)
        # For original, we expect integer class labels (cells, genes)
        original_classes_matrix = original if original.ndim == 2 else original.argmax(axis=2)
        # For regression-style metrics we compare integer labels to predicted labels
        diff_numeric = (original_classes_matrix - pred_class_matrix).astype(float)
        mse = float(np.mean(diff_numeric ** 2))
        mae = float(np.mean(np.abs(diff_numeric)))
    else:
        mse = float(np.mean((original - reconstructed) ** 2))
        mae = float(np.mean(np.abs(original - reconstructed)))
    
    # R² score
    if recon_is_probs:
        ss_res = float(np.sum(diff_numeric ** 2))
        ss_tot = float(np.sum((original_classes_matrix - np.mean(original_classes_matrix)) ** 2))
    else:
        ss_res = float(np.sum((original - reconstructed) ** 2))
        ss_tot = float(np.sum((original - np.mean(original)) ** 2))
    r2_score = 1 - (ss_res / (ss_tot + 1e-8))
    
    # Accuracy based on loss type
    if loss_type == 'cross_entropy':
        if recon_is_probs:
            original_classes = original_classes_matrix.astype(int)
            reconstructed_classes = pred_class_matrix.astype(int)
        else:
            # Fallback: treat arrays as already class-index matrices / vectors
            original_classes = original.round().astype(int)
            reconstructed_classes = reconstructed.round().astype(int)
        class_accuracy = float(np.mean(original_classes == reconstructed_classes))
        metrics = {
            'mse': mse,
            'mae': mae,
            'r2_score': float(r2_score),
            'class_accuracy': class_accuracy
        }
    else:
        # For MSE/Huber, use relative tolerance
        relative_tolerance = 0.1  # 10% relative tolerance
        absolute_tolerance = 0.05  # absolute tolerance for small values
        
        # Calculate tolerance per element
        tolerance = np.maximum(absolute_tolerance, relative_tolerance * np.abs(original))
        accuracy = np.mean(np.abs(original - reconstructed) <= tolerance)
        
        metrics = {
            'mse': mse,
            'mae': mae,
            'r2_score': float(r2_score),
            'accuracy': float(accuracy)
        }
    
    return metrics


def _save_confusion_matrix_full(original_matrix: np.ndarray, reconstructed_matrix: np.ndarray, results_dir: str, num_bins: int, loss_type: str, logger):
    """Compute and persist full confusion matrix between original and reconstructed bins/classes."""
    # Convert to bin labels
    orig_bins = np.clip(original_matrix.astype(int), 0, num_bins)
    if loss_type in ['cross_entropy', 'ce'] and reconstructed_matrix.ndim == 3:
        pred_bins = reconstructed_matrix.argmax(axis=2)
    else:
        pred_bins = np.clip(np.rint(reconstructed_matrix).astype(int), 0, num_bins)

    flat_orig = orig_bins.flatten()
    flat_pred = pred_bins.flatten()
    n_classes = num_bins + 1
    cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    for o, p in zip(flat_orig, flat_pred):
        cm[o, p] += 1

    cm_path = os.path.join(results_dir, 'results', 'confusion_matrix.npy')
    np.save(cm_path, cm)
    logger.info(f"Confusion matrix saved to: {cm_path}")

    # Derive per-class precision/recall/f1
    with np.errstate(divide='ignore', invalid='ignore'):
        tp = np.diag(cm)
        fp = cm.sum(axis=0) - tp
        fn = cm.sum(axis=1) - tp
        precision = np.divide(tp, tp + fp, out=np.zeros_like(tp, dtype=float), where=(tp + fp) != 0)
        recall = np.divide(tp, tp + fn, out=np.zeros_like(tp, dtype=float), where=(tp + fn) != 0)
        f1 = np.divide(2 * precision * recall, precision + recall, out=np.zeros_like(tp, dtype=float), where=(precision + recall) != 0)
    cm_metrics = {
        'overall_accuracy': float(tp.sum() / cm.sum() if cm.sum() else 0.0),
        'precision_per_bin': precision.tolist(),
        'recall_per_bin': recall.tolist(),
        'f1_per_bin': f1.tolist(),
        'macro_precision': float(np.mean(precision) if precision.size else 0.0),
        'macro_recall': float(np.mean(recall) if recall.size else 0.0),
        'macro_f1': float(np.mean(f1) if f1.size else 0.0)
    }
    cm_metrics_path = os.path.join(results_dir, 'results', 'confusion_matrix_metrics.json')
    with open(cm_metrics_path, 'w') as f:
        json.dump(cm_metrics, f, indent=2)
    logger.info(f"Confusion matrix metrics saved to: {cm_metrics_path}")

    # Optional plot
    if _HAS_PLOTTING:
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=False, cmap='Blues', cbar=True)
        plt.title('Autoencoder Reconstruction Confusion Matrix')
        plt.xlabel('Predicted Bin')
        plt.ylabel('True Bin')
        plt.tight_layout()
        plot_path = os.path.join(results_dir, 'results', 'confusion_matrix.png')
        plt.savefig(plot_path, dpi=300)
        plt.close()
        logger.info(f"Confusion matrix plot saved to: {plot_path}")

    return cm, cm_metrics


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
                if config.get('loss_type') in ['cross_entropy','ce'] and not config.get('save_full_reconstruction', False):
                    # Store only predicted class indices to save memory
                    if reconstructed.dim() == 3:
                        preds = reconstructed.argmax(dim=2).cpu().numpy().astype(np.int16)
                    else:
                        preds = reconstructed.round().clamp(min=0).cpu().numpy().astype(np.int16)
                    all_reconstructed.append(preds)
                else:
                    all_reconstructed.append(reconstructed.cpu().numpy())
        
        # Concatenate all batches
        original_matrix = np.concatenate(all_original, axis=0)
        reconstructed_matrix = np.concatenate(all_reconstructed, axis=0)
        
        # Save original and reconstructed data
        reconstruction_path = os.path.join(results_dir, "results", "reconstruction_data.npz")
        try:
            np.savez_compressed(
                reconstruction_path,
                original=original_matrix,
                reconstructed=reconstructed_matrix
            )
            logger.info(f"Reconstruction data saved to: {reconstruction_path}")
        except Exception as e:
            logger.warning(f"Failed to save full reconstruction ({e}); saving lightweight version.")
            np.savez_compressed(
                reconstruction_path,
                original_shape=original_matrix.shape,
                reconstructed_shape=reconstructed_matrix.shape
            )
        
        # Calculate and save reconstruction metrics
        reconstruction_metrics = calculate_reconstruction_metrics(original_matrix, reconstructed_matrix, config.get('loss_type', 'mse'))
        # Add per-bin final metrics (reuse helper)
        num_bins = config.get('num_bins', 10)
        # Compute per-bin stats on full dataset
        full_stats = _compute_per_bin_stats(torch.tensor(original_matrix), torch.tensor(reconstructed_matrix), num_bins=num_bins, loss_type=config.get('loss_type', 'mse'))
        final_per_bin_accuracy, final_per_bin_loss = _finalize_bin_metrics(full_stats)
        reconstruction_metrics['final_per_bin_accuracy'] = final_per_bin_accuracy
        reconstruction_metrics['final_per_bin_loss'] = final_per_bin_loss
        # Best accuracy per bin across epochs
        if history.get('per_bin_accuracy'):
            per_bin_acc_arr = np.array([[a if a is not None else np.nan for a in epoch_acc] for epoch_acc in history['per_bin_accuracy']])
            best_per_bin_accuracy = np.nanmax(per_bin_acc_arr, axis=0).tolist()
            reconstruction_metrics['best_per_bin_accuracy'] = best_per_bin_accuracy
        # Generate full confusion matrix on final reconstruction
        cm, cm_metrics = _save_confusion_matrix_full(original_matrix, reconstructed_matrix, results_dir, num_bins=num_bins, loss_type=config.get('loss_type', 'mse'), logger=logger)
        reconstruction_metrics['confusion_matrix_overall_accuracy'] = cm_metrics['overall_accuracy']
        
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
    
    # Persist extended history with per-bin metrics
    extended_history_path = os.path.join(results_dir, 'results', 'training_history_extended.json')
    with open(extended_history_path, 'w') as f:
        json.dump(history, f, indent=2)

    logger.info("All results saved successfully with extended metrics!")


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
    parser.add_argument('--loss_type', type=str, default='cross_entropy', 
                       choices=['mse', 'cross_entropy', 'huber'],
                       help='Loss function type (default: cross_entropy)')
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
    parser.add_argument('--early_stopping_patience', type=int, default=50,
                       help='Patience for early stopping (default: 50; disable with 0)')
    parser.add_argument('--early_stopping_min_delta', type=float, default=1e-5,
                       help='Minimum loss improvement to reset patience (default: 1e-5)')
    parser.add_argument('--grad_clip_norm', type=float, default=1.0,
                       help='Gradient clipping max norm (default: 1.0; disable with <=0)')
    parser.add_argument('--amp', action='store_true', help='Enable mixed precision training (AMP)')
    parser.add_argument('--auto_class_weights', action='store_true', help='Automatically compute inverse-frequency class weights for cross-entropy')
    parser.add_argument('--save_full_reconstruction', action='store_true', help='If set, save full reconstructed tensor (probabilities/logits). Default is to save only argmax class indices for CE to reduce memory.')
    
    # Data preprocessing arguments
    parser.add_argument('--num_bins', type=int, default=10,
                       help='Number of bins for term frequency binning (default: 10)')
    parser.add_argument('--track_epoch_confusion', action='store_true',
                       help='Track approximate epoch-level confusion matrices (diagonal + aggregated errors)')
    
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

        # Determine model configuration based on loss type (model now outputs logits; no softmax)
        output_activation = 'none'
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

        # Attach training control attributes
        model.use_amp = bool(args.amp and device.type == 'cuda')
        model.early_stopping_patience = None if args.early_stopping_patience <= 0 else args.early_stopping_patience
        model.early_stopping_min_delta = args.early_stopping_min_delta
        model.grad_clip_norm = None if args.grad_clip_norm is None or args.grad_clip_norm <= 0 else args.grad_clip_norm

        # Compute class weights if requested and cross-entropy
        class_weights = None
        if args.loss_type == 'cross_entropy' and args.auto_class_weights:
            flat = torch.tensor(gene_expression_data, dtype=torch.long).view(-1)
            counts = torch.bincount(flat, minlength=num_classes).float()
            freq = counts / counts.sum().clamp_min(1)
            inv = 1.0 / freq.clamp_min(1e-8)
            class_weights = inv / inv.sum() * num_classes  # normalize
            logger.info(f"Computed class weights: {class_weights.tolist()}")

        # Create loss function
        loss_function = get_loss_function(
            loss_type=args.loss_type,
            num_classes=num_classes,
            zero_weight=args.zero_weight,
            nonzero_weight=args.nonzero_weight,
            delta=args.huber_delta,
            class_weights=class_weights
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
            loss_type=args.loss_type,
            num_bins=args.num_bins,
            track_epoch_confusion=args.track_epoch_confusion
        )

        # Stage 3: Save results
        save_results(model, history, config, results_dir, logger, dataloader, device, gene_expression_data)

        logger.info("Training completed successfully!")

    except Exception as e:
        logger.error(f"Training failed with error: {e}")
        raise


if __name__ == "__main__":
    main()
