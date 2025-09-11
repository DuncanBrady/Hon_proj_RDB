"""
Create Confusion Matrix from Autoencoder Reconstruction Results

This script loads the reconstruction data saved by the autoencoder training script
and creates confusion matrices to analyze the reconstruction quality.

Usage:
    python create_confusion_matrix.py --results_dir path/to/results/directory
    
Example:
    python create_confusion_matrix.py --results_dir data/fc_autoencoder_synthetic_gene_expression_results_20250819_025528
"""

import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from pathlib import Path
import json


def load_reconstruction_data(results_dir):
    """
    Load the reconstruction data from the results directory.
    
    Args:
        results_dir (str): Path to the results directory
        
    Returns:
        tuple: (original_data, reconstructed_data, config)
    """
    # Load reconstruction data
    recon_path = os.path.join(results_dir, "results", "reconstruction_data.npz")
    if not os.path.exists(recon_path):
        raise FileNotFoundError(f"Reconstruction data not found at: {recon_path}")
    
    data = np.load(recon_path)
    original = data['original']
    reconstructed = data['reconstructed']
    
    # Load configuration to understand the model setup
    config_path = os.path.join(results_dir, "results", "config.json")
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    return original, reconstructed, config


def prepare_data_for_confusion_matrix(original, reconstructed, loss_type='mse', num_bins=None):
    """
    Prepare data for confusion matrix based on the loss type used.
    
    Args:
        original (np.ndarray): Original data matrix
        reconstructed (np.ndarray): Reconstructed data matrix
        loss_type (str): Type of loss function used in training
        num_bins (int): Number of bins to discretize continuous data
        
    Returns:
        tuple: (y_true, y_pred, labels)
    """
    if loss_type == 'cross_entropy':
        # For cross-entropy, data should already be in class format
        if original.ndim > 1 and original.shape[1] > 1:
            # Multi-class probabilities - take argmax
            y_true = np.argmax(original, axis=1)
            y_pred = np.argmax(reconstructed, axis=1)
        else:
            # Single values - round to nearest integer
            y_true = original.round().astype(int).flatten()
            y_pred = reconstructed.round().astype(int).flatten()
        
        labels = np.arange(max(y_true.max(), y_pred.max()) + 1)
        
    else:
        # For MSE/Huber, discretize the continuous values
        if num_bins is None:
            # Use unique values if reasonable number, otherwise default to 10 bins
            unique_vals = len(np.unique(original))
            num_bins = min(unique_vals, 10) if unique_vals <= 20 else 10
        
        # Create bins based on original data range
        bin_edges = np.linspace(original.min(), original.max(), num_bins + 1)
        
        # Discretize both original and reconstructed data
        y_true = np.digitize(original.flatten(), bin_edges) - 1
        y_pred = np.digitize(reconstructed.flatten(), bin_edges) - 1
        
        # Clip to valid range
        y_true = np.clip(y_true, 0, num_bins - 1)
        y_pred = np.clip(y_pred, 0, num_bins - 1)
        
        labels = [f"Bin {i}" for i in range(num_bins)]
    
    return y_true, y_pred, labels


def create_confusion_matrix_plot(y_true, y_pred, labels, save_path, title="Confusion Matrix"):
    """
    Create and save a confusion matrix plot.
    
    Args:
        y_true (np.ndarray): True labels
        y_pred (np.ndarray): Predicted labels
        labels (list): Label names
        save_path (str): Path to save the plot
        title (str): Title for the plot
    """
    # Calculate confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=range(len(labels)))
    
    # Create the plot
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=labels, yticklabels=labels)
    plt.title(title)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return cm


def calculate_confusion_metrics(cm):
    """
    Calculate metrics from confusion matrix.
    
    Args:
        cm (np.ndarray): Confusion matrix
        
    Returns:
        dict: Dictionary of metrics
    """
    # Overall accuracy
    accuracy = np.trace(cm) / np.sum(cm)
    
    # Per-class metrics
    n_classes = cm.shape[0]
    precision = np.zeros(n_classes)
    recall = np.zeros(n_classes)
    f1_score = np.zeros(n_classes)
    
    for i in range(n_classes):
        # True positives for class i
        tp = cm[i, i]
        # False positives for class i (predicted as i but not actually i)
        fp = np.sum(cm[:, i]) - tp
        # False negatives for class i (actually i but not predicted as i)
        fn = np.sum(cm[i, :]) - tp
        
        # Calculate metrics
        precision[i] = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall[i] = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1_score[i] = 2 * (precision[i] * recall[i]) / (precision[i] + recall[i]) if (precision[i] + recall[i]) > 0 else 0
    
    return {
        'accuracy': accuracy,
        'precision_per_class': precision.tolist(),
        'recall_per_class': recall.tolist(),
        'f1_score_per_class': f1_score.tolist(),
        'macro_precision': np.mean(precision),
        'macro_recall': np.mean(recall),
        'macro_f1': np.mean(f1_score)
    }


def main():
    """Main function to create confusion matrix analysis."""
    parser = argparse.ArgumentParser(description='Create Confusion Matrix from Autoencoder Reconstruction')
    
    parser.add_argument('--results_dir', type=str, required=True,
                       help='Path to the autoencoder results directory')
    parser.add_argument('--num_bins', type=int, default=None,
                       help='Number of bins for discretizing continuous data (auto-detected if None)')
    parser.add_argument('--subsample', type=int, default=None,
                       help='Subsample data points for faster processing (use all data if None)')
    
    args = parser.parse_args()
    
    print("Loading reconstruction data...")
    original, reconstructed, config = load_reconstruction_data(args.results_dir)
    
    print(f"Data shape: {original.shape}")
    print(f"Loss type: {config['loss_type']}")
    
    # Subsample if requested
    if args.subsample and args.subsample < original.size:
        print(f"Subsampling {args.subsample} data points...")
        indices = np.random.choice(original.size, args.subsample, replace=False)
        original_flat = original.flatten()[indices]
        reconstructed_flat = reconstructed.flatten()[indices]
        original = original_flat.reshape(-1, 1)
        reconstructed = reconstructed_flat.reshape(-1, 1)
    
    # Prepare data for confusion matrix
    print("Preparing data for confusion matrix...")
    y_true, y_pred, labels = prepare_data_for_confusion_matrix(
        original, reconstructed, config['loss_type'], args.num_bins
    )
    
    print(f"Number of classes/bins: {len(labels)}")
    print(f"Data points: {len(y_true):,}")
    
    # Create confusion matrix plot
    cm_save_path = os.path.join(args.results_dir, "results", "confusion_matrix.png")
    print(f"Creating confusion matrix plot: {cm_save_path}")
    
    title = f"Autoencoder Reconstruction Confusion Matrix\n({config['loss_type'].upper()} Loss)"
    cm = create_confusion_matrix_plot(y_true, y_pred, labels, cm_save_path, title)
    
    # Calculate metrics
    print("Calculating confusion matrix metrics...")
    metrics = calculate_confusion_metrics(cm)
    
    # Save metrics
    metrics_save_path = os.path.join(args.results_dir, "results", "confusion_matrix_metrics.json")
    with open(metrics_save_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    # Print summary
    print("\n" + "="*50)
    print("CONFUSION MATRIX ANALYSIS RESULTS")
    print("="*50)
    print(f"Overall Accuracy: {metrics['accuracy']:.4f}")
    print(f"Macro Precision: {metrics['macro_precision']:.4f}")
    print(f"Macro Recall: {metrics['macro_recall']:.4f}")
    print(f"Macro F1-Score: {metrics['macro_f1']:.4f}")
    
    print(f"\nConfusion matrix saved to: {cm_save_path}")
    print(f"Detailed metrics saved to: {metrics_save_path}")
    
    # Save a summary report
    summary_path = os.path.join(args.results_dir, "results", "confusion_matrix_summary.txt")
    with open(summary_path, 'w') as f:
        f.write("Confusion Matrix Analysis Summary\n")
        f.write("=" * 40 + "\n")
        f.write(f"Dataset: {Path(args.results_dir).name}\n")
        f.write(f"Loss Type: {config['loss_type']}\n")
        f.write(f"Data Shape: {original.shape}\n")
        f.write(f"Number of Classes/Bins: {len(labels)}\n")
        f.write(f"Total Data Points: {len(y_true):,}\n\n")
        
        f.write("Metrics:\n")
        f.write(f"  Overall Accuracy: {metrics['accuracy']:.4f}\n")
        f.write(f"  Macro Precision: {metrics['macro_precision']:.4f}\n")
        f.write(f"  Macro Recall: {metrics['macro_recall']:.4f}\n")
        f.write(f"  Macro F1-Score: {metrics['macro_f1']:.4f}\n\n")
        
        f.write("Per-class Metrics:\n")
        for i, label in enumerate(labels):
            f.write(f"  {label}:\n")
            f.write(f"    Precision: {metrics['precision_per_class'][i]:.4f}\n")
            f.write(f"    Recall: {metrics['recall_per_class'][i]:.4f}\n")
            f.write(f"    F1-Score: {metrics['f1_score_per_class'][i]:.4f}\n")
    
    print(f"Summary report saved to: {summary_path}")
    print("\nAnalysis complete!")


if __name__ == "__main__":
    main()
