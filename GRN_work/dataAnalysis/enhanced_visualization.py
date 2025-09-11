import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set style for better plots
plt.style.use('default')
sns.set_palette("husl")

# Paths to results - Updated to use absolute paths as specified
results_dirs = [
    r'C:\Users\rdbra\Documents\honoursProject\code_base\data\fc_autoencoder_sct_matrix_transposed_results',
    r'C:\Users\rdbra\Documents\honoursProject\code_base\data\fc_autoencoder_sct_matrix_transposed_results_20250819_025905'
]

def create_comprehensive_training_plot(history_path, model_name, output_dir):
    """Create comprehensive training plots including loss, accuracy, and learning rate"""
    with open(history_path, 'r') as f:
        history = json.load(f)
    
    epochs = list(range(1, len(history['losses']) + 1))
    
    # Create a figure with multiple subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(f'Training Analysis - {model_name}', fontsize=16)
    
    # Plot 1: Training Loss
    axes[0, 0].plot(epochs, history['losses'], label='Training Loss', color='blue', linewidth=2)
    if 'val_loss' in history:
        axes[0, 0].plot(epochs, history['val_loss'], label='Validation Loss', color='red', linewidth=2)
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Loss Over Time')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Training Accuracy
    if 'accuracies' in history:
        axes[0, 1].plot(epochs, history['accuracies'], label='Training Accuracy', color='green', linewidth=2)
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].set_title('Accuracy Over Time')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
    else:
        axes[0, 1].text(0.5, 0.5, 'No Accuracy Data Available', 
                       horizontalalignment='center', verticalalignment='center', 
                       transform=axes[0, 1].transAxes, fontsize=12)
        axes[0, 1].set_title('Accuracy Over Time')
    
    # Plot 3: Learning Rate Schedule
    if 'learning_rates' in history:
        axes[1, 0].plot(epochs, history['learning_rates'], label='Learning Rate', color='orange', linewidth=2)
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Learning Rate')
        axes[1, 0].set_title('Learning Rate Schedule')
        axes[1, 0].set_yscale('log')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
    else:
        axes[1, 0].text(0.5, 0.5, 'No Learning Rate Data Available', 
                       horizontalalignment='center', verticalalignment='center', 
                       transform=axes[1, 0].transAxes, fontsize=12)
        axes[1, 0].set_title('Learning Rate Schedule')
    
    # Plot 4: Loss Distribution (last 50 epochs)
    recent_losses = history['losses'][-50:] if len(history['losses']) >= 50 else history['losses']
    axes[1, 1].hist(recent_losses, bins=20, alpha=0.7, color='purple', edgecolor='black')
    axes[1, 1].set_xlabel('Loss Value')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].set_title('Loss Distribution (Last 50 Epochs)')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, f'{model_name}_comprehensive_training_analysis.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved comprehensive training analysis to: {output_path}")
    plt.show()

def plot_training_convergence(history_path, model_name, output_dir):
    """Plot training convergence analysis"""
    with open(history_path, 'r') as f:
        history = json.load(f)
    
    epochs = list(range(1, len(history['losses']) + 1))
    losses = history['losses']
    
    # Calculate moving average
    window = 10
    moving_avg = []
    for i in range(len(losses)):
        start_idx = max(0, i - window + 1)
        moving_avg.append(np.mean(losses[start_idx:i+1]))
    
    plt.figure(figsize=(12, 6))
    plt.plot(epochs, losses, alpha=0.7, label='Training Loss', color='lightblue')
    plt.plot(epochs, moving_avg, label=f'Moving Average (window={window})', color='darkblue', linewidth=2)
    
    # Mark minimum loss
    min_loss_idx = np.argmin(losses)
    plt.scatter(epochs[min_loss_idx], losses[min_loss_idx], color='red', s=100, zorder=5, 
                label=f'Min Loss: {losses[min_loss_idx]:.6f} (Epoch {epochs[min_loss_idx]})')
    
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'Training Convergence Analysis - {model_name}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, f'{model_name}_convergence_analysis.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved convergence analysis to: {output_path}")
    plt.show()

def plot_final_metrics_enhanced(metrics_path, model_name, output_dir):
    """Enhanced final metrics visualization"""
    metrics = {}
    with open(metrics_path, 'r') as f:
        for line in f:
            if ':' in line:
                key, value = line.strip().split(':', 1)
                try:
                    # Handle different number formats
                    value = value.strip().replace(',', '')  # Remove commas from large numbers
                    metrics[key.strip()] = float(value)
                except ValueError:
                    metrics[key.strip()] = value.strip()
    
    # Separate numeric and non-numeric metrics
    numeric_metrics = {k: v for k, v in metrics.items() if isinstance(v, (int, float))}
    text_metrics = {k: v for k, v in metrics.items() if not isinstance(v, (int, float))}
    
    if numeric_metrics:
        # Create different scales for different types of metrics
        loss_accuracy_metrics = {k: v for k, v in numeric_metrics.items() 
                               if any(term in k.lower() for term in ['loss', 'accuracy'])}
        other_metrics = {k: v for k, v in numeric_metrics.items() 
                        if k not in loss_accuracy_metrics}
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        fig.suptitle(f'Final Metrics - {model_name}', fontsize=16)
        
        # Plot loss and accuracy metrics
        if loss_accuracy_metrics:
            axes[0].bar(range(len(loss_accuracy_metrics)), list(loss_accuracy_metrics.values()), 
                       color=['red' if 'loss' in k.lower() else 'green' for k in loss_accuracy_metrics.keys()])
            axes[0].set_xticks(range(len(loss_accuracy_metrics)))
            axes[0].set_xticklabels(list(loss_accuracy_metrics.keys()), rotation=45)
            axes[0].set_ylabel('Value')
            axes[0].set_title('Loss and Accuracy Metrics')
            axes[0].grid(True, alpha=0.3)
            
            # Add value labels on bars
            for i, v in enumerate(loss_accuracy_metrics.values()):
                axes[0].text(i, v + max(loss_accuracy_metrics.values()) * 0.01, f'{v:.4f}', 
                           ha='center', va='bottom')
        
        # Plot other metrics (like parameter count)
        if other_metrics:
            axes[1].bar(range(len(other_metrics)), list(other_metrics.values()), color='orange')
            axes[1].set_xticks(range(len(other_metrics)))
            axes[1].set_xticklabels(list(other_metrics.keys()), rotation=45)
            axes[1].set_ylabel('Value')
            axes[1].set_title('Other Metrics')
            axes[1].grid(True, alpha=0.3)
            
            # Add value labels on bars
            for i, v in enumerate(other_metrics.values()):
                axes[1].text(i, v + max(other_metrics.values()) * 0.01, f'{v:,.0f}', 
                           ha='center', va='bottom')
        else:
            axes[1].text(0.5, 0.5, 'No Other Metrics Available', 
                        horizontalalignment='center', verticalalignment='center', 
                        transform=axes[1].transAxes, fontsize=12)
            axes[1].set_title('Other Metrics')
        
        plt.tight_layout()
        output_path = os.path.join(output_dir, f'{model_name}_enhanced_final_metrics.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved enhanced final metrics to: {output_path}")
        plt.show()
        
        # Print summary
        print(f"\nMetrics Summary for {model_name}:")
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                if 'parameters' in key.lower():
                    print(f"  {key}: {value:,.0f}")
                else:
                    print(f"  {key}: {value:.6f}")
            else:
                print(f"  {key}: {value}")
    else:
        print(f"No numeric metrics found for {model_name}")

# Main visualization
print("Starting Enhanced Training Results Visualization")
print("=" * 60)

for results_dir in results_dirs:
    if not os.path.exists(results_dir):
        print(f"Directory not found: {results_dir}")
        continue
        
    model_name = os.path.basename(results_dir)
    print(f"\nProcessing results for: {model_name}")
    print("-" * 40)
    
    # Create output directory for saving images in the results folder
    output_dir = os.path.join(results_dir, 'results')
    
    history_path = os.path.join(results_dir, 'results', 'training_history.json')
    metrics_path = os.path.join(results_dir, 'results', 'final_metrics.txt')
    npz_path = os.path.join(results_dir, 'results', 'reconstruction_data.npz')

    # Generate training visualizations
    if os.path.exists(history_path):
        print(f"✓ Found training history: {history_path}")
        create_comprehensive_training_plot(history_path, model_name, output_dir)
        plot_training_convergence(history_path, model_name, output_dir)
    else:
        print(f"✗ Training history not found: {history_path}")
        
    # Generate metrics visualizations
    if os.path.exists(metrics_path):
        print(f"✓ Found final metrics: {metrics_path}")
        plot_final_metrics_enhanced(metrics_path, model_name, output_dir)
    else:
        print(f"✗ Final metrics not found: {metrics_path}")
        
    # Check reconstruction data
    if os.path.exists(npz_path):
        print(f"✓ Found reconstruction data: {npz_path}")
        try:
            data = np.load(npz_path)
            print(f"  Available data keys: {list(data.keys())}")
            # Could add reconstruction analysis here if the file loads properly
        except Exception as e:
            print(f"  Warning: Could not load reconstruction data - {str(e)}")
    else:
        print(f"✗ Reconstruction data not found: {npz_path}")

print(f"\n{'='*60}")
print("Enhanced visualization complete!")
print("Images saved in respective results directories.")
print("Check the results folders for the following visualizations:")
print("  - Comprehensive training analysis")
print("  - Training convergence analysis") 
print("  - Enhanced final metrics")
