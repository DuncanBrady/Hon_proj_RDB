import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Paths to results - Updated to use absolute paths as specified
results_dirs = [
    r'C:\Users\rdbra\Documents\honoursProject\code_base\Hon_proj_RDB\GRN_work\results\fullData_FC_17_09',
    r'C:\Users\rdbra\Documents\honoursProject\code_base\Hon_proj_RDB\GRN_work\results\fc_autoencoder_sct_matrix_transposed_results',
    r'C:\Users\rdbra\Documents\honoursProject\code_base\Hon_proj_RDB\GRN_work\results\fc_autoencoder_sct_matrix_transposed_results_20250819_025905'
]

# Helper to plot training history
def plot_training_history(history_path, model_name, output_dir):
    with open(history_path, 'r') as f:
        history = json.load(f)
    epochs = list(range(1, len(history['losses']) + 1))
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, history['losses'], label='Training Loss')
    if 'val_loss' in history:
        plt.plot(epochs, history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'Training History - {model_name}')
    plt.legend()
    plt.tight_layout()
    output_path = os.path.join(output_dir, f'{model_name}_training_history.png')
    plt.savefig(output_path)
    print(f"Saved training history plot to: {output_path}")
    plt.show()

# Helper to plot final metrics
def plot_final_metrics(metrics_path, model_name, output_dir):
    metrics = {}
    with open(metrics_path, 'r') as f:
        for line in f:
            if ':' in line:
                key, value = line.strip().split(':', 1)
                try:
                    metrics[key] = float(value)
                except ValueError:
                    metrics[key] = value
    
    # Only plot numeric metrics
    numeric_metrics = {k: v for k, v in metrics.items() if isinstance(v, (int, float))}
    
    if numeric_metrics:
        plt.figure(figsize=(10, 6))
        sns.barplot(x=list(numeric_metrics.keys()), y=list(numeric_metrics.values()))
        plt.title(f'Final Metrics - {model_name}')
        plt.ylabel('Value')
        plt.xticks(rotation=45)
        plt.tight_layout()
        output_path = os.path.join(output_dir, f'{model_name}_final_metrics.png')
        plt.savefig(output_path)
        print(f"Saved final metrics plot to: {output_path}")
        plt.show()
    else:
        print(f"No numeric metrics found for {model_name}")

# Helper to plot reconstruction error histogram
def plot_reconstruction_histogram(npz_path, model_name, output_dir):
    data = np.load(npz_path)
    if 'reconstruction_error' in data:
        errors = data['reconstruction_error']
        plt.figure(figsize=(8, 5))
        sns.histplot(errors, bins=50, kde=True)
        plt.title(f'Reconstruction Error Histogram - {model_name}')
        plt.xlabel('Error')
        plt.ylabel('Frequency')
        plt.tight_layout()
        output_path = os.path.join(output_dir, f'{model_name}_reconstruction_error_hist.png')
        plt.savefig(output_path)
        print(f"Saved reconstruction error histogram to: {output_path}")
        plt.show()
    else:
        print(f"Available keys in {npz_path}: {list(data.keys())}")
        # Try to plot other reconstruction data if available
        for key in data.keys():
            if 'error' in key.lower() or 'reconstruction' in key.lower():
                values = data[key]
                if len(values.shape) == 1:  # Only plot 1D arrays
                    plt.figure(figsize=(8, 5))
                    sns.histplot(values, bins=50, kde=True)
                    plt.title(f'{key.replace("_", " ").title()} - {model_name}')
                    plt.xlabel('Value')
                    plt.ylabel('Frequency')
                    plt.tight_layout()
                    output_path = os.path.join(output_dir, f'{model_name}_{key}_hist.png')
                    plt.savefig(output_path)
                    print(f"Saved {key} histogram to: {output_path}")
                    plt.show()

# Main visualisation
for results_dir in results_dirs:
    if not os.path.exists(results_dir):
        print(f"Directory not found: {results_dir}")
        continue
        
    model_name = os.path.basename(results_dir)
    print(f"\nProcessing results for: {model_name}")
    
    # Create output directory for saving images in the results folder
    output_dir = os.path.join(results_dir, 'results')
    
    history_path = os.path.join(results_dir, 'results', 'training_history.json')
    metrics_path = os.path.join(results_dir, 'results', 'final_metrics.txt')
    npz_path = os.path.join(results_dir, 'results', 'reconstruction_data.npz')

    if os.path.exists(history_path):
        print(f"Found training history: {history_path}")
        plot_training_history(history_path, model_name, output_dir)
    else:
        print(f"Training history not found: {history_path}")
        
    if os.path.exists(metrics_path):
        print(f"Found final metrics: {metrics_path}")
        plot_final_metrics(metrics_path, model_name, output_dir)
    else:
        print(f"Final metrics not found: {metrics_path}")
        
    if os.path.exists(npz_path):
        print(f"Found reconstruction data: {npz_path}")
        plot_reconstruction_histogram(npz_path, model_name, output_dir)
    else:
        print(f"Reconstruction data not found: {npz_path}")

print('\nVisualization complete! Images saved in respective results directories.')
