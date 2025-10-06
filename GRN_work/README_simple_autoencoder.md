# Simple Autoencoder for Gene Expression Reconstruction

This repository contains a simple fully connected autoencoder implementation designed for reconstructing gene expression matrices. The autoencoder learns latent representations of gene expression data and can be used for dimensionality reduction, denoising, and feature learning.

## Overview

The training workflow consists of three main stages:

1. **Data Loading and Processing**: Load gene expression data from NPZ files and apply term frequency binning
2. **Model Training**: Train a fully connected autoencoder to reconstruct the gene expression matrix
3. **Results Saving**: Save the trained model, training history, and configuration files

## Features

- **Simple Architecture**: Fully connected encoder-decoder architecture with configurable layers
- **Weighted Loss Function**: Emphasizes reconstruction of non-zero values while handling sparse data
- **Automatic Results Organization**: Creates structured output directories with models, results, and logs
- **Comprehensive Logging**: Detailed training logs and metrics tracking
- **Flexible Configuration**: Command-line arguments for all training parameters
- **Data Preprocessing**: Built-in term frequency binning for gene expression data

## Installation

Ensure you have the required dependencies installed:

```bash
pip install torch numpy pathlib argparse logging
```

## File Structure

```
GRN_work/
├── src/
│   ├── model/
│   │   └── fc_autoencoder.py          # Simple autoencoder model implementation
│   └── preprocess/
│       └── binning.py                 # Data preprocessing utilities
├── train_simple_autoencoder.py       # Main training script
├── example_usage_simple_autoencoder.py # Example usage script
└── README_simple_autoencoder.md       # This file
```

## Usage

### Basic Usage

```bash
python train_simple_autoencoder.py --data_path /path/to/your/data.npz
```

### Advanced Usage with Custom Parameters

```bash
python train_simple_autoencoder.py \
    --data_path /path/to/your/data.npz \
    --hidden_dims 2048 1024 512 \
    --latent_dim 256 \
    --epochs 1000 \
    --batch_size 128 \
    --learning_rate 1e-3 \
    --num_bins 20
```

### Command Line Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--data_path` | str | **Required** | Path to NPZ file containing gene expression data |
| `--hidden_dims` | int+ | [1024, 512] | Hidden layer dimensions for encoder |
| `--latent_dim` | int | 128 | Latent space dimension |
| `--dropout_rate` | float | 0.1 | Dropout rate for regularization |
| `--epochs` | int | 1000 | Number of training epochs |
| `--batch_size` | int | 64 | Batch size |
| `--learning_rate` | float | 1e-3 | Initial learning rate |
| `--weight_decay` | float | 1e-5 | Weight decay for optimizer |
| `--num_bins` | int | 10 | Number of bins for term frequency binning |
| `--seed` | int | 42 | Random seed |
| `--device` | str | auto | Device to use (cuda/cpu/auto) |

## Data Format

The input data should be:
- Stored in NPZ format (NumPy compressed array)
- Shape: `(n_cells, n_genes)` - cells as rows, genes as columns
- Non-negative values (gene expression levels)

Example data loading:
```python
import numpy as np
data = np.load('your_data.npz')['arr_0']  # or appropriate key
print(f"Data shape: {data.shape}")  # Should be (n_cells, n_genes)
```

## Output Structure

The training script automatically creates a results directory with the following structure:

```
{model_type}_{dataset_name}_results/
├── models/
│   ├── autoencoder_model.pth          # Trained model weights
│   └── model_architecture.json        # Model architecture details
├── results/
│   ├── training_history.json          # Training metrics over epochs
│   ├── config.json                     # Full training configuration
│   └── final_metrics.txt               # Summary of final metrics
└── logs/
    └── training.log                    # Detailed training logs
```

## Model Architecture

The autoencoder consists of:

### Encoder
- Fully connected layers with configurable dimensions
- Batch normalization after each hidden layer
- ReLU activation functions
- Dropout for regularization
- Final layer maps to latent space

### Decoder
- Symmetric architecture to encoder (reversed layer sizes)
- Batch normalization and ReLU activations
- Final layer reconstructs input dimensions
- No activation on output layer (allows negative values)

### Loss Function
- Weighted Mean Squared Error (MSE)
- Higher weight for non-zero values (default: 1.0)
- Lower weight for zero values (default: 0.1)
- Handles sparse gene expression data effectively

## Training Process

1. **Data Preprocessing**: Apply term frequency binning to discretize gene expression values
2. **Model Initialization**: Create autoencoder with specified architecture
3. **Training Loop**: 
   - Forward pass through encoder-decoder
   - Calculate weighted MSE loss
   - Backpropagation and parameter updates
   - Learning rate scheduling with OneCycleLR
4. **Monitoring**: Track loss, accuracy, and learning rate
5. **Saving**: Store model, metrics, and configuration

## Example Usage Script

Run the provided example script to see different training configurations:

```bash
python example_usage_simple_autoencoder.py
```

This will run several examples:
- Basic training with default parameters
- Small model for quick testing
- Large model with high capacity
- Custom binning configuration

## Monitoring Training

The training script provides real-time monitoring:
- Loss and accuracy printed every 10 epochs
- Estimated time remaining (ETA)
- Learning rate tracking
- GPU memory usage (if using CUDA)

Example output:
```
Epoch   10/1000 | Loss: 0.045623 | Accuracy: 0.8234 | LR: 2.34e-03 | ETA: 45.2m
Epoch   20/1000 | Loss: 0.042156 | Accuracy: 0.8456 | LR: 4.12e-03 | ETA: 44.1m
```

## Extended Reconstruction Metrics (Per-Bin Analysis)

The training pipeline now produces detailed, bin-level metrics to help you diagnose where reconstruction succeeds or fails. These are especially useful when using binned gene expression data.

Generated artifacts (in `results/`):

| File | Description |
|------|-------------|
| `training_history_extended.json` | Includes per-epoch `per_bin_accuracy`, `per_bin_loss`, and optional approximate epoch confusion matrices |
| `reconstruction_metrics.json` | Adds `final_per_bin_accuracy`, `final_per_bin_loss`, and `best_per_bin_accuracy` across epochs |
| `confusion_matrix.npy` | Full confusion matrix (rows=true bins, cols=predicted bins) |
| `confusion_matrix_metrics.json` | Overall / per-bin precision, recall, F1, macro scores |
| `confusion_matrix.png` | Heatmap (if matplotlib + seaborn installed) |

### Per-Bin Metrics Definitions
* Per-bin accuracy: fraction of elements from a true bin that were reconstructed (rounded / argmax) into the same bin.
* Per-bin loss: mean squared error of elements belonging to that bin (independent of weighting used in the training loss for interpretability).
* Best per-bin accuracy: maximum accuracy achieved for each bin over all epochs (helps identify under-fit bins).

### Confusion Matrix
The final confusion matrix is computed on the full reconstructed dataset. For cross-entropy models it uses argmax over class probabilities; for MSE/Huber it rounds predictions to nearest integer bin.

### Optional Epoch-Level Confusion Tracking
Use `--track_epoch_confusion` to store lightweight approximate confusion matrices per epoch (diagonal + aggregated errors). This keeps memory usage low while enabling temporal evolution analysis.

### Enabling Features
Add the flag during training:
```bash
python train_simple_autoencoder.py --data_path data.npz --num_bins 10 --track_epoch_confusion
```

### Plotting Dependencies
Install plotting extras (already added to `requirements_simple_autoencoder.txt`):
```
pip install matplotlib seaborn
```

### Programmatic Access Example
```python
import json, numpy as np
hist = json.load(open('results/training_history_extended.json'))
final_per_bin_acc = json.load(open('results/reconstruction_metrics.json'))['final_per_bin_accuracy']
cm = np.load('results/confusion_matrix.npy')
```

These metrics make it straightforward to: (1) detect bins with systematic under-performance, (2) compare reconstruction fidelity across expression intensity levels, and (3) guide re-weighting or architectural adjustments.


## Model Loading and Inference

To load a trained model for inference:

```python
import torch
from src.model.fc_autoencoder import SimpleAutoencoder

# Load model architecture info
import json
with open('results/models/model_architecture.json', 'r') as f:
    model_info = json.load(f)

# Create model with same architecture
model = SimpleAutoencoder(
    input_dim=model_info['input_dim'],
    hidden_dims=model_info['hidden_dims'],
    latent_dim=model_info['latent_dim'],
    dropout_rate=model_info['dropout_rate']
)

# Load trained weights
model.load_state_dict(torch.load('results/models/autoencoder_model.pth'))
model.eval()

# Use for inference
with torch.no_grad():
    # For encoding only
    latent_representation = model.encode(input_data)
    
    # For full reconstruction
    reconstructed_data, latent_repr = model(input_data)
```

## Customization

### Custom Loss Functions
Modify the `weighted_mse_loss` function in `fc_autoencoder.py` to implement different loss functions:

```python
def custom_loss_function(reconstructed, original):
    # Implement your custom loss here
    return loss_value
```

### Custom Architectures
Extend the `SimpleAutoencoder` class to implement different architectures:

```python
class CustomAutoencoder(SimpleAutoencoder):
    def __init__(self, input_dim, **kwargs):
        super().__init__(input_dim, **kwargs)
        # Add custom layers or modifications
```

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   - Reduce batch size: `--batch_size 32`
   - Reduce model size: `--hidden_dims 512 256`

2. **Poor Reconstruction Quality**
   - Increase model capacity: `--hidden_dims 2048 1024 512`
   - Increase latent dimension: `--latent_dim 256`
   - Adjust binning: `--num_bins 20`

3. **Training Too Slow**
   - Increase batch size: `--batch_size 128`
   - Reduce epochs: `--epochs 500`
   - Use GPU if available

4. **Data Loading Errors**
   - Verify NPZ file format and structure
   - Check data path and file permissions
   - Ensure data is in correct shape (n_cells, n_genes)

## Performance Tips

1. **Use GPU**: Ensure CUDA is available for faster training
2. **Batch Size**: Larger batches generally train faster but require more memory
3. **Learning Rate**: Start with 1e-3 and adjust based on convergence
4. **Architecture**: Start with smaller models and increase capacity as needed

## Citation

If you use this autoencoder implementation in your research, please cite:

```
Simple Autoencoder for Gene Expression Reconstruction
[Your Name/Institution]
[Year]
```

## License

[Specify your license here]

## Contributing

[Contribution guidelines if applicable]

## Contact

[Your contact information]
