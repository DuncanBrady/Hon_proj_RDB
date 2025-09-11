# Autoencoder Reconstruction Analysis

This document explains how to use the modified autoencoder training script to save reconstruction data and create confusion matrices for analyzing reconstruction quality.

## Overview

The training script has been enhanced to automatically save the final reconstructed matrix alongside the original data, enabling detailed analysis of the autoencoder's reconstruction performance through confusion matrices.

## What's New

### Modified Training Script (`train_simple_autoencoder.py`)

The training script now includes:

1. **Automatic Reconstruction Saving**: After training, the script automatically generates reconstructions for all input data and saves both original and reconstructed matrices.

2. **Comprehensive Metrics**: Calculates and saves detailed reconstruction metrics including MSE, MAE, R² score, and accuracy.

3. **Ready-to-Use Data Format**: Saves data in a format suitable for confusion matrix analysis.

### New Confusion Matrix Script (`create_confusion_matrix.py`)

A new script specifically designed to:

1. **Load Reconstruction Data**: Automatically loads the saved reconstruction data from training results.

2. **Create Confusion Matrices**: Generates confusion matrix visualizations for reconstruction quality analysis.

3. **Calculate Detailed Metrics**: Provides comprehensive metrics including precision, recall, and F1-scores per class/bin.

## Usage

### Step 1: Train the Autoencoder

Train your autoencoder as usual. The script will automatically save reconstruction data:

```bash
# Example training command
python train_simple_autoencoder.py \
    --data_path data/your_data.npz \
    --epochs 1000 \
    --batch_size 64 \
    --hidden_dims 1024 512 \
    --latent_dim 128 \
    --loss_type mse \
    --num_bins 10
```

This will create a results directory with the following structure:
```
fc_autoencoder_your_data_results_YYYYMMDD_HHMMSS/
├── models/
│   ├── autoencoder_model.pth
│   └── model_architecture.json
├── results/
│   ├── reconstruction_data.npz          # ← Original and reconstructed matrices
│   ├── reconstruction_metrics.json      # ← Detailed reconstruction metrics
│   ├── reconstruction_summary.txt       # ← Human-readable summary
│   ├── training_history.json
│   ├── config.json
│   └── final_metrics.txt
└── logs/
    └── training.log
```

### Step 2: Create Confusion Matrix

Use the confusion matrix script to analyze the reconstruction quality:

```bash
# Basic usage
python create_confusion_matrix.py --results_dir path/to/results/directory

# With subsampling for large datasets
python create_confusion_matrix.py \
    --results_dir path/to/results/directory \
    --subsample 10000

# Custom number of bins for continuous data
python create_confusion_matrix.py \
    --results_dir path/to/results/directory \
    --num_bins 8
```

This will add the following files to the results directory:
```
results/
├── confusion_matrix.png                 # ← Confusion matrix visualization
├── confusion_matrix_metrics.json        # ← Detailed confusion matrix metrics
└── confusion_matrix_summary.txt         # ← Human-readable confusion matrix summary
```

## Understanding the Results

### Reconstruction Data (`reconstruction_data.npz`)

Contains two arrays:
- `original`: The original input data matrix (n_samples × n_features)
- `reconstructed`: The autoencoder's reconstruction of the input data

```python
import numpy as np

# Load the data
data = np.load('reconstruction_data.npz')
original = data['original']
reconstructed = data['reconstructed']

print(f"Original shape: {original.shape}")
print(f"Reconstructed shape: {reconstructed.shape}")
```

### Reconstruction Metrics

Basic reconstruction quality metrics:
- **MSE**: Mean Squared Error between original and reconstructed
- **MAE**: Mean Absolute Error
- **R² Score**: Coefficient of determination
- **Accuracy**: Percentage of values within tolerance

### Confusion Matrix Analysis

For **continuous data (MSE/Huber loss)**:
- Data is discretized into bins for confusion matrix analysis
- Each bin represents a range of values
- Shows how well the autoencoder reconstructs different value ranges

For **discrete data (cross-entropy loss)**:
- Direct comparison of class predictions
- Shows per-class reconstruction accuracy

### Key Metrics from Confusion Matrix

- **Overall Accuracy**: Percentage of correctly reconstructed values/classes
- **Precision**: For each bin/class, what fraction of predictions were correct
- **Recall**: For each bin/class, what fraction of true values were correctly predicted
- **F1-Score**: Harmonic mean of precision and recall

## Tips for Analysis

### 1. For Gene Expression Data

```bash
# Use more bins for detailed analysis
python create_confusion_matrix.py \
    --results_dir your_results_dir \
    --num_bins 15
```

### 2. For Large Datasets

```bash
# Subsample for faster processing
python create_confusion_matrix.py \
    --results_dir your_results_dir \
    --subsample 50000
```

### 3. Interpreting Results

- **High diagonal values** in confusion matrix = good reconstruction
- **Scattered off-diagonal values** = poor reconstruction for those ranges
- **Class imbalance** may affect metrics - check per-class results

## Example Workflow

```bash
# 1. Train autoencoder
python train_simple_autoencoder.py \
    --data_path data/gene_expression.npz \
    --epochs 500 \
    --batch_size 128 \
    --loss_type cross_entropy \
    --num_bins 7

# 2. Results saved to: fc_autoencoder_gene_expression_results_20250819_123456/

# 3. Create confusion matrix
python create_confusion_matrix.py \
    --results_dir fc_autoencoder_gene_expression_results_20250819_123456

# 4. View results:
#    - reconstruction_summary.txt: Overall reconstruction quality
#    - confusion_matrix.png: Visual confusion matrix
#    - confusion_matrix_summary.txt: Detailed per-class metrics
```

## Custom Analysis

You can also load the reconstruction data directly for custom analysis:

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

# Load data
data = np.load('results/reconstruction_data.npz')
original = data['original']
reconstructed = data['reconstructed']

# Your custom analysis here
# For example, correlation analysis
correlation = np.corrcoef(original.flatten(), reconstructed.flatten())[0, 1]
print(f"Correlation: {correlation:.4f}")

# Or custom visualization
plt.scatter(original.flatten()[:1000], reconstructed.flatten()[:1000], alpha=0.5)
plt.xlabel('Original Values')
plt.ylabel('Reconstructed Values')
plt.title('Original vs Reconstructed Values')
plt.show()
```

This enhanced workflow provides comprehensive analysis capabilities for understanding how well your autoencoder reconstructs the input data, particularly useful for gene expression analysis where understanding reconstruction patterns across different expression levels is crucial.
