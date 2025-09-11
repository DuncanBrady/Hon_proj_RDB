"""
Create a synthetic gene expression dataset for testing the autoencoder.

This script generates a synthetic gene expression matrix that mimics real data
characteristics and saves it as an NPZ file.
"""

import numpy as np
import os

def create_synthetic_gene_data(n_cells=1000, n_genes=2000, n_cell_types=5, sparsity=0.8, seed=42):
    """
    Create synthetic gene expression data with realistic characteristics.
    
    Args:
        n_cells (int): Number of cells
        n_genes (int): Number of genes  
        n_cell_types (int): Number of distinct cell types
        sparsity (float): Fraction of zero values (0-1)
        seed (int): Random seed for reproducibility
        
    Returns:
        np.ndarray: Synthetic gene expression matrix (n_cells x n_genes)
    """
    np.random.seed(seed)
    
    # Create cell type assignments
    cells_per_type = n_cells // n_cell_types
    cell_types = np.repeat(range(n_cell_types), cells_per_type)
    
    # Add some extra cells to the last type if needed
    if len(cell_types) < n_cells:
        extra_cells = n_cells - len(cell_types)
        cell_types = np.concatenate([cell_types, np.full(extra_cells, n_cell_types-1)])
    
    # Initialize expression matrix
    expression_matrix = np.zeros((n_cells, n_genes), dtype=np.float32)
    
    # Create gene expression patterns for each cell type
    for cell_type in range(n_cell_types):
        # Identify cells of this type
        type_mask = cell_types == cell_type
        n_cells_type = np.sum(type_mask)
        
        # Create cell-type-specific gene expression
        # Some genes are highly expressed in this cell type
        n_marker_genes = n_genes // 10  # 10% marker genes per cell type
        marker_start = (cell_type * n_marker_genes) % n_genes
        marker_end = min(marker_start + n_marker_genes, n_genes)
        
        # Background expression (low level, sparse)
        background_expr = np.random.exponential(0.5, (n_cells_type, n_genes))
        background_expr = background_expr * (np.random.random((n_cells_type, n_genes)) > sparsity)
        
        # Marker gene expression (higher level)
        marker_expr = np.random.exponential(3.0, (n_cells_type, marker_end - marker_start))
        marker_expr = marker_expr * (np.random.random((n_cells_type, marker_end - marker_start)) > 0.3)
        
        # Combine background and marker expression
        expression_matrix[type_mask, :] = background_expr
        expression_matrix[type_mask, marker_start:marker_end] += marker_expr
        
        # Add some housekeeping genes (expressed in all cell types)
        housekeeping_genes = np.arange(0, min(100, n_genes))  # First 100 genes as housekeeping
        housekeeping_expr = np.random.exponential(2.0, (n_cells_type, len(housekeeping_genes)))
        housekeeping_expr = housekeeping_expr * (np.random.random((n_cells_type, len(housekeeping_genes))) > 0.5)
        expression_matrix[type_mask][:, housekeeping_genes] += housekeeping_expr
    
    # Add some noise
    noise = np.random.normal(0, 0.1, expression_matrix.shape)
    expression_matrix = np.maximum(0, expression_matrix + noise)  # Ensure non-negative
    
    return expression_matrix, cell_types


def main():
    """Create and save synthetic gene expression data."""
    print("Creating synthetic gene expression dataset...")
    
    # Create data directory if it doesn't exist
    data_dir = "data"
    os.makedirs(data_dir, exist_ok=True)
    
    # Generate synthetic data
    expression_data, cell_types = create_synthetic_gene_data(
        n_cells=1000,
        n_genes=2000,
        n_cell_types=5,
        sparsity=0.8,
        seed=42
    )
    
    print(f"Generated data shape: {expression_data.shape}")
    print(f"Data range: [{expression_data.min():.4f}, {expression_data.max():.4f}]")
    print(f"Sparsity: {100 * (expression_data == 0).sum() / expression_data.size:.1f}% zeros")
    print(f"Mean expression: {expression_data.mean():.4f}")
    print(f"Cell types: {len(np.unique(cell_types))} unique types")
    
    # Save as NPZ file
    npz_path = os.path.join(data_dir, "synthetic_gene_expression.npz")
    np.savez_compressed(npz_path, expression=expression_data, cell_types=cell_types)
    print(f"Data saved to: {npz_path}")
    
    # Also save as individual numpy arrays for convenience
    np.save(os.path.join(data_dir, "synthetic_expression_matrix.npy"), expression_data)
    np.save(os.path.join(data_dir, "synthetic_cell_types.npy"), cell_types)
    
    print("Dataset creation completed!")
    
    # Print some statistics
    print("\nDataset Statistics:")
    print(f"  Shape: {expression_data.shape} (cells x genes)")
    print(f"  Data type: {expression_data.dtype}")
    print(f"  Memory usage: {expression_data.nbytes / 1024**2:.1f} MB")
    print(f"  Non-zero values: {np.count_nonzero(expression_data):,}")
    print(f"  Zero values: {(expression_data == 0).sum():,}")
    print(f"  Sparsity: {100 * (expression_data == 0).sum() / expression_data.size:.1f}%")
    
    # Per-cell-type statistics
    print(f"\nPer-cell-type statistics:")
    for ct in range(len(np.unique(cell_types))):
        ct_mask = cell_types == ct
        ct_data = expression_data[ct_mask]
        print(f"  Cell type {ct}: {ct_mask.sum()} cells, "
              f"mean expr: {ct_data.mean():.4f}, "
              f"sparsity: {100 * (ct_data == 0).sum() / ct_data.size:.1f}%")


if __name__ == "__main__":
    main()
