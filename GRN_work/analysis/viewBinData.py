import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# --- Data Generation and Binning Helper Functions ---

def generate_sample_expression_data(num_samples=1000, num_genes=50):
    """Generates a sample gene expression dataset."""
    # Create bimodal distribution for more interesting visualization
    np.random.seed(42)
    data1 = np.random.normal(loc=2.5, scale=1.5, size=(num_samples // 2, num_genes))
    data2 = np.random.normal(loc=8.0, scale=2.0, size=(num_samples // 2, num_genes))
    data = np.vstack([data1, data2])
    np.random.shuffle(data)
    
    gene_names = [f'Gene_{i+1}' for i in range(num_genes)]
    sample_names = [f'Sample_{i+1}' for i in range(num_samples)]
    
    return pd.DataFrame(data, index=sample_names, columns=gene_names)

def apply_binning(data, num_bins):
    """Applies equal-width binning to the dataset."""
    # The `pd.cut` function is a powerful tool for binning.
    # It segments and sorts data values into bins.
    binned_data = data.apply(lambda x: pd.cut(x, bins=num_bins, labels=False, include_lowest=True))
    return binned_data

# --- Visualization Functions ---

def plot_histograms(original_series, binned_series, feature_name, num_bins):
    """
    Plots side-by-side histograms of original vs. binned data.
    
    Args:
        original_series (pd.Series): The original continuous data.
        binned_series (pd.Series): The data after binning.
        feature_name (str): The name of the data column/feature.
        num_bins (int): The number of bins used.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot original data histogram
    sns.histplot(original_series, ax=ax1, kde=False, bins=30)
    ax1.set_title(f'Original Distribution of {feature_name}')
    ax1.set_xlabel('Expression Value')
    ax1.set_ylabel('Frequency')
    
    # Plot binned data histogram
    sns.histplot(binned_series, ax=ax2, discrete=True)
    ax2.set_title(f'Binned Distribution of {feature_name} ({num_bins} Bins)')
    ax2.set_xlabel('Bin Number')
    ax2.set_ylabel('Frequency')
    
    plt.suptitle(f'Histogram Comparison for {feature_name}', fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

def plot_density_overlay(original_series, binned_series, feature_name):
    """
    Overlays the original data's density plot on the binned histogram.
    
    Args:
        original_series (pd.Series): The original continuous data.
        binned_series (pd.Series): The data after binning.
        feature_name (str): The name of the data column/feature.
    """
    plt.figure(figsize=(10, 6))
    
    # Plot binned data as a histogram
    ax = sns.histplot(binned_series, discrete=True, stat='density', label='Binned Data')
    ax.set_xlabel('Bin Number')
    
    # Create a second y-axis for the original data's density
    ax2 = ax.twinx()
    sns.kdeplot(original_series, ax=ax2, color='r', label='Original Data KDE')
    
    ax.set_title(f'Density Overlay for {feature_name}')
    plt.legend()
    plt.show()

def plot_boxplots_by_bin(original_series, binned_series, feature_name):
    """
    Shows boxplots of the original data distribution within each bin.

    Args:
        original_series (pd.Series): The original continuous data.
        binned_series (pd.Series): The data after binning.
        feature_name (str): The name of the data column/feature.
    """
    df_combined = pd.DataFrame({
        'original_values': original_series,
        'bin_assignment': binned_series
    })
    
    plt.figure(figsize=(12, 7))
    sns.boxplot(x='bin_assignment', y='original_values', data=df_combined)
    plt.title(f'Distribution of Original Values within Each Bin for {feature_name}')
    plt.xlabel('Bin Number')
    plt.ylabel('Original Expression Value')
    plt.show()

def plot_scatter_comparison(original_df, x_feature, y_feature, num_bins):
    """
    Compares a standard scatter plot with a 2D binned hexbin plot.

    Args:
        original_df (pd.DataFrame): DataFrame containing the original data.
        x_feature (str): Name of the feature for the x-axis.
        y_feature (str): Name of the feature for the y-axis.
        num_bins (int): The grid size for the hexbin plot.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    
    # Standard scatter plot
    sns.scatterplot(data=original_df, x=x_feature, y=y_feature, ax=ax1, alpha=0.5)
    ax1.set_title(f'Original Scatter Plot: {x_feature} vs {y_feature}')
    
    # 2D Binned (Hexbin) plot
    # A higher gridsize means smaller (more) bins
    hb = ax2.hexbin(original_df[x_feature], original_df[y_feature], gridsize=num_bins, cmap='viridis')
    ax2.set_title(f'Hexagonal Binning: {x_feature} vs {y_feature}')
    ax2.set_xlabel(x_feature)
    ax2.set_ylabel(y_feature)
    cb = fig.colorbar(hb, ax=ax2)
    cb.set_label('counts')
    
    plt.suptitle('Scatter Plot vs. 2D Binning', fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

def plot_heatmap_comparison(original_df, binned_df):
    """
    Plots side-by-side heatmaps of the original and binned datasets.

    Args:
        original_df (pd.DataFrame): The original data.
        binned_df (pd.DataFrame): The binned data.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
    
    # Use a subset for clarity if the dataset is too large
    subset_size = min(50, original_df.shape[1])
    original_subset = original_df.iloc[:subset_size, :subset_size]
    binned_subset = binned_df.iloc[:subset_size, :subset_size]

    sns.heatmap(original_subset, ax=ax1, cmap='viridis', cbar_kws={'label': 'Expression Value'})
    ax1.set_title('Original Expression Data')
    ax1.set_xlabel('Genes')
    ax1.set_ylabel('Samples')

    sns.heatmap(binned_subset, ax=ax2, cmap='viridis', cbar_kws={'label': 'Bin Number'})
    ax2.set_title('Binned Expression Data')
    ax2.set_xlabel('Genes')
    ax2.set_ylabel('Samples')
    
    plt.suptitle('Heatmap Comparison', fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()


# --- Main Execution: Example Usage ---

if __name__ == '__main__':
    # 1. Define Parameters
    NUM_BINS = 10
    
    #Load data from npz files
    fullDataPath = "C:/Users/rdbra/Documents/honoursProject/code_base/Hon_proj_RDB/GRN_work/data/sct_matrix_transposed.npz"
    top5kDataPath = "C:/Users/rdbra/Documents/honoursProject/code_base/Hon_proj_RDB/GRN_work/data/sct_top5k.npz"
    fullData = np.load(fullDataPath)
    top5kData = np.load(top5kDataPath)
    
    # Reshape top5kData to have samples as rows and genes as columns
    # Convert top5kData['data'] from (genes x cells) to (cells x genes)
    top5k_genes_x_cells = top5kData['data']            # shape: (G, C)
    print("Original (genes x cells):", top5k_genes_x_cells.shape)

    top5k_matrix = top5k_genes_x_cells.T        # view transpose, no copy unless modified
    print("Transposed (cells x genes):", top5k_matrix.shape)

    # Optional: keep convenient references to metadata (unchanged)
    top5k_gene_names = top5kData['genes']
    top5k_cell_ids = top5kData['cells']

    # 3. Apply a Binning Algorithm
    binned_expression_data = apply_binning(top5k_matrix, num_bins=NUM_BINS)
    print("\nBinned Data Head:")
    print(binned_expression_data.head())

    # 4. Visualize the Effects
    
    # Select a single gene to visualize its distribution
    gene_to_visualize = 'Gene_1'
    
    print(f"\n--- Generating visualizations for {gene_to_visualize} ---")
    
    # Histogram comparison
    plot_histograms(
        original_series=original_expression_data[gene_to_visualize],
        binned_series=binned_expression_data[gene_to_visualize],
        feature_name=gene_to_visualize,
        num_bins=NUM_BINS
    )
    
    # Density overlay
    plot_density_overlay(
        original_series=original_expression_data[gene_to_visualize],
        binned_series=binned_expression_data[gene_to_visualize],
        feature_name=gene_to_visualize
    )
    
    # Box plots for each bin
    plot_boxplots_by_bin(
        original_series=original_expression_data[gene_to_visualize],
        binned_series=binned_expression_data[gene_to_visualize],
        feature_name=gene_to_visualize
    )
    
    # Select two genes for scatter plot comparison
    gene_x = 'Gene_5'
    gene_y = 'Gene_10'
    print(f"\n--- Generating scatter plot comparison for {gene_x} and {gene_y} ---")
    plot_scatter_comparison(
        original_df=original_expression_data,
        x_feature=gene_x,
        y_feature=gene_y,
        num_bins=20  # Hexbin grid size
    )

    # Heatmap comparison for the entire dataset
    print("\n--- Generating heatmap comparison for the dataset ---")
    plot_heatmap_comparison(
        original_df=original_expression_data,
        binned_df=binned_expression_data
    )