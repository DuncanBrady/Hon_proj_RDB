# This file contains the code for multiple approaches to binning the data.
# It is designed to be used in conjunction with the training script to preprocess data for training autoencoders.
# All methods expect that non-zero values are not present in the data
import numpy as np
data = "C:/Users/rdbra/Documents/honoursProject/code_base/data/sct_matrix_transposed.npz"

def term_freq_bin(data, num_bins):
    #Get all non-zero values from the data
    non_zero_mask = np.nonzero(data)
    non_zero_vals = data[non_zero_mask]
    if len(non_zero_vals) == 0:
        return data

    # Calculate the edges of the bins
    bin_edges = np.linspace(non_zero_vals.min(), non_zero_vals.max(), num_bins+1)
    print("Bin edges:", bin_edges)
    #Using the bin edges, mask the original data values into bin number
    bin_vals = np.digitize(non_zero_vals, bin_edges, right=False)
    bin_vals[non_zero_vals == non_zero_vals.max()] = num_bins
    data[non_zero_mask] = bin_vals
    return data

def k_means_bin(data, num_bins):
    # Get the total number of genes
    num_genes = data.shape[0]
    # Randomly select initial centroids
    centroids = data[np.random.choice(num_genes, num_bins, replace=False)]
    # Assign genes to the nearest centroid
    labels = np.argmin(np.linalg.norm(data[:, np.newaxis] - centroids, axis=2), axis=1)
    # Create bins based on labels
    bins = [data[labels == i] for i in range(num_bins)]
    return bins
