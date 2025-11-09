# This file contains the code for multiple approaches to binning the data.
# It is designed to be used in conjunction with the training script to preprocess data for training autoencoders.
# All methods expect that non-zero values are not present in the data
import numpy as np
import argparse

def term_freq_bin(data, num_bins):
    #Get all non-zero values from the data
    non_zero_mask = np.nonzero(data)
    non_zero_vals = data[non_zero_mask]
    #calculate percentage of values that are non-zero
    non_zero_percentage = len(non_zero_vals) / data.size * 100 if data.size > 0 else 0
    print(f"Percentage of non-zero values: {non_zero_percentage:.2f}%")
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

if __name__ == "__main__":
    # Example usage
    parser = argparse.ArgumentParser(description="Binning methods for data preprocessing")
    parser.add_argument('--data', type=str, required=True, help='Path to the input data file (numpy .npz format)')
    parser.add_argument('--num_bins', type=int, default=7, help='Number of bins to create')
    parser.add_argument('--method', type=str, choices=['term_freq', 'k_means'], default='term_freq', help='Binning method to use')
    parser.add_argument('--output', type=str, required=False, help='Path to save the binned data (numpy .npz format)')
    parser.add_argument('--transpose', action='store_true', help='Whether to transpose the data matrix before binning')
    args = parser.parse_args()  
    data = np.load(args.data)
    matrix_data = data['data'] if 'data' in data else data[list(data.files)[0]]
    
    if args.transpose: 
        print("Transposing data matrix, shape before:", matrix_data.shape)
        matrix_data = matrix_data.T
        print("Shape after transpose:", matrix_data.shape)
    binned_data = term_freq_bin(matrix_data.copy(), num_bins=5)
    print("Binning complete.")
    print("Original data shape:\n", matrix_data)
    print("Binned data (Term Frequency):\n", binned_data)
    #If output path is provided, add gene_ids and cell_ids if available
    if args.output:
        if 'genes' in data and 'cells' in data:
            np.savez_compressed(args.output, data=binned_data, genes=data['genes'], cells=data['cells'])
            print(f"Binned data saved to {args.output} with gene and cell IDs.")
        else:
            np.savez_compressed(args.output, data=binned_data)
            print(f"Binned data saved to {args.output}.")
