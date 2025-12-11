# This file is used to aggregate preprocessing functions to allow easier data prep.
import binning
import genCoExp
import scanpy as sc
import numpy as np



def load_data(file_path):
    data = np.load(file_path, allow_pickle=True)
    if "data" in data.files:
        exp_matrix = data["data"]
    else:
        exp_matrix = data[data.files[0]]  # Assume the first array in the npz file is the data
    genes = data["genes"] if "genes" in data.files else None
    cells = data["cells"] if "cells" in data.files else None
    assert exp_matrix is not None, "Data could not be loaded properly."

    return (exp_matrix, genes, cells)

def save_filtered_file(sc_adata, original_file, top_n_genes):
    # Save filtered data without preprocessing
    import os
    input_name = os.path.splitext(original_file)[0]
    filtered_output_name = f"{input_name}_top{top_n_genes}.npz"
    np.savez_compressed(filtered_output_name, data=sc_adata.X, genes=sc_adata.var_names, cells=sc_adata.obs_names)
    print(f"Filtered data saved to {filtered_output_name}")
    return filtered_output_name

def save_files(sc_adata, original_file, binning_method, coexp_method, num_bins, top_n_genes):
    # Save two npz files: binned data and co-expression matrix
    # Create file name based on original file name and methods used
    # binned data should save binned matrix, genes, cells
    # co-expression matrix should save the co-expression matrix, genes
    # Get full input file name without extension
    import os
    input_name = os.path.splitext(original_file)[0]
    binning_suffix = f"{binning_method}_bins{num_bins}"
    if top_n_genes is not None:
        binning_suffix += f"_top{top_n_genes}"
        coexp_suffix = f"{coexp_method}_top{top_n_genes}"
    else:
        coexp_suffix = coexp_method
    binned_output_name = f"{input_name}_{binning_suffix}.npz"
    coexp_output_name = f"{input_name}_{coexp_suffix}.npz"
    np.savez_compressed(binned_output_name, data=sc_adata.X, genes=sc_adata.var_names, cells=sc_adata.obs_names)
    print(f"Binned data saved to {binned_output_name}")
    if coexp_method is not None:
        np.savez_compressed(coexp_output_name, coexp_matrix=sc_adata.varp['coexp_matrix'], genes=sc_adata.var_names)
        print(f"Co-expression matrix saved to {coexp_output_name}")

def bin_data(exp_matrix, binning_method='term_freq', num_bins=7):
    if binning_method == 'term_freq':
        binned_data = binning.term_freq_bin(exp_matrix, num_bins)
    elif binning_method == 'k_means':
        binned_data = binning.k_means_bin(exp_matrix, num_bins)
    else:
        raise ValueError(f"Unknown binning method: {binning_method}")
    return binned_data

def generate_coexp_matrix(exp_matrix, coexp_method='spearman'):
    if coexp_method == 'spearman':
        coexp_matrix = genCoExp.spearman_corr(exp_matrix)
    elif coexp_method == 'pearson':
        coexp_matrix = genCoExp.pearson_corr(exp_matrix)
    elif coexp_method == 'covariance':
        coexp_matrix = genCoExp.genCoVar(exp_matrix)
    else:
        raise ValueError(f"Unknown co-expression method: {coexp_method}")
    return coexp_matrix

def preprocess_data(sc_adata, binning_method='term_freq', num_bins=7, coexp_method=None):
    # Bin the data
    sc_adata.X = bin_data(sc_adata.X, binning_method, num_bins)
    if coexp_method is not None:
        sc_adata.varp['coexp_matrix'] = generate_coexp_matrix(sc_adata.layers['counts'], coexp_method)
    return sc_adata

def subset_top_expressed_genes(sc_data, top_n):
    sc_data.var['mean_expression'] = sc_data.X.mean(axis=0)
    top_n_index = sc_data.var.nlargest(top_n, 'mean_expression').index
    return sc_data[:, top_n_index]  # Subset the AnnData object
   

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Preprocess single-cell data.")
    parser.add_argument('--file', type=str, required=True, help='Path to the input data file.')
    parser.add_argument('--binning_method', type=str, default=None, help='Binning method to use.')
    parser.add_argument('--num_bins', type=int, default=7, help='Number of bins for binning.')
    parser.add_argument('--coexp_method', type=str, default=None, help='Co-expression calculation method.')
    parser.add_argument('--top_n_genes', type=int, default=None, help='Number of highest expressed genes to select.')
    parser.add_argument('--filter_only', action='store_true', help='Only filter by top N genes without preprocessing.')
    args = parser.parse_args()

    exp_matrix, genes, cells = load_data(args.file)
    # check if the data needs to be transposed into cells x genes
    if exp_matrix.shape[0] < exp_matrix.shape[1]:   
        exp_matrix = exp_matrix.T
    # Create AnnData object for Scanpy compatibility
    sc_adata = sc.AnnData(exp_matrix)
    sc_adata.var_names = genes if genes is not None else [f"Gene_{i}" for i in range(exp_matrix.shape[1])]
    sc_adata.obs_names = cells if cells is not None else [f"Cell_{i}" for i in range(exp_matrix.shape[0])]
    
    # If specified, select top N expressed genes based on mean expression
    if args.top_n_genes is not None:
        sc_adata = subset_top_expressed_genes(sc_adata, args.top_n_genes)
    
    # If filter_only flag is set, save filtered data and exit
    if args.filter_only:
        if args.top_n_genes is None:
            raise ValueError("--filter_only requires --top_n_genes to be specified.")
        save_filtered_file(sc_adata, args.file, args.top_n_genes)
        print("Filtering complete.")
    else:
        # Perform full preprocessing
        if args.binning_method is None:
            args.binning_method = 'term_freq'
        sc_adata.layers['counts'] = sc_adata.X.copy()
        sc_adata = preprocess_data(sc_adata, args.binning_method, args.num_bins, args.coexp_method)
        save_files(sc_adata, args.file, args.binning_method, args.coexp_method, args.num_bins, args.top_n_genes)
        print("Preprocessing complete.")