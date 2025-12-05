# This file is used to aggregate preprocessing functions to allow easier data prep.
import binning
import genCoExp
import scanpy as sc


def load_data(file_path):
    data = np.load(file_path, allow_pickle=True)
    if "data" in data.files:
        exp_matrix = data["data"]
    else:
        exp_matrix = data[0]
    genes = data["genes"] if "genes" in data.files else None
    cells = data["cells"] if "cells" in data.files else None
    assert exp_matrix is not None, "Data could not be loaded properly."

    return (exp_matrix, genes, cells)

def preprocess_data(sc_adata, binning_method='term_freq', num_bins=7, coexp_method='spearman'):
    if binning_method == 'term_freq':
        sc_adata.X = binning.term_freq_binning(sc_adata.X, num_bins)
    elif binning_method == 'quantile':
        sc_adata.X = binning.quantile_binning(sc_adata.X, num_bins)
    else:
        raise ValueError(f"Unknown binning method: {binning_method}")

    if coexp_method == 'spearman':
        sc_adata.varp['coexp_matrix']= genCoExp.spearman_corr(sc_adata.layers['counts'])
    elif coexp_method == 'pearson':
        sc_adata.varp['coexp_matrix'] = genCoExp.pearson_corr(sc_adata.layers['counts'])
    elif coexp_method == 'covariance':
        sc_adata.varp['coexp_matrix'] = genCoExp.genCoVar(sc_adata.layers['counts'])
    else:
        raise ValueError(f"Unknown co-expression method: {coexp_method}")

    return binned_data, sc_adata.var['coexp_matrix']

def subset_top_expressed_genes(sc_data, top_n):
    sc_data.var['mean_expression'] = sc_data.X.mean(axis=0)
    top_n_index = sc_data.var.nlargest(top_n, 'mean_expression').index
    return sc_data[:, top_n_index]  # Subset the AnnData object
   

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Preprocess single-cell data.")
    parser.add_argument('--file', type=str, required=True, help='Path to the input data file.')
    parser.add_argument('--binning_method', type=str, default='term_freq', help='Binning method to use.')
    parser.add_argument('--num_bins', type=int, default=5, help='Number of bins for binning.')
    parser.add_argument('--coexp_method', type=str, default='spearman', help='Co-expression calculation method.')
    parser.add_argument('--top_n_genes', type=int, default=None, help='Number of highest expressed genes to select.')
    args = parser.parse_args()

    exp_matrix, genes, cells = load_data(args.file)
    # check if the data needs to be transposed into cells x genes
    if exp_matrix.shape[1] < exp_matrix.shape[0]:   
        exp_matrix = exp_matrix.T
    # Create AnnData object for Scanpy compatibility
    sc_adata = sc.AnnData(exp_matrix)
    sc_adata.var_names = genes if genes is not None else [f"Gene_{i}" for i in range(exp_matrix.shape[1])]
    sc_adata.obs_names = cells if cells is not None else [f"Cell_{i}" for i in range(exp_matrix.shape[0])]
    # If specified, select top N expressed genes based on mean expression
    if args.top_n_genes is not None:
        sc_adata = subset_top_expressed_genes(sc_adata, args.top_n_genes)
    sc_adata.layers['counts'] = sc_adata.X.copy()
    sc_adata = preprocess_data(sc_adata.X, args.binning_method, args.num_bins, args.coexp_method)

    save_data(binned_data, 'binned_data.npy')
    print("Preprocessing complete.")