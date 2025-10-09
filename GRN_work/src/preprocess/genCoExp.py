#This file contains the functions to process raw single-cell data into a co-variation matrix.
#Expected that data is already loaded into a numpy array of shape (cells, genes)
#file has commandline options to allow easy generation and saving of co-expression matrices from data files.
import sys
import os
import numpy as np
import scanpy as sc
import pandas as pd
import argparse
from scipy.stats import spearmanr, pearsonr

def parse_args():
    try:
        parser = argparse.ArgumentParser(description="Generate co-expression matrix from single-cell data.")
        parser.add_argument('-f', '--file', type=str, required=True, help='Path to the input single-cell data file (numpy .npz format).')
        parser.add_argument('-o', '--output', type=str, required=True, help='Path to save the output co-expression matrix (numpy .npy format).')
        parser.add_argument('--transpose', type=bool, help='Transpose the input data before processing.', default=None)
        parser.add_argument('--method', type=str, choices=['spearman', 'pearson', 'covariance'], default='spearman', help='Method to compute co-expression matrix.')
    except argparse.ArgumentError as err:
        print(str(err))
        sys.exit(2)
    return parser

def spearman_corr(sc_data, transpose=False):
    """
    Compute the Spearman correlation matrix for each pair of genes in the input matrix.
    Parameters:
        sc_data (numpy.ndarray): Input single-cell data of shape (cells, genes).
    returns:
        corr_matrix (numpy.ndarray): matrix containing correlation coefficients of shape (genes, genes), indexes are the same as input data.

    """
    if transpose:
        corr_matrix, _ = spearmanr(sc_data.T, axis=0)
    else:
        corr_matrix, _ = spearmanr(sc_data, axis=0)
    return corr_matrix


def pearson_corr(sc_data,transpose = False):  
    """
    Compute the Pearson correlation matrix for each pair of genes in the input matrix.
    Parameters:
        sc_data (numpy.ndarray): Input single-cell data of shape (cells, genes).
    returns:
        corr_matrix (numpy.ndarray): matrix containing correlation coefficients of shape (genes, genes), indexes are the same as input data.
    """
    if transpose:
        corr_matrix = pearsonr(sc_data.T, axis = 0)
    else:
        corr_matrix = pearsonr(sc_data, axis = 0)
    return corr_matrix

def genCoVar(sc_data):
    """
    Generate a co-variation matrix from single-cell data.

    Parameters:
    sc_data (numpy.ndarray): Input single-cell data of shape (cells, genes).

    Returns:
    numpy.ndarray: Co-variation matrix of shape (genes, genes).
    """
    # Convert to AnnData object for Scanpy processing
    adata = sc.AnnData(sc_data)
    
    # Normalize the data
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    
    # Compute the co-variation matrix
    cov_matrix = np.cov(adata.X, rowvar=False)
    
    return cov_matrix

if __name__ == "__main__":
    parser = parse_args()
    args = parser.parse_args()

    # Load the single-cell data
    data = np.load(args.file, allow_pickle=True)
    sc_data = data['data']  # Assuming the data is stored under the key 'data'
    # Create file name based on file name and method
    input_name = os.path.basename(args.file).split('.')[0]
    output_name = f"{input_name}_{args.method}_coexp.npy"

    # Check if a file with that name already exists and prompt user if they wish to continue
    if os.path.exists(args.output+output_name):
        print(f"Warning: A file named {args.output+output_name} already exists and will be overwritten.")
        response = input("Do you wish to continue? (y/n): ")
        valid_responses = ['y', 'n', 'Y', 'N']
        while response not in valid_responses:
            response = input("Invalid response. Please enter 'y' to continue or 'n' to exit: ")
        if response.lower() == 'y':
            print("Overwriting the existing file.")
        else:
            print("Exiting without overwriting the file.")
            sys.exit(0)

    # Check to see if the data is in the correct shape (cells, genes)
    if sc_data.shape[0] < sc_data.shape[1] and args.transpose is None:
        print("Warning: The Input data appears to be in the shape (genes, cells) instead of (cells, genes). Transposing the data if ")
        transpose = True
    else:   
        transpose = args.transpose  
    # Generate the co-expression matrix based on the selected method
    if args.method == 'spearman':
        coexp_matrix = spearman_corr(sc_data, transpose=transpose)
    elif args.method == 'pearson':
        coexp_matrix = pearson_corr(sc_data, transpose=transpose)
    elif args.method == 'covariance':
        coexp_matrix = genCoVar(sc_data)
    else:
        print(f"Error: Unknown method '{args.method}'.")
        sys.exit(2)

    # Save the co-expression matrix
    np.save(args.output+output_name, coexp_matrix)
    print(f"Co-expression matrix saved to {args.output+output_name}")