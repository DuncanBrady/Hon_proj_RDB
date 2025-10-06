#This file contains the functions to process raw single-cell data into a co-variation matrix.
#Expected that data is already loaded into a numpy array of shape (cells, genes)
import numpy as np
import scanpy as sc
import pandas as pd
from scipy.stats import spearmanr


def spearman_corr(sc_data, column_names= "genes", row_names= "cells", transpose=False):
    """
    Compute the Spearman correlation matrix for each pair of genes in the input matrix.
    Parameters:
        sc_data (numpy.ndarray): Input single-cell data of shape (cells, genes).
    returns:
        corr_matrix (numpy.ndarray): matrix containing correlation coefficients of shape (genes, genes), indexes are the same as input data.

    """
    if transpose:
        corr_matrix, _ = spearmanr(sc_data.T, axis=0)
    corr_matrix, _ = spearmanr(sc_data, axis=0)
    return corr_matrix


def pearson_corr(sc_data):  
    """
    Compute the Pearson correlation matrix for each pair of genes in the input matrix.
    Parameters:
        sc_data (numpy.ndarray): Input single-cell data of shape (cells, genes).
    returns:
        corr_matrix (numpy.ndarray): matrix containing correlation coefficients of shape (genes, genes), indexes are the same as input data.
    """
    corr_matrix = np.corrcoef(sc_data, rowvar=False)
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

