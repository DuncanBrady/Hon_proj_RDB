# File to load single cell RNA data to manipulate and prepare it for training a model.along with gain understanding the data structure.
import sys
import argparse
import scanpy as sc
import numpy as np 
import anndata as ad

#Function to generate numpy array from anndata object for training
def save_to_numpy(data, filePath):
    """
    Convert an AnnData object to a numpy array.
    
    Parameters:
    data (AnnData): The AnnData object to convert.
    
    Returns:
    np.ndarray: The numpy array representation of the data.
    """
    expression_matrix = data.X # Extract the expression matrix from the AnnData object
    if isinstance(expression_matrix, ad.AnnData):
        # If the expression matrix is sparse, convert it to a dense numpy array
        expression_matrix = expression_matrix.toarray()
    # Save the numpy array and the gene and cell names to a file
    try:
        np.save(filePath+"Integrated_matrix.npy", expression_matrix)
        np.save(filePath+"gene_names.npy", data.var_names.to_numpy())
        np.save(filePath+"cell_names.npy", data.obs_names.to_numpy())
    except Exception as e:
        print(f"Error saving numpy arrays: {e}")
        print("Ensure the file path is correct and you have write permissions.")

#function to save numpy array to file
def save_numpy_array(array, filename):
    """
    Save a numpy array to a file.
    
    Parameters:
    array (np.ndarray): The numpy array to save.
    filename (str): The path to the file where the array will be saved.
    """
    np.save(filename, array)


if __name__ == "__main__":
    try:
        parser = argparse.ArgumentParser(description="Load single cell RNA data from a txt file.")
        parser.add_argument("filename", type=str, help="Path to the txt file containing the data")
        args = parser.parse_args()

        data = sc.read_text(args.filename)
        data.var_names_make_unique()
        data.obs_names_make_unique()
        print(data)
        print(data.var_names)
        print(data.obs_names)
        print(f"data.n_obs: {data.n_obs}")
        print(f"data.n_vars: {data.n_vars}")
        print(f"data.obs.head(): {type(data.obs)}")

        exit(0)
    except Exception as e:
        print(f"Error loading data: {e}")
        sys.exit(1)