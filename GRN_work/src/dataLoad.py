# File to load single cell RNA data to manipulate and prepare it for training a model.along with gain understanding the data structure.
import sys
import argparse
import scanpy as sc
import numpy as np 
import anndata as ad




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
        print(f"data.n_obs: {data.n_obs}")
        print(f"data.n_vars: {data.n_vars}")
        print(f"data.obs.head(): {data.obs.head()}")
        exit(0)
    except Exception as e:
        print(f"Error loading data: {e}")
        sys.exit(1)