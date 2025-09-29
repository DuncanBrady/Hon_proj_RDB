# file containing functions for dimensionality reduction and visualisation of high-dimensional data
import numpy as np


def pca_reduction(data, n_components=2):
    from sklearn.decomposition import PCA
    pca = PCA(n_components=n_components)
    reduced_data = pca.fit_transform(data)
    return reduced_data, pca.explained_variance_ratio_

def tsne_reduction(data, n_components=2, perplexity=30, n_iter=1000):
    from sklearn.manifold import TSNE
    tsne = TSNE(n_components=n_components, perplexity=perplexity, n_iter=n_iter)
    reduced_data = tsne.fit_transform(data)
    return reduced_data

def umap_reduction(data, n_components=2, n_neighbors=15, min_dist=0.1):
    import umap
    reducer = umap.UMAP(n_components=n_components, n_neighbors=n_neighbors, min_dist=min_dist)
    reduced_data = reducer.fit_transform(data)
    return reduced_data