import numpy as np 

import umap

import sys
sys.path.append('/home/pdomonkos/HYPERBOLIC/') #TODO: set path for Poincaré Maps https://github.com/facebookresearch/PoincareMaps/tree/main
from poincare_maps import PoincareMaps as PMAP


def fit_PCA(embeds, dimension):
    mean_embeds = embeds - np.mean(embeds , axis = 0)
    cov_mat = np.cov(mean_embeds , rowvar = False)
    eigen_values , eigen_vectors = np.linalg.eigh(cov_mat)
    sorted_index = np.argsort(eigen_values)[::-1]
    sorted_eigenvalue = eigen_values[sorted_index]
    sorted_eigenvectors = eigen_vectors[:,sorted_index]
    eigenvector_subset = sorted_eigenvectors[:,0:dimension]
    return np.dot(eigenvector_subset.transpose(),mean_embeds.transpose()).transpose()

def fit_UMAP(embeds, dimension, n_neighbors=5, min_dist=0.1, metric="cosine"):
    reducer = umap.UMAP(n_components=dimension, n_neighbors=n_neighbors, min_dist=min_dist, metric=metric, output_metric="euclidean", verbose=False)
    return reducer.fit_transform(embeds)

def fit_HUMAP(embeds, dimension, n_neighbors=5, min_dist=0.1, metric="cosine"):
    reducer = umap.UMAP(n_components=dimension, n_neighbors=n_neighbors, min_dist=min_dist, metric=metric, output_metric="hyperboloid", verbose=False)
    embeddings = reducer.fit_transform(embeds)
    z = np.sqrt(1 + np.sum(embeddings**2, axis=1))
    return embeddings / (1+np.expand_dims(z, 1))

def fit_PMAP(embeds, dimension, n_neighbors=5, learning_rate=1.0, metric="cosine"):
    embeddings = PMAP.compute_poincare_maps(embeds, n_components = dimension, mode='features', normalize=False, n_pca=0, distlocal=metric, 
                                        k_neighbours=n_neighbors, sigma=1.0, gamma=2.0,
                                        epochs = 2000, batchsize=256, lr=learning_rate, burnin=500, lrm=1.0, earlystop=0.0001, cuda=1)
    return embeddings