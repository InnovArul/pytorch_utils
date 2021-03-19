# Originally from: https://github.com/facebookresearch/deepcluster
# This source code is licensed under the license found in the
# LICENSE file found in that repository (Attribution-NonCommercial 4.0 International)
# Modified

import time
import faiss
import numpy as np
from PIL import Image
from PIL import ImageFile
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from sklearn.preprocessing import normalize

def preprocess_features(npdata, pca=64, pca_info=None):
    """
    Preprocess an array of features.
    Args:
        npdata (np.array N * ndim): features to preprocess
        pca (int): dim of output
    Returns:
        np.array of dim N * pca: data PCA-reduced, whitened and L2-normalized
    """
    _, ndim = npdata.shape
    npdata = npdata.astype('float32')

    if pca_info is None:
        # Apply PCA-whitening with Faiss
        pca_matrix = faiss.PCAMatrix(ndim, pca, eigen_power=-0.5)
        pca_matrix.train(npdata)
        assert pca_matrix.is_trained
        npdata = pca_matrix.apply_py(npdata)

        pca_A = np.transpose(faiss.vector_to_array(pca_matrix.A).reshape((pca, ndim)))
        pca_b = faiss.vector_to_array(pca_matrix.b)
        pca_info = (pca_A, pca_b)
    else:
        npdata = np.dot(npdata, pca_info[0]) + pca_info[1]

    # L2 normalization
    row_sums = np.linalg.norm(npdata, axis=1)
    npdata = npdata / row_sums[:, np.newaxis]

    assert not np.isnan(npdata).any()
    assert not np.isinf(npdata).any()
    return npdata, pca_info


def get_index(distance, dim, use_gpu=True, temp_memory=500):
    """to get the appropriate index for a given distance

    Args:
        distance : cosine or euclidean distance
        dim (int): dimension of vectors
        use_gpu (bool, optional): whether to use the GPU. Defaults to True.

    Returns:
        faiss index: index 
    """    
    # define gpu resources
    if use_gpu:
        res = faiss.StandardGpuResources()
        res.setTempMemory(temp_memory)

        flat_config = faiss.GpuIndexFlatConfig()
        flat_config.useFloat16 = False
        flat_config.device = 0
        index = faiss.GpuIndexFlatL2(res, dim, flat_config) if distance == 'euclidean' \
                                                        else faiss.GpuIndexFlatIP(res, dim, flat_config)

    else:
        index = faiss.IndexFlatL2(dim) if distance == 'euclidean' else faiss.IndexFlatIP(dim)
    
    return index

def run_kmeans(x, nmb_clusters, distance, verbose=False, use_gpu=True):
    """Runs kmeans on 1 GPU.
    Args:
        x: data
        nmb_clusters (int): number of clusters
    Returns:
        list: ids of data in each cluster
    """
    n_data, d = x.shape

    # faiss implementation of k-means
    clus = faiss.Clustering(d, nmb_clusters)
    clus.niter = 20
    clus.max_points_per_centroid = 10000000

    index = get_index(distance, d, use_gpu)

    # perform the training
    clus.train(x, index)
    _, I = index.search(x, 1)
    centroids = faiss.vector_to_array(clus.centroids).reshape((nmb_clusters, d))  # Also return centroids!
    #losses = faiss.vector_to_array(clus.obj)

    stats = clus.iteration_stats
    losses = np.array([
        stats.at(i).obj for i in range(stats.size())
    ])

    if verbose:
        print('k-means loss evolution: {0}'.format(losses))

    return [int(n[0]) for n in I], losses, centroids, index


class Kmeans:
    """class used to cluster features.
        distance option: euclidean | cosine | norm_cosine

    Usage:

        km = Kmeans(k=100, distance='cosine')
        assignments, loss, centroids = km.cluster_features(data) # data = N x dim
        new_assignments = km.assign(new_data) # to assign labels to new data
        km.reset() # to clear memory
    """    
    def __init__(self, k, distance, normalize=False, pca_dim=-1):
        self.k = k
        self.index = None
        self.distance = distance
        print(f'Kmeans with k={k}, distance={distance}')
        self.pca_dim = pca_dim
        self.pca_info = None
        self.normalize = normalize
    
    def prep_data(self, data):
        #  L2 normalize if needed
        if self.normalize: 
            print('normalzing data for Kmeans or search')
            data = normalize(data, norm='l2')

        pca_info = None
        # perform pca if pca dim is less than data dim AND pca_dim os not -1
        if self.pca_dim != -1 and (data.shape[-1] > self.pca_dim):
            print(f"applying PCA ({data.shape[-1]} dim -> {self.pca_dim} dim)")
            data, pca_info = preprocess_features(data, pca=self.pca_dim, 
                                                pca_info=self.pca_info)

        return data, pca_info

    def cluster_features(self, data, verbose=False, use_gpu=True):
        """Performs k-means clustering.
            Args:
                x_data (np.array N * dim): data to cluster

            returns cluster indices for each feature
        """
        end = time.time()

        # reset the previous index
        self.reset()

        # PCA-reducing, whitening and L2-normalization
        # xb, pca_info = preprocess_features(data, pca=self.pca_dim)
        xb, pca_info = self.prep_data(data)

        # cluster the data
        assignments, loss, centroids, index = run_kmeans(xb, self.k,  distance=self.distance, 
                                                        verbose=verbose, use_gpu=use_gpu)

        # store index for future use
        self.index = index
        self.pca_info = pca_info

        if verbose:
            print('k-means time: {0:.0f} s'.format(time.time() - end))

        return assignments, loss, centroids

    def assign(self, data, centroids=None):
        """assign cluster ids to the given data max_points_per_centroid
        centroids parameter is given for convenience. In practice, after clustering,
        the index is saved in the class itself. So, only data input is required.

        Args:
            data (array): data array
            centroids (array, optional): cluster centroids. Defaults to None.

        Returns:
            cluster id array
        """        
        # use cached index if centroids are not given
        if centroids is None: index = self.index
        else:
            index = get_index(self.distance, data.shape[1], use_gpu=True)
            index.add(centroids)

        data, _ = self.prep_data(data)
        _, assignments = index.search(data, 1)

        # if centroids is given, clean up the index
        if centroids is not None:
            # release mem used by kmeans
            index.reset()
            del index

        return assignments

    def reset(self):
        if self.index is not None:
            # release mem used by kmeans
            self.index.reset()
            del self.index
            self.index = None