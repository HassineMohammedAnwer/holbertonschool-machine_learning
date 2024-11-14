#!/usr/bin/env python3
"""10. Hello, sklearn!"""
import sklearn.cluster


def kmeans(X, k):
    """ performs K-means on a dataset:
    X is a numpy.ndarray of shape (n, d) containing the dataset
    k is the number of clusters
    The only import you are allowed to use is import sklearn.cluster
    Returns: C, clss
    C is a numpy.ndarray of shape (k, d) containing the centroid
    __means for each cluster
    clss is a numpy.ndarray of shape (n,) containing the index of
    the cluster in C that each data point belongs to"""
    k_means = sklearn.cluster.KMeans(n_clusters=k).fit(X)
    C = k_means.cluster_centers_
    clss = k_means.labels_
    return C, clss
