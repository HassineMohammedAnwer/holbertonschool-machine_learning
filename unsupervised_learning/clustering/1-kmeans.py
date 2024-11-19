#!/usr/bin/env python3
"""1. kmeans"""

import numpy as np


def initialize(X, k):
    """initializes cluster centroids for K-means:
    X is a numpy.ndarray of shape (n, d) containing
    __the dataset that will be used for K-means clustering
    n is the number of data points
    d is the number of dimensions for each data point
    k is a positive integer containing the number of clusters
    The cluster centroids should be initialized with a multivariate
    __uniform distribution along each dimension in d:
    The minimum values for the distribution should be the minimum
    __values of X along each dimension in d
    The maximum values for the distribution should be the maximum
    __values of X along each dimension in d
    You should use numpy.random.uniform exactly once
    You are not allowed to use any loops
    Returns: a numpy.ndarray of shape (k, d) containing the initialized
    __centroids for each cluster, or None on failure
    """
    if not isinstance(k, int) or k <= 0:
        return None
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None
    n, d = X.shape
    low = np.min(X, axis=0)
    high = np.max(X, axis=0)
    centroids = np.random.uniform(low=low, high=high, size=(k, d))
    return centroids

def kmeans(X, k, iterations=1000):
    """performs K-means on a dataset:
    X is a numpy.ndarray of shape (n, d) containing the dataset
    n is the number of data points
    d is the number of dimensions for each data point
    k is a positive integer containing the number of clusters
    iterations is a positive integer containing the maximum number
    __of iterations that should be performed
    If no change in the cluster centroids occurs between iterations,
    __your function should return
    Initialize the cluster centroids using a multivariate uniform
    __distribution (based on0-initialize.py)
    If a cluster contains no data points during the update step,
    __reinitialize its centroid
    You should use numpy.random.uniform exactly twice
    You may use at most 2 loops
    Returns: C, clss, or None, None on failure
    C is a numpy.ndarray of shape (k, d) containing the centroid means for each cluster
    clss is a numpy.ndarray of shape (n,) containing the index of the cluster in C that each data point belongs to
    """
    if not isinstance(k, int) or k <= 0:
        return None
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None
    centroids = initialize(X, k)
    for _ in range(iterations):
        prev_ctds = np.copy(centroids)
        # Euclidean
        euc = np.sqrt(np.sum((X - centroids[:, np.newaxis]) ** 2, axis=2))
        clss = np.argmin(euc, axis=0)
        # Update centroids
        for i in range(k):
            points_in_cluster = X[clss == i]
            if len(points_in_cluster) == 0:
                centroids[i] = initialize(X, 1)
            else:
                centroids[i] = np.mean(points_in_cluster, axis=0)
        if np.allclose(centroids, prev_ctds):
            break
        prev_ctds = np.copy(centroids)
        euc = np.sqrt(np.sum((X - centroids[:, np.newaxis]) ** 2, axis=2))
        clss = np.argmin(euc, axis=0)

    return centroids, clss
