#!/usr/bin/env python3
"""0. Initialize K-means"""

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
