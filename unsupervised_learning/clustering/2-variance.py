#!/usr/bin/env python3
"""2. Variance"""
import numpy as np


def variance(X, C):
    """calculates the total intra-cluster variance for a data set:
    X is a numpy.ndarray of shape (n, d) containing the data set
    C is a numpy.ndarray of shape (k, d) containing the centroid
    __means for each cluster
    You are not allowed to use any loops
    Returns: var, or None on failure
    var is the total variance"""
    if not isinstance(X, np.ndarray) or X.ndim != 2:
        return None
    if not isinstance(C, np.ndarray) or C.ndim != 2:
        return None
    if X.shape[1] != C.shape[1]:
        return None
    distances = np.linalg.norm(X[:, np.newaxis] - C, axis=2)
    min_dist = np.min(distances, axis=1)
    var = np.sum(min_dist ** 2)
    return var