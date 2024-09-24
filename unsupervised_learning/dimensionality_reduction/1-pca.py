#!/uSr/bin/env python3
"""PCA v2"""
import numpy as np


def pca(X, ndim):
    """X is a numpy.ndarray of shape (n, d) where:
    n is the number of data points
    d is the number of dimensions in each point
    We first subtract the mean to center data then Perform SVD
    then Select first ndim principal components
    then transform data in reduced space"""
    X_meaned = X - np.mean(X, axis=0)
    U, S, Vt = np.linalg.svd(X_meaned)
    W = Vt.T[:, :ndim]
    T = np.matmul(X_meaned, W)
    return T
