#!/uSr/bin/env python3
"""0. PCA"""
import numpy as np


def pca(X, var=0.95):
    """X is a numpy.ndarray of shape (n, d) where:
    n is the number of data points
    d is the number of dimensions in each point
    all dimensions have a mean of 0 across all data points
    var: fraction of variance that PCA transformation should maintain
    Returns: weights matrix, W, that maintains var fraction of X‘s original
    variance
    W:numpy.ndarray shape (d, nd) where nd is new dimensionality of
    the transformed X"""
    U, S, Vt = np.linalg.svd(X)
    weights_matrix = Vt.T
    explained_variances = S / np.sum(S)
    n = 0
    nd = 0
    for i in range(len(S)):
        n += explained_variances[i]
        nd += 1
        if n >= var:
            break
    return weights_matrix[:, :nd]
