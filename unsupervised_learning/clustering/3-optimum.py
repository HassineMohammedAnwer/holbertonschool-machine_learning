#!/usr/bin/env python3
"""3. Optimize k"""

import numpy as np
kmeans = __import__('1-kmeans').kmeans
variance = __import__('2-variance').variance


def optimum_k(X, kmin=1, kmax=None, iterations=1000):
    """
    Tests for the optimum number of clusters by variance.
    - X (numpy.ndarray): 2D numpy array of shape (n, d) containing the dataset
    - kmin (int): Minimum number of clusters to check for (inclusive).
    - kmax (int): Maximum number of clusters to check for (inclusive).
    - iterations (int): Maximum number of iterations for K-means.
    Returns:
    - tuple: (results, d_vars), or (None, None) on failure.
        - results is a list containing the outputs of K-means for each
        cluster size.
        - d_vars is a list containing the difference in variance from the
        smallest cluster size for each cluster size.
    """
    if not isinstance(X, np.ndarray) or X.ndim != 2:
        return None, None
    if not isinstance(kmin, int) or kmin <= 0:
        return None, None
    if kmax is not None and (not isinstance(kmax, int) or kmax < kmin):
        return None, None
    if not isinstance(iterations, int) or iterations <= 0:
        return None, None

    if kmax is None:
        kmax = X.shape[0]

    results = []
    variances = []
    for k in range(kmin, kmax + 1):
        C, clss = kmeans(X, k, iterations=iterations)
        if C is None or clss is None:
            return None, None
        results.append((C, clss))
        variances.append(variance(X, C))
    min_variance = variances[0]
    d_vars = [min_variance - var for var in variances]

    return results, d_vars