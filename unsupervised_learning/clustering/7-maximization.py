#!/usr/bin/env python3
"""7. Maximization"""
import numpy as np


def maximization(X, g):
    """calculates the maximization step in the EM algorithm for a GMM:
    updates the model parameters using the posterior probabilities
    (responsibilities) computed in the Expectation step
    X is a numpy.ndarray of shape (n, d) containing the data set
    g is a numpy.ndarray of shape (k, n) containing the posterior
    __probabilities for each data point in each cluster
    You may use at most 1 loop
    Returns: pi, m, S, or None, None, None on failure
    pi is a numpy.ndarray of shape (k,) containing the updated
    __priors for each cluster
    m is a numpy.ndarray of shape (k, d) containing the updated
    __centroid means for each cluster
    S is a numpy.ndarray of shape (k, d, d) containing the updated
    __covariance matrices for each cluster
    """
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None, None
    if not isinstance(g, np.ndarray) or len(g.shape) != 2:
        return None, None, None
    n, d = X.shape
    k, nn = g.shape
    if n != nn:
        return None, None, None
    # Priors
    N_k = np.sum(g, axis=1)
    if N_k != n:
        return None, None, None
    pi = N_k / n
    # Mean
    m = np.zeros((k, d))
    # Covariance
    S = np.zeros((k, d, d))
    for i in range(k):
        m[i] = np.matmul(g[i], X) / np.sum(g[i])
        X_cent = X - m[i]
        S[i] = np.matmul(np.multiply(g[i], X_cent.T), X_cent) / np.sum(g[i])
    return pi, m, S
