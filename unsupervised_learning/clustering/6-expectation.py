#!/usr/bin/env python3
"""6. Expectation"""
import numpy as np
pdf = __import__('5-pdf').pdf


def expectation(X, pi, m, S):
    """calculates the expectation step in the EM algorithm for a GMM
    X is a numpy.ndarray of shape (n, d) containing the data set
    pi is a numpy.ndarray of shape (k,) containing the priors for each cluster
    m is a numpy.ndarray of shape (k, d) containing
    __the centroid means for each cluster
    S is a numpy.ndarray of shape (k, d, d) containing the
    __covariance matrices for each cluster
    You may use at most 1 loop
    Returns: g, l, or None, None on failure
    g is a numpy.ndarray of shape (k, n) containing the posterior
    __probabilities for each data point in each cluster
    l is the total log likelihood
    """
    try:
        n, d = X.shape
        k = pi.shape[0]
        g = np.zeros((k, n))

        # (pi_k * P_k(X)) / sum((pi_k * P_k(X)))
        for j in range(k):
            P_k = pdf(X, m[j], S[j])
            g[j] = pi[j] * P_k
        d = np.sum(g, axis=0)
        g /= d
        # l = sum(log( sum((pi_k * P_k(X)) ))
        l = np.sum(np.log(d))
        return g, l
    except Exception:
        return None, None
    