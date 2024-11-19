#!/usr/bin/env python3
"""6. Expectation"""
import numpy as np
pdf = __import__('5-pdf').pdf


def expectation(X, pi, m, S):
    """calculates the expectation step in the EM algorithm for a GMM
    X is a numpy.ndarray of shape (n, d) containing the data set
    pi is a numpy.ndarray of shape (k,) containing the priors for each cluster
    m is a numpy.ndarray of shape (k, d) containing the centroid means for each cluster
    S is a numpy.ndarray of shape (k, d, d) containing the covariance matrices for each cluster
    """
    k = pi.shape[0]
    pdfs = np.array([pdf(X, m[i], S[i]) for i in range(k)])
    