#!/usr/bin/env python3
"""
9. BIC"""
import numpy as np
expectation_maximization = __import__('8-EM').expectation_maximization


def BIC(X, kmin=1, kmax=None, iterations=1000, tol=1e-5, verbose=False):
    """finds the best number of clusters for a
    __GMM using the Bayesian Information Criterion:
    X is a numpy.ndarray of shape (n, d) containing the data set
    kmin is a positive integer containing the minimum
    __number of clusters to check for (inclusive)
    kmax is a positive integer containing the maximum
    __number of clusters to check for (inclusive)
    If kmax is None, kmax should be set to the maximum
    __number of clusters possible
    iterations is a positive integer containing the
    __maximum number of iterations for the EM algorithm
    tol is a non-negative float containing the tolerance
    __for the EM algorithm
    verbose is a boolean that determines if the EM algorithm
    __should print information to the standard output
    You may use at most 1 loop
    Returns: best_k, best_result, l, b, or
    __None, None, None, None on failure
    best_k is the best value for k based on its BIC
    best_result is tuple containing pi, m, S
    pi is a numpy.ndarray of shape (k,) containing the
    __cluster priors for the best number of clusters
    m is a numpy.ndarray of shape (k, d) containing the
    __centroid means for the best number of clusters
    S is a numpy.ndarray of shape (k, d, d) containing the
    __covariance matrices for the best number of clusters
    l is a numpy.ndarray of shape (kmax - kmin + 1) containing
    __the log likelihood for each cluster size tested
    b is a numpy.ndarray of shape (kmax - kmin + 1) containing
    __the BIC value for each cluster size tested
    Use: BIC = p * ln(n) - 2 * l
    p is the number of parameters required for the model
    n is the number of data points used to create the model
    l is the log likelihood of the model"""
    if (
        not isinstance(X, np.ndarray) or X.ndim != 2
        or not isinstance(kmin, int) or kmin <= 0
        or kmax is not None and (not isinstance(kmax, int) or kmax < kmin)
        or not isinstance(iterations, int) or iterations <= 0
        or isinstance(kmax, int) and kmax <= kmin
        or not isinstance(iterations, int) or iterations <= 0
        or not isinstance(tol, float) or tol < 0
        or not isinstance(verbose, bool)
    ):
        return None, None, None, None
    n, d = X.shape
    if kmax is None:
        kmax = n
    k_range = kmax - kmin + 1
    log_likelihoods = np.zeros(k_range)
    bic_values = np.zeros(k_range)
    best_k = None
    best_bic = np.inf
    best_result = None
    for i in range(k_range):
        k = kmin + i
        pi, m, S, _, log_likelihood = expectation_maximization(
            X, k, iterations, tol, verbose
        )
        if pi is None or m is None or S is None:
            return None, None, None, None
        p = k - 1 + k * d + k * d * (d + 1) / 2
        bic = p * np.log(n) - 2 * log_likelihood
        log_likelihoods[i] = log_likelihood
        bic_values[i] = bic
        if bic < best_bic:
            best_k = k
            best_bic = bic
            best_result = (pi, m, S)

    return best_k, best_result, log_likelihoods, bic_values
