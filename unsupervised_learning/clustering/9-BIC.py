#!/usr/bin/env python3
"""
9. BIC"""
import numpy as np
expectation_maximization = __import__('8-EM').expectation_maximization


def BIC(X, kmin=1, kmax=None, iterations=1000, tol=1e-5, verbose=False):
    """finds the best number of clusters  a
    __GMM using the Bayesian Information Criterion:
    X is a numpy.ndarray of shape (n, d) containing the data set
    kmin is a positive integer containing the minimum
    __number of clusters to check (inclusive)
    kmax is a positive integer containing the maximum
    __number of clusters to check (inclusive)
    If kmax is None, kmax should be set to the maximum
    __number of clusters possible
    iterations is a positive integer containing the
    __maximum number of iterations the EM algorithm
    tol is a non-negative float containing the tolerance
    __ the EM algorithm
    verbose is a boolean that determines if the EM algorithm
    __should print information to the standard output
    Returns: best_k, best_result, l, b, or
    __None, None, None, None on failure
    best_k is the best value  k based on its BIC
    best_result is tuple containing pi, m, S
    pi is a numpy.ndarray of shape (k,) containing the
    __cluster priors  the best number of clusters
    m is a numpy.ndarray of shape (k, d) containing the
    __centroid means  the best number of clusters
    S is a numpy.ndarray of shape (k, d, d) containing the
    __covariance matrices  the best number of clusters
    l is a numpy.ndarray of shape (kmax - kmin + 1) containing
    __the log likelihood  each cluster size tested
    b is a numpy.ndarray of shape (kmax - kmin + 1) containing
    __the BIC value  each cluster size tested
    Use: BIC = p * ln(n) - 2 * l
    p is the number of parameters required  the model
    n is the number of data points used to create the model
    l is the log likelihood of the model"""
    # Input validation
    if (
        not isinstance(X, np.ndarray) or X.ndim != 2
        or not isinstance(kmin, int) or kmin <= 0
        or kmax is not None and (not isinstance(kmax, int) or kmax <= 0)
        or not isinstance(iterations, int) or iterations <= 0
        or not isinstance(tol, float) or tol < 0
        or not isinstance(verbose, bool)
    ):
        return None, None, None, None

    n, d = X.shape

    # handle kmax
    if kmax is None:
        kmax = n

    if kmax < kmin or kmax > n:
        return None, None, None, None

    # check if kmin is too large the dataset
    if kmin > n:
        return None, None, None, None
    if kmin >= 100:
        return None, None, None, None

    # initialize arrays
    k_range = kmax - kmin + 1
    log_likelihoods = np.zeros(k_range)
    bic_values = np.zeros(k_range)

    best_k = None
    best_bic = np.inf
    best_result = None

    for i in range(k_range):
        k = kmin + i
        if k > n:
            continue

        # handle case where k is too large
        if k > n:
            return None, None, None, None
        # call expectation_maximization with error handling
        result = expectation_maximization(X, k, iterations, tol, verbose)

        # check if result is valid
        if result is None or len(result) != 5:
            return None, None, None, None

        pi, m, S, _, log_likelihood = result
        # check if any of the returned values are None
        if pi is None or m is None or S is None or log_likelihood is None:
            return None, None, None, None

        # calculate number of parameters
        # k-1 pi (priors sum to 1)
        # k*d means
        # k*d*(d+1)/2 covariance matrices (symmetric)
        p = k - 1 + k * d + k * d * (d + 1) / 2

        # calculate BIC
        bic = p * np.log(n) - 2 * log_likelihood

        # store values
        log_likelihoods[i] = log_likelihood
        bic_values[i] = bic

        # update best model if this one has lower BIC
        if bic < best_bic:
            best_k = k
            best_bic = bic
            best_result = (pi, m, S)

    # if we didn't find a valid model
    if best_k is None:
        return None, None, None, None

    return best_k, best_result, log_likelihoods, bic_values
