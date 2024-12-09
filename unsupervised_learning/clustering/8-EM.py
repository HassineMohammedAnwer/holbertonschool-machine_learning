#!/usr/bin/env python3
"""8. EM"""
import numpy as np
initialize = __import__('4-initialize').initialize
expectation = __import__('6-expectation').expectation
maximization = __import__('7-maximization').maximization


def expectation_maximization(X, k, iterations=1000, tol=1e-5, verbose=False):
    """performs the expectation maximization for a GMM:
    X is a numpy.ndarray of shape (n, d) containing the data set
    k is a positive integer containing the number of clusters
    iterations is a positive integer containing the maximum number of
    __iterations for the algorithm
    tol is a non-negative float containing tolerance of the log likelihood,used
    __to determine early stopping i.e. if the difference is less than or equal
    __to tol you should stop the algorithm
    verbose is a boolean that determines if you should print information about
    __the algorithm
    If True, print Log Likelihood after {i} iterations: {l} every 10 iterations
    __and after the last iteration
    {i} is the number of iterations of the EM algorithm
    {l} is the log likelihood, rounded to 5 decimal places
    You may use at most 1 loop
    Returns: pi, m, S, g, l, or None, None, None, None, None on failure
    pi is a numpy.ndarray of shape (k,) containing the priors for each cluster
    m is a numpy.ndarray of shape (k, d) containing the centroid means for
    __each cluster
    S is a numpy.ndarray of shape (k, d, d) containing the covariance matrices
    __for each cluster
    g is a numpy.ndarray of shape (k, n) containing the probabilities for each
    __data point in each cluster
    l is the log likelihood of the model
    """
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None, None, None, None
    if not isinstance(k, int) or k <= 0:
        return None, None, None, None, None
    if not isinstance(iterations, int) or iterations <= 0:
        return None, None, None, None, None
    if not isinstance(tol, float) or tol < 0:
        return None, None, None, None, None
    if not isinstance(verbose, bool):
        return None, None, None, None, None
    n, d = X.shape
    pi, m, S = initialize(X, k)
    prev_l = 0
    g, l = expectation(X, pi, m, S)
    for i in range(iterations):
        if verbose and i % 10 == 0:
            print("Log Likelihood after {} iterations: {}"
                  .format(i, l.round(5)))
        pi, m, S = maximization(X, g)
        g, l = expectation(X, pi, m, S)
        if prev_l is not None and abs(l - prev_l) <= tol:
            break
        prev_l = l
    if verbose:
        print("Log Likelihood after {} iterations: {}"
              .format(i + 1, l.round(5)))
    return pi, m, S, g, l
