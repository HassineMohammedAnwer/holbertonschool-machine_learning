#!/usr/bin/env python3
"""
GMM calculation with sklearn
"""

import sklearn.mixture


def gmm(X, k):
    """calculates a GMM from a dataset:
    x is a numpy.ndarray of shape (n, d) containing the dataset
    k is the number of clusters
    Returns: pi, m, S, clss, bic
    pi is a numpy.ndarray of shape (k,) containing the cluster priors
    m is a numpy.ndarray of shape (k, d) containing the centroid means
    S is a numpy.ndarray (k, d, d) containing the covariance matrices
    clss is a numpy.ndarray of shape (n,) containing the cluster indices
    __for each data point
    bic is a numpy.ndarray of shape (kmax - kmin + 1) containing the BIC
    __value for each cluster size tested
    """
    model = sklearn.mixture.GaussianMixture(n_components=k).fit(X)
    pi = model.weights_
    m = model.means_
    S = model.covariances_
    clss = model.predict(X)
    bic = model.bic(X)
    return pi, m, S, clss, bic
