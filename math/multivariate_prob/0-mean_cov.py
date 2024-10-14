#!/usr/bin/env python3
"""0. Mean and Covariance"""
import numpy as np


def mean_cov(X):
    """calculates the mean and covariance of a data set"""
    n, d = X.shape
    if type(X) != np.ndarray or len(X.shape) != 2:
        raise TypeError("X must be a 2D numpy.ndarray")
    if n < 2:
        raise ValueError("X must contain multiple data points")
    mean = np.mean(X, axis=0).reshape(1, d)
    tmp = X - mean
    cov = np.dot(tmp.T, tmp) / (n - 1)
    return mean, cov
